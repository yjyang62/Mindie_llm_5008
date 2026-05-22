/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "single_req_tgi_text_infer_interface.h"
#include "single_llm_pnd_req_handler.h"
#include "request.h"
#include "response.h"
#include "mock_util.h"
#include "base64_util.h"

using namespace mindie_llm;
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {

class TgiTextInferTest : public testing::Test {
protected:
    void SetUp() override
    {
        httpRequest = httplib::Request();
        httpResponse = httplib::Response();
        request = std::make_shared<Request>(RequestIdNew("mockRequest"));
        response = std::make_shared<Response>(RequestIdNew("mockResponse"));
        requestContext = std::make_shared<RequestContext>(httpRequest, httpResponse);
        pndReqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
        tgiInferInterface = std::make_shared<SingleReqTgiTextInferInterface>(pndReqHandler, false);
        tgiInferInterface->inputParam = std::make_shared<InferParam>();
        tgiInferInterface->request_ = std::make_shared<Request>();
    }
    void TearDown() override
    {
        GlobalMockObject::reset();
    }

    httplib::Request httpRequest;
    httplib::Response httpResponse;
    RequestSPtr request;
    ResponseSPtr response;
    std::string error;
    std::shared_ptr<RequestContext> requestContext;
    std::shared_ptr<SingleLLMPnDReqHandler> pndReqHandler;
    std::shared_ptr<SingleReqTgiTextInferInterface> tgiInferInterface;
};

TEST_F(TgiTextInferTest, testValidateAndPrepareReqToken)
{
    OrderedJson body;
    std::string errorMsg, input;
    uint64_t timestamp;
    auto serverConfig = ConfigManager::GetInstance().GetServerConfig();
    uint32_t maxInputLen = serverConfig.maxRequestLength * 1024 * 1024;

    auto stubs = MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool*, const std::string&,
        std::vector<int64_t>&, HeadFlag, uint64_t&)).stubs();

    // valid values
    //      should return false given tokenizer failed
    body["inputs"] = "mockInputs";
    stubs.will(invoke(&MockTokenizerEncodeFail));
    EXPECT_FALSE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return true given tokenize success
    stubs.will(invoke(&MockTokenizerEncodeSuccess));
    EXPECT_TRUE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    EXPECT_EQ(tgiInferInterface->inputParam->textInput, "mockInputs");
    EXPECT_EQ(tgiInferInterface->reqTokens_, (std::vector<int64_t>{1, 2, 3}));
    body["inputs"] = OrderedJson::array({
        {"image_url", "http://example.com/image1.jpg"},
        {"image_url", "https://example.com/image2.png"},
        {"video_url", "https://example.com/video.mp4"}
    });
    EXPECT_TRUE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    EXPECT_EQ(tgiInferInterface->inputParam->textInput, "[[\"image_url\",\"http://example.com/image1.jpg\"],[\"image_url\",\"https://example.com/image2.png\"],[\"video_url\",\"https://example.com/video.mp4\"]]");

    // corner case
    //      should return true given input length is within limit
    input = "a";
    input.resize(maxInputLen);
    body["inputs"] = input;
    EXPECT_TRUE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));

    // invalid values
    //      should return false given input is empty
    stubs.will(invoke(&MockTokenizerEncodeSuccess));
    body.clear();
    EXPECT_FALSE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given input is an empty array
    body["inputs"] = OrderedJson::array();
    EXPECT_FALSE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given input type is of invalid
    body["inputs"] = 1;
    EXPECT_FALSE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given input length is out of limit
    input = "a";

    input.resize(maxInputLen + 1);
    body["inputs"] = input;
    EXPECT_FALSE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));

    // recompute
    //      round 1: mock the previous round before recompute
    body["inputs"] = "mockInputsRoundOne";
    stubs.will(invoke(&MockTokenizerEncodeSuccess));
    EXPECT_TRUE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    EXPECT_EQ(tgiInferInterface->inputParam->textInput, "mockInputsRoundOne");
    EXPECT_EQ(tgiInferInterface->reqTokens_, (std::vector<int64_t>{1, 2, 3}));
    //      round 2: mock recompute
    //      should return true given origin_inputs is not empty, regardless that tokenizer fails
    tgiInferInterface->isReCompute_ = true;
    body["origin_inputs"] = body["inputs"];
    stubs.will(invoke(&MockTokenizerEncodeFail));
    EXPECT_FALSE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given origin_inputs is empty, regardless that tokenizer succeeds
    stubs.will(invoke(&MockTokenizerEncodeSuccess));
    body["origin_inputs"] = "";
    EXPECT_FALSE(tgiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
};

TEST_F(TgiTextInferTest, testSetupInferParams)
{
    // valid values
    tgiInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "inputs": "mock test",
        "parameters": {
            "decoder_input_details": true,
            "details": true,
            "do_sample": true,
            "max_new_tokens": 20,
            "repetition_penalty": 1.03,
            "return_full_text": false,
            "seed": 1234,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "truncate": 2,
            "typical_p": 0.5,
            "watermark": false,
            "stop": ["test"],
            "adapter_id": "mockLora"
        }
    })");
    //      should return true given valid reqJsonBody
    std::string errorMsg;
    EXPECT_EQ(tgiInferInterface->SetupInferParams(request, errorMsg), true);
    //      validate request's items with reqJsonBody
    //      input_ids, maxOutputLen are not validated since they are assigned outside SetupInferParams
    EXPECT_EQ(tgiInferInterface->inputParam->decoderInputDetails, true);
    EXPECT_EQ(*request->stopStrings, Base64Util::Encode("[\"test\"]"));
    EXPECT_TRUE(request->temperature.has_value());
    EXPECT_NEAR(request->temperature.value(), 0.5, 1e-6);
    EXPECT_EQ(request->topK.value(), 10);
    EXPECT_NEAR(request->topP.value(), 0.95, 1e-6);
    EXPECT_NEAR(request->typicalP.value(), 0.5, 1e-6);
    EXPECT_EQ(request->doSample.value(), true);
    EXPECT_EQ(request->seed, 1234);
    EXPECT_NEAR(request->repetitionPenalty.value(), 1.03, 1e-6);
    EXPECT_EQ(request->watermark, false);
    EXPECT_EQ(request->loraId, "mockLora");
    EXPECT_EQ(tgiInferInterface->inputParam->maxNewTokens, 20);
    EXPECT_EQ(tgiInferInterface->inputParam->showDetails, true);
    EXPECT_EQ(tgiInferInterface->inputParam->decoderInputDetails, true);
    EXPECT_EQ(tgiInferInterface->inputParam->returnFullText, false);
    EXPECT_EQ(tgiInferInterface->inputParam->truncate, 2);
    
    // invalid values
    //      should return false given truncate exceeds limits
    tgiInferInterface->reqJsonBody_["parameters"]["truncate"] = 0;
    EXPECT_EQ(tgiInferInterface->SetupInferParams(request, errorMsg), false);
    //      should return false given repetition_penalty exceeds limits
    tgiInferInterface->reqJsonBody_["parameters"]["repetition_penalty"] = -1;
    EXPECT_EQ(tgiInferInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(TgiTextInferTest, testProcess)
{
    MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool*, const std::string&,
        std::vector<int64_t>&, HeadFlag, uint64_t&, const bool)).stubs().will(invoke(&MockTokenizerEncodeSuccess));
    
    // HTTP request
    httpRequest.method = "mockMethod";
    httpRequest.path = "mockPath";
    httpRequest.version = "mockVersion";
    httpRequest.body =  R"({
        "inputs": "mock test",
        "parameters": {
            "decoder_input_details": true, "details": true, "do_sample": true, "max_new_tokens": 20,
            "repetition_penalty": 1.03, "return_full_text": false, "seed": 1234, "temperature": 0.5,
            "top_k": 10, "top_p": 0.95, "truncate": 5, "typical_p": 0.5, "watermark": false,
            "stop": ["test"], "adapter_id": "None", "priority": 1, "timeout": 1,
            "presence_penalty": 0, "frequency_penalty": 0, "length_penalty": 1
    }})";
    requestContext = std::make_shared<RequestContext>(httpRequest, httpResponse);
    pndReqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
    tgiInferInterface = std::make_shared<SingleReqTgiTextInferInterface>(pndReqHandler, false);
    // InferInterface::Process: HTTP request => JSON format => MindIE Request
    EXPECT_NO_THROW(tgiInferInterface->Process());
    // check values
}

TEST_F(TgiTextInferTest, testBuildResponseJson)
{
    // valid values
    tgiInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "inputs": "mock test",
        "parameters": {
            "decoder_input_details": true,
            "details": true,
            "do_sample": true,
            "max_new_tokens": 20,
            "repetition_penalty": 1.03,
            "return_full_text": false,
            "seed": 1234,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "truncate": 2,
            "typical_p": 0.5,
            "watermark": false,
            "stop": ["test"],
            "adapter_id": "mockLora"
        }
    })");
    std::string errorMsg;
    EXPECT_EQ(tgiInferInterface->SetupInferParams(request, errorMsg), true);
    tgiInferInterface->Process();
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    tgiInferInterface->respTokenMap[0] = {0};
    EXPECT_TRUE(tgiInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    // invalid values
    //      should return false ProcessResponseSingle fails
    MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseSingle, bool (*)(ResponseSPtr, const uint64_t &))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(tgiInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
}

TEST_F(TgiTextInferTest, TestSendStreamResponse)
{
    tgiInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "inputs": "mock test",
        "parameters": {
            "decoder_input_details": true,
            "details": true,
            "do_sample": true,
            "max_new_tokens": 20,
            "repetition_penalty": 1.03,
            "return_full_text": false,
            "seed": 1234,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "truncate": 2,
            "typical_p": 0.5,
            "watermark": false,
            "stop": ["test"],
            "adapter_id": "mockLora"
        }
    })");
    std::string errorMsg;
    EXPECT_EQ(tgiInferInterface->SetupInferParams(request, errorMsg), true);
    RespBodyQueue jsonStrings;
    auto stubs = MOCKER_CPP(&SingleReqInferInterfaceBase::GetAvailableOutputCache,
                            bool (*)(std::vector<SingleReqInferInterfaceBase::StreamCache> &))
                     .stubs();
    stubs.will(returnValue(false));
    tgiInferInterface->respTokenMap[0] = {0};
    tgiInferInterface->SendStreamResponse(jsonStrings);
    stubs.will(returnValue(true));
    tgiInferInterface->SendStreamResponse(jsonStrings);
}

TEST_F(TgiTextInferTest, TestChangeUtf8Str)
{
    std::string result;
    // valid values
    std::string input = "";
    result = tgiInferInterface->ChangeUtf8Str(input);
    EXPECT_EQ(result, "");
    input = "abc";
    result = tgiInferInterface->ChangeUtf8Str(input);
    EXPECT_EQ(result, "abc");
    // invalid value
    //      invalid characters are filtered out
    input = "before";
    input += static_cast<char>(0x00);
    input += static_cast<char>(0x01);
    input += "middle";
    input += static_cast<char>(0x1F);
    input += "after";
    result = tgiInferInterface->ChangeUtf8Str(input);
    EXPECT_EQ(result, "beforemiddleafter");
}

TEST_F(TgiTextInferTest, testBuildTgiReComputeBody)
{
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    request->doSample = true;
    request->repetitionPenalty = 1.11f;
    request->temperature = 0.75f;
    request->topK = 50;
    request->topP = 0.9f;
    request->seed = static_cast<uint64_t>(2025);
    request->typicalP = 0.7f;
    request->watermark = false;
    request->loraId = "mockLora";

    auto inferParam = std::make_shared<InferParam>();
    inferParam->textInput = "hello world";
    inferParam->showDetails = true;
    inferParam->streamMode = true;
    inferParam->returnFullText = true;
    inferParam->decoderInputDetails = false;
    inferParam->maxNewTokens = 64;

    tgiInferInterface->inputParam = inferParam;
    tgiInferInterface->request_ = request;
    tgiInferInterface->isReCompute_ = true;
    tgiInferInterface->truncate = 128;
    request->stopStrings = std::string(R"(["\n\n","</s>"])");

    std::vector<BestNTokens> emptyTokens;
    std::string ret = tgiInferInterface->BuildTgiReComputeBody(emptyTokens);

    OrderedJson obj = OrderedJson::parse(ret);
    ASSERT_TRUE(obj.contains("inputs"));
    EXPECT_TRUE(obj["inputs"].is_string());
    ASSERT_TRUE(obj.contains("origin_inputs"));
    EXPECT_EQ(obj["origin_inputs"].get<std::string>(), "hello world");
    ASSERT_TRUE(obj.contains("stream"));
    EXPECT_TRUE(obj["stream"].get<bool>());
    ASSERT_TRUE(obj.contains("parameters"));
    const auto &p = obj["parameters"];
    ASSERT_TRUE(p.contains("details"));
    EXPECT_TRUE(p["details"].get<bool>());
    ASSERT_TRUE(p.contains("return_full_text"));
    EXPECT_TRUE(p["return_full_text"].get<bool>());
    ASSERT_TRUE(p.contains("decoder_input_details"));
    EXPECT_FALSE(p["decoder_input_details"].get<bool>());
    ASSERT_TRUE(p.contains("max_new_tokens"));
    EXPECT_EQ(p["max_new_tokens"].get<int32_t>(), 64);
    ASSERT_TRUE(p.contains("truncate"));
    EXPECT_EQ(p["truncate"].get<int32_t>(), 128);
    ASSERT_TRUE(p.contains("adapter_id"));
    EXPECT_EQ(p["adapter_id"].get<std::string>(), "mockLora");
    ASSERT_TRUE(p.contains("do_sample"));
    EXPECT_TRUE(p["do_sample"].get<bool>());
    ASSERT_TRUE(p.contains("repetition_penalty"));
    EXPECT_NEAR(p["repetition_penalty"].get<double>(), 1.11, 1e-6);
    ASSERT_TRUE(p.contains("temperature"));
    EXPECT_NEAR(p["temperature"].get<double>(), 0.75, 1e-6);
    ASSERT_TRUE(p.contains("top_k"));
    EXPECT_EQ(p["top_k"].get<int32_t>(), 50);
    ASSERT_TRUE(p.contains("top_p"));
    EXPECT_NEAR(p["top_p"].get<double>(), 0.9, 1e-6);
    ASSERT_TRUE(p.contains("seed"));
    EXPECT_EQ(p["seed"].get<uint64_t>(), static_cast<uint64_t>(2025));
    ASSERT_TRUE(p.contains("typical_p"));
    EXPECT_NEAR(p["typical_p"].get<double>(), 0.7, 1e-6);
    ASSERT_TRUE(p.contains("watermark"));
    EXPECT_FALSE(p["watermark"].get<bool>());
    ASSERT_TRUE(p.contains("stop"));
    ASSERT_TRUE(p["stop"].is_array());
    ASSERT_EQ(p["stop"].size(), 2u);
    EXPECT_EQ(p["stop"][0].get<std::string>(), "\n\n");
    EXPECT_EQ(p["stop"][1].get<std::string>(), "</s>");
}

TEST_F(TgiTextInferTest, TestEncodeTGIResponse)
{
    std::shared_ptr<SingleReqGeneralTgiTextInferInterface> generalTgiInferInterface;
    generalTgiInferInterface = std::make_shared<SingleReqGeneralTgiTextInferInterface>(pndReqHandler, false);
    generalTgiInferInterface->request_ = std::make_shared<Request>();
    RespBodyQueue jsonStrings;
    // invalid values
    //      should return false sequence id is null
    EXPECT_FALSE(generalTgiInferInterface->EncodeTGIResponse(jsonStrings));
    // valid values
    generalTgiInferInterface->respTokenMap[0] = {0};
    EXPECT_TRUE(generalTgiInferInterface->EncodeTGIResponse(jsonStrings));
    generalTgiInferInterface->inputParam->showDetails = true;
    EXPECT_TRUE(generalTgiInferInterface->EncodeTGIResponse(jsonStrings));
}
} // namespace mindie_llm