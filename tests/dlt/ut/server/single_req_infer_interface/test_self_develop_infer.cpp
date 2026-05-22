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
#include "single_req_self_develop_infer_interface.h"
#include "single_llm_pnd_req_handler.h"
#include "request.h"
#include "response.h"
#include "mock_util.h"
#include "base_config_manager.h"

using namespace mindie_llm;
using namespace mockcpp;
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {

class SelfDevelopInterfaceTest : public testing::Test {
protected:
    void SetUp() override
    {
        requestJson.clear();
        httpRequest = httplib::Request();
        httpResponse = httplib::Response();
        request = std::make_shared<Request>(RequestIdNew("mockRequest"));
        response = std::make_shared<Response>(RequestIdNew("mockResponse"));
        requestContext = std::make_shared<RequestContext>(httpRequest, httpResponse);
        pndReqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
        selfDevelopInterface = std::make_shared<SingleReqSelfDevelopInferInterface>(pndReqHandler, false);
        selfDevelopInterface->inputParam = std::make_shared<InferParam>();
        selfDevelopInterface->request_ = std::make_shared<Request>();
    }
    void TearDown() override
    {
        GlobalMockObject::reset();
    }

    httplib::Request httpRequest;
    httplib::Response httpResponse;
    OrderedJson requestJson;
    RequestSPtr request;
    ResponseSPtr response;
    std::string errorMsg;
    std::shared_ptr<RequestContext> requestContext;
    std::shared_ptr<SingleLLMPnDReqHandler> pndReqHandler;
    std::shared_ptr<SingleReqSelfDevelopInferInterface> selfDevelopInterface;
};

TEST_F(SelfDevelopInterfaceTest, testSetupInferParams)
{
    // valid values
    //      should return true given "parameter" is missing
    selfDevelopInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "inputs": "mock test"
    })");
    EXPECT_EQ(selfDevelopInterface->SetupInferParams(request, errorMsg), true);
    //      should return true given "parameter" is found
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    selfDevelopInterface->inputParam = std::make_shared<InferParam>();
    selfDevelopInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "inputs": "mock test",
        "stream": true,
        "parameters": {
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "do_sample": true,
            "seed": 1234,
            "repetition_penalty": 1.03,
            "watermark": true,
            "priority": 1,
            "max_new_tokens": 100,
            "timeout": 3,
            "details": true
        }
    })");
    EXPECT_EQ(selfDevelopInterface->SetupInferParams(request, errorMsg), true);
    EXPECT_FLOAT_EQ(request->temperature.value(), 0.5);
    EXPECT_EQ(request->topK.value(), 10);
    EXPECT_FLOAT_EQ(request->topP.value(), 0.95);
    EXPECT_EQ(request->doSample.value(), true);
    EXPECT_EQ(request->seed, 1234);
    EXPECT_FLOAT_EQ(request->repetitionPenalty.value(), 1.03);
    EXPECT_EQ(request->watermark, true);
    EXPECT_EQ(request->priority, 1);
    EXPECT_EQ(selfDevelopInterface->inputParam->maxNewTokens, 100);
    EXPECT_EQ(selfDevelopInterface->inputParam->timeout, 3);
    EXPECT_EQ(selfDevelopInterface->inputParam->showDetails, true);
    EXPECT_EQ(selfDevelopInterface->inputParam->streamMode, true);
    
    // invalid values
    //      should return false given any sub-functions failed
    auto stubsAssignSeed = MOCKER_CPP(&AssignSeed, bool(*)(const OrderedJson&, RequestSPtr, std::string&)).stubs();
    stubsAssignSeed.will(returnValue(false));
    EXPECT_EQ(selfDevelopInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(SelfDevelopInterfaceTest, testCheckTokenInput)
{
    OrderedJson body;
    std::string msg;
    // invalid values
    //      should return false given "input_id" is missing
    EXPECT_FALSE(selfDevelopInterface->CheckTokenInput(body, msg));
    //      should return false given "input_id" is not integer type
    body["input_id"] = {"inputId"};
    EXPECT_FALSE(selfDevelopInterface->CheckTokenInput(body, msg));
    //      should return false given "input_id" is less than 0
    body["input_id"] = {1, -1, 2};
    EXPECT_FALSE(selfDevelopInterface->CheckTokenInput(body, msg));
    // valid values
    body["input_id"] = {1, 2, 3};
    EXPECT_TRUE(selfDevelopInterface->CheckTokenInput(body, msg));
}

TEST_F(SelfDevelopInterfaceTest, testCheckTextInput)
{
    OrderedJson body;
    std::string msg;
    uint64_t timestamp;
    auto stubsTokenizer = MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool*, const std::string&,
        std::vector<int64_t>&, HeadFlag, uint64_t&)).stubs();
    // valid values
    stubsTokenizer.will(invoke(&MockTokenizerEncodeSuccess));
    body["inputs"] = OrderedJson::array({
        {"image_url", "http://example.com/image1.jpg"},
        {"video_url", "http://example.com/video1.mp4"},
        {"audio_url", "http://example.com/audio1.mp3"}
    });
    EXPECT_TRUE(selfDevelopInterface->CheckTextInput(body, msg, timestamp));
    // invalid values
    //      should return false given "inputs" is missing
    stubsTokenizer.will(invoke(&MockTokenizerEncodeFail));
    body["inputs"] = nullptr;
    EXPECT_FALSE(selfDevelopInterface->CheckTextInput(body, msg, timestamp));
    //      should return false when TokenizerProcessPool::Encode failed
    body["inputs"] = "mockInputs";
    EXPECT_FALSE(selfDevelopInterface->CheckTextInput(body, msg, timestamp));
    //      should return false when type of inputs is abnormal
    body["inputs"] = 1;
    EXPECT_FALSE(selfDevelopInterface->CheckTextInput(body, msg, timestamp));
    //      should return false when inputs arr is empty
    body["inputs"] = OrderedJson::array();
    EXPECT_FALSE(selfDevelopInterface->CheckTextInput(body, msg, timestamp));
    //      should return false when type of inputs length is 0
    body["inputs"] = "";
    EXPECT_FALSE(selfDevelopInterface->CheckTextInput(body, msg, timestamp));
    body["inputs"] = "mockInputs";
    selfDevelopInterface->isReCompute_ = true;
    EXPECT_FALSE(selfDevelopInterface->CheckTextInput(body, msg, timestamp));
}

TEST_F(SelfDevelopInterfaceTest, testValidateAndPrepareReqToken)
{
    OrderedJson body;
    std::string msg;
    uint64_t timestamp;
    EXPECT_FALSE(selfDevelopInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
    // should return false when have inputs and input_id at the same time
    body["inputs"] = "mockInputs";
    body["input_id"] = {1, 2, 3};
    EXPECT_FALSE(selfDevelopInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
    body.erase("input_id");
    // valid values
    MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool*, const std::string&,
        std::vector<int64_t>&, HeadFlag, uint64_t&)).stubs().will(invoke(&MockTokenizerEncodeSuccess));
    EXPECT_TRUE(selfDevelopInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
    // invalid values
    //      should return false when type of inputs is abnormal
    body["inputs"] = 1;
    EXPECT_FALSE(selfDevelopInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
    //      should return false given "input_id" is less than 0
    body.erase("inputs");
    body["input_id"] = {1, -1, 2};
    EXPECT_FALSE(selfDevelopInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
    // valid values
    body["input_id"] = {1, 2, 3};
    EXPECT_TRUE(selfDevelopInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
}

TEST_F(SelfDevelopInterfaceTest, testBuildResponseJson)
{
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    // should return false when respTokenMap is empty
    EXPECT_FALSE(selfDevelopInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    // valid values
    selfDevelopInterface->respTokenMap[0] = {0};
    selfDevelopInterface->inputParam->showDetails = true;
    EXPECT_TRUE(selfDevelopInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    // should return false when ProcessResponseSingle fail
    MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseSingle, bool (*)(ResponseSPtr, const uint64_t &))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(selfDevelopInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    // should return false when ProcessResponseStream fail
    MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseStream,
               bool (*)(ResponseSPtr, const std::vector<BestNTokens> &, RespBodyQueue &, const uint64_t &))
        .stubs()
        .will(returnValue(false));
    selfDevelopInterface->inputParam->streamMode = true;
    EXPECT_FALSE(selfDevelopInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
}

TEST_F(SelfDevelopInterfaceTest, testSendStreamResponse)
{
    selfDevelopInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "inputs": "mock test",
        "stream": true,
        "parameters": {
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "do_sample": true,
            "seed": 1234,
            "repetition_penalty": 1.03,
            "watermark": true,
            "priority": 1,
            "max_new_tokens": 100,
            "timeout": 3,
            "details": true
        }
    })");
    EXPECT_TRUE(selfDevelopInterface->SetupInferParams(request, errorMsg));
    RespBodyQueue jsonStrings;
    // sequence id not exist
    selfDevelopInterface->SendStreamResponse(jsonStrings);
    // stream response not finished
    selfDevelopInterface->respTokenMap[0] = {0};
    selfDevelopInterface->SendStreamResponse(jsonStrings);
    EXPECT_EQ(jsonStrings.size(), 1);
    // stream response finished
    selfDevelopInterface->isEnd = true;
    selfDevelopInterface->SendStreamResponse(jsonStrings);
    EXPECT_EQ(jsonStrings.size(), 2);
}

TEST_F(SelfDevelopInterfaceTest, testBuildSelfDevelopReComputeBody)
{
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    request->priority = static_cast<uint64_t>(9);
    request->temperature = 0.72f;
    request->topK = 64;
    request->topP = 0.93f;
    request->doSample = true;
    request->seed = static_cast<uint64_t>(202501);
    request->repetitionPenalty = 1.11f;
    request->watermark = false;
    request->typicalP = 0.7f;

    auto inferParam = std::make_shared<InferParam>();
    inferParam->streamMode = true;
    inferParam->showDetails = true;
    inferParam->timeout = static_cast<uint64_t>(8888);
    inferParam->maxNewTokens = 128;

    selfDevelopInterface->inputParam = inferParam;
    selfDevelopInterface->request_ = request;
    selfDevelopInterface->isReCompute_ = true;

    std::vector<BestNTokens> emptyTokens;
    std::string ret = selfDevelopInterface->BuildSelfDevelopReComputeBody(emptyTokens);

    OrderedJson obj = OrderedJson::parse(ret);
    ASSERT_TRUE(obj.contains("inputs"));
    EXPECT_TRUE(obj["inputs"].is_string());
    ASSERT_TRUE(obj.contains("stream"));
    EXPECT_TRUE(obj["stream"].get<bool>());
    ASSERT_TRUE(obj.contains("parameters"));
    const auto &p = obj["parameters"];
    ASSERT_TRUE(p.contains("priority"));
    EXPECT_EQ(p["priority"].get<uint64_t>(), static_cast<uint64_t>(9));
    ASSERT_TRUE(p.contains("temperature"));
    EXPECT_NEAR(p["temperature"].get<double>(), 0.72, 1e-6);
    ASSERT_TRUE(p.contains("top_k"));
    EXPECT_EQ(p["top_k"].get<int32_t>(), 64);
    ASSERT_TRUE(p.contains("top_p"));
    EXPECT_NEAR(p["top_p"].get<double>(), 0.93, 1e-6);
    ASSERT_TRUE(p.contains("do_sample"));
    EXPECT_TRUE(p["do_sample"].get<bool>());
    ASSERT_TRUE(p.contains("seed"));
    EXPECT_EQ(p["seed"].get<uint64_t>(), static_cast<uint64_t>(202501));
    ASSERT_TRUE(p.contains("repetition_penalty"));
    EXPECT_NEAR(p["repetition_penalty"].get<double>(), 1.11, 1e-6);
    ASSERT_TRUE(p.contains("watermark"));
    EXPECT_FALSE(p["watermark"].get<bool>());
    ASSERT_TRUE(p.contains("typical_p"));
    EXPECT_NEAR(p["typical_p"].get<double>(), 0.7, 1e-6);
    ASSERT_TRUE(p.contains("timeout"));
    EXPECT_EQ(p["timeout"].get<uint64_t>(), static_cast<uint64_t>(8888));
    ASSERT_TRUE(p.contains("details"));
    EXPECT_TRUE(p["details"].get<bool>());
    ASSERT_TRUE(p.contains("max_new_tokens"));
    EXPECT_EQ(p["max_new_tokens"].get<int32_t>(), 128);
}

} // namespace mindie_llm