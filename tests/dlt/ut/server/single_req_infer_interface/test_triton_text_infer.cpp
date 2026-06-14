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
#include "single_req_triton_text_infer_interface.h"
#include "single_llm_pnd_req_handler.h"
#include "request.h"
#include "response.h"
#include "mock_util.h"

using namespace mindie_llm;
using namespace mockcpp;
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {

class TritonTextInferTest : public testing::Test {
protected:
    void SetUp() override
    {
        body.clear();
        httpRequest = httplib::Request();
        httpResponse = httplib::Response();
        request = std::make_shared<Request>(RequestIdNew("mockRequest"));
        response = std::make_shared<Response>(RequestIdNew("mockResponse"));
        requestContext = std::make_shared<RequestContext>(httpRequest, httpResponse);
        pndReqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
        tritonTextInferInterface = std::make_shared<SingleReqTritonTextInferInterface>(pndReqHandler, false);
        tritonTextInferInterface->inputParam = std::make_shared<InferParam>();
        tritonTextInferInterface->request_ = std::make_shared<Request>();
    }
    void TearDown() override
    {
        GlobalMockObject::reset();
    }

    httplib::Request httpRequest;
    httplib::Response httpResponse;
    OrderedJson body;
    RequestSPtr request;
    ResponseSPtr response;
    std::string errorMsg;
    std::shared_ptr<RequestContext> requestContext;
    std::shared_ptr<SingleLLMPnDReqHandler> pndReqHandler;
    std::shared_ptr<SingleReqTritonTextInferInterface> tritonTextInferInterface;
};

TEST_F(TritonTextInferTest, testParseRequestIs)
{
    // valid values
    tritonTextInferInterface->reqJsonBody_["id"] = "a-valid-id";
    EXPECT_EQ(tritonTextInferInterface->ParseRequestId(errorMsg), true);
    tritonTextInferInterface->reqJsonBody_.clear();
    EXPECT_EQ(tritonTextInferInterface->ParseRequestId(errorMsg), true);

    // invalid values
    tritonTextInferInterface->reqJsonBody_["id"] = 123;
    EXPECT_EQ(tritonTextInferInterface->ParseRequestId(errorMsg), false);
}

TEST_F(TritonTextInferTest, testValidateAndPrepareReqToken)
{
    uint64_t timestamp;
    auto stubsTokenizer = MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool*, const std::string&,
        std::vector<int64_t>&, HeadFlag, uint64_t&)).stubs();

    // valid values
    //      string as "text_input"
    body["text_input"] = "a-valid-text-input";
    //      should return false given tokenizer fails
    stubsTokenizer.will(invoke(&MockTokenizerEncodeFail));
    EXPECT_EQ(tritonTextInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), false);
    //      should return true given tokenizer succeeds
    stubsTokenizer.will(invoke(&MockTokenizerEncodeSuccess));
    EXPECT_EQ(tritonTextInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), true);
    //      array as "text_input"
    body["text_input"] = {"a-valid-text-input"};
    EXPECT_EQ(tritonTextInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), true);

    // invalid values
    //      should return false given "text_input" is not found
    body.clear();
    EXPECT_EQ(tritonTextInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), false);
    //      should return false given value of "text_input" is not a string or array
    body["text_input"] = 123;
    EXPECT_EQ(tritonTextInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), false);
    //      should return false given value of "text_input" arr is empty
    body["text_input"] = {};
    EXPECT_EQ(tritonTextInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), false);
    //      should return false given value of "text_input" contains invalid character
    body["text_input"] = "\xFF";
    EXPECT_EQ(tritonTextInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), false);
    //      should return false given value of "text_input" is string and length is 0
    body["text_input"] = "";
    EXPECT_EQ(tritonTextInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), false);
}

TEST_F(TritonTextInferTest, testSetupInferParams)
{
    // valid values
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    tritonTextInferInterface->inputParam = std::make_shared<InferParam>();
    tritonTextInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "text_input": "mock test",
        "parameters": {
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "do_sample": true,
            "seed": 1234,
            "repetition_penalty": 1.03,
            "watermark": true,
            "priority": 1,
            "max_new_tokens": 100
        }
    })");
    EXPECT_EQ(tritonTextInferInterface->SetupInferParams(request, errorMsg), true);
    EXPECT_EQ(request->temperature.value(), 0.5);
    EXPECT_EQ(request->topK.value(), 10);
    EXPECT_FLOAT_EQ(request->topP.value(), 0.95);
    EXPECT_EQ(request->seed, 1234);
    EXPECT_FLOAT_EQ(request->repetitionPenalty.value(), 1.03);
    EXPECT_EQ(request->watermark, true);
    EXPECT_EQ(request->priority, 1);
    EXPECT_EQ(tritonTextInferInterface->inputParam->maxNewTokens, 100);
    // invalid values
    //      should return false given any sub-functions failed
    auto stubsAssignSeed = MOCKER_CPP(&AssignSeed, bool(*)(const OrderedJson&, RequestSPtr, std::string&)).stubs();
    stubsAssignSeed.will(returnValue(false));
    EXPECT_EQ(tritonTextInferInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(TritonTextInferTest, testSetupInferParamsWhenReCompute)
{
    // invalid values
    //      should return false given "firstTokenCost" is not found
    tritonTextInferInterface->isReCompute_ = true;
    EXPECT_EQ(tritonTextInferInterface->SetupInferParams(request, errorMsg), false);
    // valid values
    tritonTextInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "text_input": "mock test",
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
            "firstTokenCost": 123,
            "decodeTime": [1, 2]
        }
    })");
    EXPECT_EQ(tritonTextInferInterface->SetupInferParams(request, errorMsg), true);
}

TEST_F(TritonTextInferTest, testBuildResponseJson)
{
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    resp->isEos = false;
    // should return true when resp is not EOS
    EXPECT_EQ(tritonTextInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp), true);
    // when resp is EOS
    resp->isEos = true;
    //      should return false when respTokenMap is empty
    EXPECT_EQ(tritonTextInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp), false);
    //      should return false when convert token to text output failed
    tritonTextInferInterface->respTokenMap[0] = {0};
    auto stubs =
        MOCKER_CPP(&TokenizerProcessPool::Decode,
                   Status(*)(TokenizerProcessPool *, std::vector<int64_t> &, std::string &, const uint64_t &, bool))
            .stubs();
    stubs.will(invoke(&MockTokenizerDecodeFail));
    EXPECT_EQ(tritonTextInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp), false);
    //      should return true when convert token to text output succeed
    stubs.will(invoke(&MockTokenizerDecodeSuccess));
    ResponseContent content;
    content.truncationIndex = 1;
    resp->responseContents.push_back(content);
    tritonTextInferInterface->inputParam->showDetails = true;
    EXPECT_EQ(tritonTextInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp), true);
}

TEST_F(TritonTextInferTest, testBuildTritonTextReComputeBody)
{
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    request->doSample = true;
    request->repetitionPenalty = 1.07f;
    request->seed = static_cast<uint64_t>(424242);
    request->temperature = 0.65f;
    request->topK = 40;
    request->topP = 0.92f;
    request->typicalP = 0.8f;
    request->watermark = true;
    request->priority = 3;

    auto inferParam = std::make_shared<InferParam>();
    inferParam->userInputId = std::string("u-12345");
    inferParam->showDetails = true;
    inferParam->timeout = static_cast<uint64_t>(7777);
    inferParam->batchSize = 2;
    inferParam->maxNewTokens = 128;

    tritonTextInferInterface->inputParam = inferParam;
    tritonTextInferInterface->request_ = request;
    tritonTextInferInterface->isReCompute_ = true;

    auto &m = tritonTextInferInterface->singleLLMReqHandlerBase_->GetMetrics();
    m.firstTokenCost = static_cast<uint64_t>(12);
    m.decodeTime = std::vector<uint64_t>{25, 35};

    std::vector<BestNTokens> emptyTokens;
    std::string ret = tritonTextInferInterface->BuildTritonTextReComputeBody(emptyTokens);

    OrderedJson obj = OrderedJson::parse(ret);
    ASSERT_TRUE(obj.contains("text_input"));
    EXPECT_TRUE(obj["text_input"].is_string());
    ASSERT_TRUE(obj.contains("id"));
    EXPECT_EQ(obj["id"], "u-12345");
    ASSERT_TRUE(obj.contains("parameters"));
    const auto &p = obj["parameters"];
    ASSERT_TRUE(p.contains("details"));
    EXPECT_TRUE(p["details"].get<bool>());
    ASSERT_TRUE(p.contains("timeout"));
    EXPECT_EQ(p["timeout"].get<uint64_t>(), static_cast<uint64_t>(7777));
    ASSERT_TRUE(p.contains("batch_size"));
    EXPECT_EQ(p["batch_size"].get<int32_t>(), 2);
    ASSERT_TRUE(p.contains("max_new_tokens"));
    EXPECT_EQ(p["max_new_tokens"].get<int32_t>(), 128);
    ASSERT_TRUE(p.contains("do_sample"));
    EXPECT_TRUE(p["do_sample"].get<bool>());
    ASSERT_TRUE(p.contains("repetition_penalty"));
    EXPECT_NEAR(p["repetition_penalty"].get<double>(), 1.07, 1e-6);
    ASSERT_TRUE(p.contains("seed"));
    EXPECT_EQ(p["seed"].get<uint64_t>(), static_cast<uint64_t>(424242));
    ASSERT_TRUE(p.contains("temperature"));
    EXPECT_NEAR(p["temperature"].get<double>(), 0.65, 1e-6);
    ASSERT_TRUE(p.contains("top_k"));
    EXPECT_EQ(p["top_k"].get<int32_t>(), 40);
    ASSERT_TRUE(p.contains("top_p"));
    EXPECT_NEAR(p["top_p"].get<double>(), 0.92, 1e-6);
    ASSERT_TRUE(p.contains("typical_p"));
    EXPECT_NEAR(p["typical_p"].get<double>(), 0.8, 1e-6);
    ASSERT_TRUE(p.contains("watermark"));
    EXPECT_TRUE(p["watermark"].get<bool>());
    ASSERT_TRUE(p.contains("priority"));
    EXPECT_EQ(p["priority"].get<uint64_t>(), static_cast<uint64_t>(3));
    ASSERT_TRUE(p.contains("firstTokenCost"));
    EXPECT_EQ(p["firstTokenCost"].get<uint64_t>(), static_cast<uint64_t>(12));
    ASSERT_TRUE(p.contains("decodeTime"));
    ASSERT_TRUE(p["decodeTime"].is_array());
    const auto dt = p["decodeTime"].get<std::vector<uint64_t>>();
    ASSERT_EQ(dt.size(), 2u);
    EXPECT_EQ(dt[0], static_cast<uint64_t>(25));
    EXPECT_EQ(dt[1], static_cast<uint64_t>(35));
}

} // namespace mindie_llm