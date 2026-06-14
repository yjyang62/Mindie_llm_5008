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
#include "single_req_triton_token_infer_interface.h"
#include "single_llm_pnd_req_handler.h"
#include "request.h"
#include "response.h"
#include "mock_util.h"
#include "base_config_manager.h"

using namespace mindie_llm;
using namespace mockcpp;
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {

class TritonTokenInferTest : public testing::Test {
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
        tritonTokenInferInterface = std::make_shared<SingleReqTritonTokenInferInterface>(pndReqHandler, false);
        tritonTokenInferInterface->inputParam = std::make_shared<InferParam>();
        tritonTokenInferInterface->request_ = std::make_shared<Request>();
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
    std::shared_ptr<SingleReqTritonTokenInferInterface> tritonTokenInferInterface;
};

TEST_F(TritonTokenInferTest, testCheckReqInputName)
{
    // valid values
    body["name"] = "a-valid-name";
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputName(body, errorMsg), true);
    EXPECT_EQ(tritonTokenInferInterface->inputNames, std::vector<std::string>({"a-valid-name"}));

    // invalid values
    //      should return false given "name" is not found
    body.clear();
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputName(body, errorMsg), false);
    //      should return false given length of "name" exceeds limit
    body["name"] = std::string(256U+10, 'a');
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputName(body, errorMsg), false);
}

TEST_F(TritonTokenInferTest, testCheckReqInputShape)
{
    // valid values
    //      should return true given "shape" only has 1 element
    tritonTokenInferInterface->inputShape.clear();
    body["shape"] = std::vector<uint16_t>({100});
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputShape(body, errorMsg), true);
    EXPECT_EQ(tritonTokenInferInterface->inputShape, std::vector<std::vector<int64_t>>({{1, 100}}));
    //      when "shape" has 2 elements
    tritonTokenInferInterface->inputShape.clear();
    body["shape"] = std::vector<uint16_t>({1, 100});
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputShape(body, errorMsg), true);
    EXPECT_EQ(tritonTokenInferInterface->inputShape, std::vector<std::vector<int64_t>>({{1, 100}}));

    // invalid values
    //      should return false given "shape" is not found
    body.clear();
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputShape(body, errorMsg), false);
    //      should return false given type of "shape" is not unsigned int
    body["shape"] = std::vector<std::string>({"2"}); // vector of string
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputShape(body, errorMsg), false);
    //      should return false given "shape" has more than 2 elements
    body["shape"] = std::vector<uint16_t>({1, 1, 100});
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputShape(body, errorMsg), false);
}

TEST_F(TritonTokenInferTest, testCheckReqInputDataType)
{
    // valid values
    //      should return true given "datatype" is found in DATA_TYPE_MAPPING
    body["datatype"] = "UINT32";
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputDataType(body, errorMsg), true);
    EXPECT_EQ(tritonTokenInferInterface->inputDataType, std::vector<InferDataType>({InferDataType::TYPE_UINT32}));

    // invalid values
    //      return false given "datatype" is not found
    body.clear();
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputDataType(body, errorMsg), false);
    //      return false given value of "datatype" is not string
    body["datatype"] = 123;
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputDataType(body, errorMsg), false);
    //      return false given datatype is not found in DATA_TYPE_MAPPING
    body["datatype"] = "uint32"; // uint32 is invalid, UINT32 is valid
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputDataType(body, errorMsg), false);
}

TEST_F(TritonTokenInferTest, testCheckReqInputData)
{
    // valid values
    //      should return true given "data" is found and is an array of integers
    body["data"] = std::vector<int64_t>({1, 2, 3});
    body["shape"] = std::vector<uint16_t>({3});
    tritonTokenInferInterface->inputShape = std::vector<std::vector<int64_t>>({{1, 3}});
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputData(body, errorMsg), true);

    // invalid values
    //      should return false given "data" is not found
    body.clear();
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputData(body, errorMsg), false);
    //      should return false given "data" is not an array
    body["data"] = 123;
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputData(body, errorMsg), false);
    //      should return false given size of "data" is of out limit
    body["data"] = std::vector<int64_t>(MAX_TOKENS_NUM+10, 1);
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputData(body, errorMsg), false);
    //      should return false given any member of "data" is not integer
    body["data"][0] = 1.0;
    EXPECT_EQ(tritonTokenInferInterface->CheckReqInputData(body, errorMsg), false);/**/
}

TEST_F(TritonTokenInferTest, testCheckReqId)
{
    // when "id" is found
    //      should return true given "id" is string
    body["id"] = "a-valid-id";
    EXPECT_EQ(tritonTokenInferInterface->CheckReqId(body, errorMsg), true);
    EXPECT_EQ(tritonTokenInferInterface->inputParam->userInputId, "a-valid-id");
    //      should return false given "id" is not a string
    body["id"] = 123;
    EXPECT_EQ(tritonTokenInferInterface->CheckReqId(body, errorMsg), false);
    //      should return false given "id" does not match pre-set patterns
    body["id"] = "an-invalid-id-with-!@#$%^&*()";
    EXPECT_EQ(tritonTokenInferInterface->CheckReqId(body, errorMsg), false);

    // when "id" is not found
    body.clear();
    //      should return true even when "id" is not found since it is an optional item
    EXPECT_EQ(tritonTokenInferInterface->CheckReqId(body, errorMsg), true);
}

TEST_F(TritonTokenInferTest, testValidateAndPrepareReqToken)
{
    OrderedJson body;
    std::string msg;
    uint64_t timestamp;
    auto stubsTokenizer = MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool*, const std::string&,
        std::vector<int64_t>&, HeadFlag, uint64_t&)).stubs();
    // invalid values
    //      should return false given "inputs" is not found
    EXPECT_EQ(tritonTokenInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp), false);
    //      should return false given "inputs" size is not 1
    body["inputs"] = OrderedJson::array({
        {"image_url", "http://example.com/image1.jpg"},
        {"image_url", "https://example.com/image2.png"},
        {"video_url", "https://example.com/video.mp4"}
    });
    EXPECT_EQ(tritonTokenInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp), false);
    //      should return false given "name" is not found
    body["inputs"] = OrderedJson::array({
        {"image_url", "http://example.com/image1.jpg"}
    });
    EXPECT_EQ(tritonTokenInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp), false);
    //      should return false given "id" is not found
    body = OrderedJson::parse(R"({
    "inputs": [
        {
            "name": "input0",
            "shape": [1, 10],
            "datatype": "UINT32",
            "data": [396, 319, 13996, 29877, 29901, 29907, 3333, 20718, 316, 23924],
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
        }
    ]
    })");
    EXPECT_EQ(tritonTokenInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp), false);
    // valid values
    body["id"] = "a-valid-id";
    body["outputs"] = std::vector<OrderedJson>({
        OrderedJson({{"name", "a-valid-name-one"}}),
        OrderedJson({{"name", "a-valid-name-two"}})
    });
    EXPECT_EQ(tritonTokenInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp), true);
}

TEST_F(TritonTokenInferTest, testBuildResponseJson)
{
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    resp->isEos = false;
    // should return true when resp is not EOS
    EXPECT_EQ(tritonTokenInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp), true);
    resp->isEos = true;
    // invalid values
    //      should return false when outputNames of inputParam is empty
    EXPECT_EQ(tritonTokenInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp), false);
    //      should return false when respTokenMap is empty
    tritonTokenInferInterface->inputParam->outputNames = {"name1", "name2", "name3"};
    EXPECT_EQ(tritonTokenInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp), false);
    // valid values
    tritonTokenInferInterface->respTokenMap[0] = {0};
    EXPECT_EQ(tritonTokenInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp), true);
}

TEST_F(TritonTokenInferTest, testBuildTritonTokenReComputeBody)
{
    OrderedJson body;
    std::string msg;
    uint64_t timestamp;
    std::vector<BestNTokens> tempTokens;
    body = OrderedJson::parse(R"({
    "id": "a-valid-id",
    "outputs": [{"name": "a-valid-name-one"}],
    "inputs": [
        {
            "name": "input0",
            "shape": [1, 10],
            "datatype": "UINT32",
            "data": [396, 319, 13996, 29877, 29901, 29907, 3333, 20718, 316, 23924]
        }
    ]
    })");
    tritonTokenInferInterface->isReCompute_ = true;
    EXPECT_EQ(tritonTokenInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp), true);
    std::string ret = tritonTokenInferInterface->BuildTritonTokenReComputeBody(tempTokens);
    OrderedJson newReqJsonObj = OrderedJson::parse(ret);
    EXPECT_EQ(newReqJsonObj["id"], "a-valid-id");
    EXPECT_EQ(newReqJsonObj["outputs"][0]["name"], "output0");
    EXPECT_EQ(newReqJsonObj["inputs"][0]["name"], "input0");
    EXPECT_EQ(newReqJsonObj["inputs"][0]["datatype"], "UINT32");
}

TEST_F(TritonTokenInferTest, testCheckOutputs)
{
    // valid values
    //      should return true given "outputs" is found and is an array of objects
    body["outputs"] = std::vector<OrderedJson>({
        OrderedJson({{"name", "a-valid-name-one"}}),
        OrderedJson({{"name", "a-valid-name-two"}})
    });
    tritonTokenInferInterface->inputDataType = std::vector<InferDataType>({
        InferDataType::TYPE_STRING, InferDataType::TYPE_STRING
    });
    EXPECT_EQ(tritonTokenInferInterface->CheckOutputs(body, errorMsg), true);
    EXPECT_EQ(tritonTokenInferInterface->inputParam->outputNames.size(), 2);
    
    // invalid values
    //      should return false given "outputs" is not found
    body.clear();
    EXPECT_EQ(tritonTokenInferInterface->CheckOutputs(body, errorMsg), false);
    //      should return false given "outputs" is not an array
    body["outputs"] = 123;
    EXPECT_EQ(tritonTokenInferInterface->CheckOutputs(body, errorMsg), false);
    //      should return false given size of "outputs" is not equal to size of inputDataType
    tritonTokenInferInterface->inputDataType = std::vector<InferDataType>({InferDataType::TYPE_UINT32});
    body["outputs"] = std::vector<int64_t>({1, 2});
    EXPECT_EQ(tritonTokenInferInterface->CheckOutputs(body, errorMsg), false);
    //      should return false given "name" is not found in value of "output"
    body["outputs"] = std::vector<int64_t>({1});
    EXPECT_EQ(tritonTokenInferInterface->CheckOutputs(body, errorMsg), false);
}

TEST_F(TritonTokenInferTest, testSetupInferParams)
{
    // valid values
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    tritonTokenInferInterface->inputParam = std::make_shared<InferParam>();
    tritonTokenInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
    "inputs": [
        {
            "name": "input0",
            "shape": [1, 10],
            "datatype": "UINT32",
            "data": [396, 319, 13996, 29877, 29901, 29907, 3333, 20718, 316, 23924],
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
        }
    ]
    })");

    EXPECT_EQ(tritonTokenInferInterface->SetupInferParams(request, errorMsg), true);
    EXPECT_EQ(request->temperature.value(), 0.5);
    EXPECT_EQ(request->topK.value(), 10);
    EXPECT_FLOAT_EQ(request->topP.value(), 0.95);
    EXPECT_EQ(request->seed, 1234);
    EXPECT_FLOAT_EQ(request->repetitionPenalty.value(), 1.03);
    EXPECT_EQ(request->watermark, true);
    EXPECT_EQ(request->priority, 1);
    EXPECT_EQ(tritonTokenInferInterface->inputParam->maxNewTokens, 100);
    
    // invalid values
    //      should return false given any sub-functions failed
    auto stubsAssignSeed = MOCKER_CPP(&AssignSeed, bool(*)(const OrderedJson&, RequestSPtr, std::string&)).stubs();
    stubsAssignSeed.will(returnValue(false));
    EXPECT_EQ(tritonTokenInferInterface->SetupInferParams(request, errorMsg), false);
}

} // namespace mindie_llm