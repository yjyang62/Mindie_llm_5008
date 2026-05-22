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
#include "single_req_openai_infer_interface.h"
#include "single_llm_pnd_req_handler.h"
#include "request.h"
#include "response.h"
#include "mock_util.h"
#include "base_config_manager.h"

using namespace mindie_llm;
using namespace mockcpp;
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {

class OpenAiInferTest : public testing::Test {
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
        openaiInferInterface = std::make_shared<SingleReqOpenAiInferInterface>(pndReqHandler, false);
        openaiInferInterface->inputParam = std::make_shared<InferParam>();
        openaiInferInterface->request_ = std::make_shared<Request>();
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
    std::shared_ptr<SingleReqOpenAiInferInterface> openaiInferInterface;
};

TEST_F(OpenAiInferTest, testParseModelName) // "model" item in OpenAI API
{
    std::string model;
    std::vector<ModelDeployConfig> mockConfigs;
    ModelDeployConfig mockConfig;
    mockConfig.modelName = "a-valid-model-name";
    mockConfigs.push_back(mockConfig);
    auto stubsGetParam = MOCKER_CPP(&ModelDeployConfigManager::GetParam,
       std::vector<ModelDeployConfig>& (*)()).stubs();
    stubsGetParam.will(returnValue(mockConfigs));
            
    // valid values
    body["model"] = "a-valid-model-name";
    EXPECT_EQ(openaiInferInterface->ParseModelName(body, model, errorMsg), true);
    EXPECT_EQ(model, "a-valid-model-name");

    // invalid values
    //      should return false given "model" item is empty
    body.clear();
    EXPECT_EQ(openaiInferInterface->ParseModelName(body, model, errorMsg), false);
    body["model"] = "";
    EXPECT_EQ(openaiInferInterface->ParseModelName(body, model, errorMsg), false);
    //      should return false given "model" is not found in ModelDeployConfig
    body["model"] = "a-model-name-not-in-config";
    EXPECT_EQ(openaiInferInterface->ParseModelName(body, model, errorMsg), false);
    //      should return false given "model" type is not string
    body["model"] = 123;
    EXPECT_EQ(openaiInferInterface->ParseModelName(body, model, errorMsg), false);
}

TEST_F(OpenAiInferTest, testParseTextInput) // "messages" items in OpenAI API
{
    body["model"] = "a-valid-model-name";
    
    // valid values
    //      should return true given both "role" and "content" are found,
    //                               and "role" is in {system, assistant, user, tool}
    body["messages"] = nlohmann::json::array();
    body["messages"].push_back({{"role", "user"}, {"content", "what is your name?"}});


    // invalid values
    //      should return false given message is empty
    body["messages"].clear();
    EXPECT_EQ(openaiInferInterface->ParseTextInput(body, errorMsg), false);
    //      should return false given message is not an array
    body["messages"] = 123;
    EXPECT_EQ(openaiInferInterface->ParseTextInput(body, errorMsg), false);
    //      should return false given "role" is not found in keys
    body["messages"] = {{"content", "hello"}};
    EXPECT_EQ(openaiInferInterface->ParseTextInput(body, errorMsg), false);
    //      should return false given "content" is not found in keys
    body["messages"] = {{"role", "user"}};
    EXPECT_EQ(openaiInferInterface->ParseTextInput(body, errorMsg), false);
    //      should return false given "role" is not in {system, assistant, user, tool}
    body["messages"] = {{"role", "an-invalid-role"}, {"content", "hello"}};
    EXPECT_EQ(openaiInferInterface->ParseTextInput(body, errorMsg), false);
}

TEST_F(OpenAiInferTest, testValidToolCallID) // "tool_call_id" item in OpenAI API
{
    // valid values
    //      should return true given "tool_call_id" is not empty and is a string
    body["tool_call_id"] = "a-valid-tool-call-id";
    EXPECT_EQ(openaiInferInterface->ValidToolCallID(body, errorMsg), true);

    // invalid values
    //      should return false given "tool_call_id" is null
    body.clear();
    EXPECT_EQ(openaiInferInterface->ValidToolCallID(body, errorMsg), false);
    //      should return false given the type of "tool_call_id" is not string
    body["tool_call_id"] = 123;
    EXPECT_EQ(openaiInferInterface->ValidToolCallID(body, errorMsg), false);
    //      should return false given length of "tool_call_id" is 0
    body["tool_call_id"] = "";
    EXPECT_EQ(openaiInferInterface->ValidToolCallID(body, errorMsg), false);
}

TEST_F(OpenAiInferTest, testValidAssistantMessage) // "content" item in OpenAI API
{
    // valid values
    //      should return true given "content" is not empty and length > 0
    body["content"] = "a-valid-assistant-content";
    EXPECT_EQ(openaiInferInterface->ValidAssistantMessage(body, errorMsg), true);

    // invalid values
    //      should return false given "content" is empty
    body.clear();
    EXPECT_EQ(openaiInferInterface->ValidAssistantMessage(body, errorMsg), false);
    //      should return false given length of "content" is 0
    body["content"] = "";
    EXPECT_EQ(openaiInferInterface->ValidAssistantMessage(body, errorMsg), false);
}

TEST_F(OpenAiInferTest, testCheckInputLen)
{
    std::string input;
    // valid values
    input = "a-short-input";
    EXPECT_EQ(openaiInferInterface->CheckTextInputLen(input, errorMsg), true);

    // invalid values
    auto serverConfig = mindie_llm::ConfigManager::GetInstance().GetServerConfig();
    uint32_t MAX_INPUTS_NUM = serverConfig.maxRequestLength * 1024 * 1024;
    input = std::string(MAX_INPUTS_NUM + 1, 'a');
    EXPECT_EQ(openaiInferInterface->CheckTextInputLen(input, errorMsg), false);
}

TEST_F(OpenAiInferTest, testValidateAndPrepareReqToken)
{
    OrderedJson body;
    std::string errorMsg, input;
    uint64_t timestamp;
    auto stubsTokenizer = MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool*, const std::string&,
        std::vector<int64_t>&, HeadFlag, uint64_t&)).stubs();

    std::vector<ModelDeployConfig> mockConfigs;
    ModelDeployConfig mockConfig;
    mockConfig.modelName = "a-valid-model-name";
    mockConfigs.push_back(mockConfig);
    auto stubsGetParam = MOCKER_CPP(&ModelDeployConfigManager::GetParam,
       std::vector<ModelDeployConfig>& (*)()).stubs();
    stubsGetParam.will(returnValue(mockConfigs));
    
    // valid values
    body["model"] = "a-valid-model-name";
    body["messages"] = nlohmann::json::array();
    body["messages"].push_back({{"role", "user"}, {"content", "what is your name?"}});
    //      should return false given tokenizer fails
    stubsTokenizer.will(invoke(&MockTokenizerEncodeFail));
    EXPECT_EQ(openaiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), false);
    //      should return true given tokenize succeeds
    stubsTokenizer.will(invoke(&MockTokenizerEncodeSuccess));
    openaiInferInterface->inputParam->useToolsCall = false;
    EXPECT_EQ(openaiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), true);
    EXPECT_EQ(openaiInferInterface->reqTokens_, (std::vector<int64_t>{1, 2, 3}));

    // invalid values
    //      should return false given "model" is not found in ModelDeployConfig
    body.clear();
    EXPECT_EQ(openaiInferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp), false);
};

TEST_F(OpenAiInferTest, testSetupInferParams)
{
    // valid values
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    openaiInferInterface->inputParam = std::make_shared<InferParam>();
    openaiInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "messages": [{"role": "user", "content": "mock test"}],
        "ignore_eos": true,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.95,
        "seed": 1234,
        "repetition_penalty": 1.03,
        "frequency_penalty": 1.05,
        "presence_penalty": 1.07,
        "skip_special_tokens": false,
        "best_of": 5,
        "use_beam_search": true,
        "stream": true
    })");
    EXPECT_EQ(openaiInferInterface->SetupInferParams(request, errorMsg), true);
    EXPECT_EQ(request->temperature.value(), 0.5);
    EXPECT_EQ(request->topK.value(), 10);
    EXPECT_FLOAT_EQ(request->topP.value(), 0.95);
    EXPECT_EQ(request->seed, 1234);
    EXPECT_FLOAT_EQ(request->repetitionPenalty.value(), 1.03);
    EXPECT_FLOAT_EQ(request->frequencyPenalty.value(), 1.05);
    EXPECT_FLOAT_EQ(request->presencyPenalty.value(), 1.07);
    EXPECT_EQ(request->skipSpecialTokens.value(), false);
    EXPECT_EQ(request->bestOf.value(), 5);
    EXPECT_EQ(request->useBeamSearch.value(), true);
    EXPECT_EQ(openaiInferInterface->inputParam->streamMode, true);
    
    // invalid values
    //      should return false given any sub-functions failed
    auto stubsAssignSeed = MOCKER_CPP(&AssignSeed, bool(*)(const OrderedJson&, RequestSPtr, std::string&)).stubs();
    stubsAssignSeed.will(returnValue(false));
    EXPECT_EQ(openaiInferInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(OpenAiInferTest, testCheckToolType)
{
    OrderedJson body;
    std::string error;
    // valid values
    body["type"] = "function";
    EXPECT_TRUE(openaiInferInterface->CheckToolType(body, error));
    // invalid values
    //      should return false given "type" not exist
    body.erase("type");
    EXPECT_FALSE(openaiInferInterface->CheckToolType(body, error));
    //      should return false given "type" is null
    body["type"] = nullptr;
    EXPECT_FALSE(openaiInferInterface->CheckToolType(body, error));
    //      should return false given "type" not string type
    body["type"] = 1;
    EXPECT_FALSE(openaiInferInterface->CheckToolType(body, error));
    //      should return false given "type" not string type
    body["type"] = "validType";
    EXPECT_FALSE(openaiInferInterface->CheckToolType(body, error));
}

TEST_F(OpenAiInferTest, testCheckFunction)
{
    OrderedJson body;
    std::string error;
    // valid values
    body["function"]["name"] = "valid_function123";
    EXPECT_TRUE(openaiInferInterface->CheckFunction(body, error));
    // invalid values
    //      should return false given "function" not exist
    body.erase("function");
    EXPECT_FALSE(openaiInferInterface->CheckFunction(body, error));
    //      should return false given "function" is null
    body["function"] = nullptr;
    EXPECT_FALSE(openaiInferInterface->CheckFunction(body, error));
    //      should return false given "name" not exist
    body["function"]["invalidName"] = "invalidName";
    EXPECT_FALSE(openaiInferInterface->CheckFunction(body, error));
    //      should return false given "name" is null
    body["function"]["name"] = nullptr;
    EXPECT_FALSE(openaiInferInterface->CheckFunction(body, error));
    //      should return false given wrong function name
    body["function"]["name"] = "func#123";
    EXPECT_FALSE(openaiInferInterface->CheckFunction(body, error));
}

TEST_F(OpenAiInferTest, testAssignOpenAiToolChoice)
{
    OrderedJson body;
    std::string error;
    // valid values
    EXPECT_TRUE(openaiInferInterface->AssignOpenAiToolChoice(body, error));
    body["tool_choice"] = "auto";
    EXPECT_TRUE(openaiInferInterface->AssignOpenAiToolChoice(body, error));
    OrderedJson toolObj;
    toolObj["type"] = "function";
    toolObj["function"]["name"] = "valid_function123";
    body["tool_choice"] = toolObj;
    EXPECT_TRUE(openaiInferInterface->AssignOpenAiToolChoice(body, error));
    // invalid values
    //      should return false given "tool_choice" not in [none, auto, required]
    body["tool_choice"] = "value";
    EXPECT_FALSE(openaiInferInterface->AssignOpenAiToolChoice(body, error));
    body["tool_choice"] = 1;
    EXPECT_FALSE(openaiInferInterface->AssignOpenAiToolChoice(body, error));
    toolObj["type"] = "invalidType";
    body["tool_choice"] = toolObj;
    EXPECT_FALSE(openaiInferInterface->AssignOpenAiToolChoice(body, error));
}

TEST_F(OpenAiInferTest, testAssignOpenAiTools)
{
    OrderedJson body;
    std::string error;
    // valid values
    OrderedJson tool1;
    tool1["type"] = "function";
    tool1["function"]["name"] = "valid_func_1";
    OrderedJson tool2;
    tool2["type"] = "function";
    tool2["function"]["name"] = "valid-func-2";
    body["tools"] = OrderedJson::array({tool1, tool2});
    EXPECT_TRUE(openaiInferInterface->AssignOpenAiTools(body, error));
    // invalid values
    //      should return false given "tools" not array
    body["tools"] = "abc";
    EXPECT_FALSE(openaiInferInterface->AssignOpenAiTools(body, error));
    //      should return false given tool type not exist
    body.erase("tools");
    tool1.erase("type");
    body["tools"] = OrderedJson::array({tool1});
    EXPECT_FALSE(openaiInferInterface->AssignOpenAiTools(body, error));
}

TEST_F(OpenAiInferTest, testSetReturnSeqCount)
{
    OrderedJson body;
    std::string error;
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    // valid values
    request->useBeamSearch.emplace(true);
    EXPECT_TRUE(openaiInferInterface->SetReturnSeqCount(request, error));
    // invalid values
    //      should return false when bestOf is greater than 1 without using beam search
    request->useBeamSearch.emplace(false);
    request->bestOf.emplace(5);
    EXPECT_FALSE(openaiInferInterface->SetReturnSeqCount(request, error));
    //      should return false when n is greater than 1 without using beam search
    request->bestOf.reset();
    request->n.emplace(5);
    EXPECT_FALSE(openaiInferInterface->SetReturnSeqCount(request, error));
    //      should return false when temperature not exist
    request->bestOf.emplace(5);
    EXPECT_FALSE(openaiInferInterface->SetReturnSeqCount(request, error));
    //      should return false when best_of less than n
    request->bestOf.emplace(1);
    EXPECT_FALSE(openaiInferInterface->SetReturnSeqCount(request, error));
    //      should return false when best_of not equal to n in stream mode
    openaiInferInterface->inputParam->streamMode = true;
    EXPECT_FALSE(openaiInferInterface->SetReturnSeqCount(request, error));
}

TEST_F(OpenAiInferTest, testValidToolCall)
{
    OrderedJson toolCalls;
    std::string msg;
    OrderedJson validToolCall;
    validToolCall["id"] = "id1";
    validToolCall["type"] = "function";
    validToolCall["function"]["name"] = "valid_func-123";
    validToolCall["function"]["arguments"] = "{}";
    toolCalls.push_back(validToolCall);
    // valid values
    EXPECT_TRUE(openaiInferInterface->ValidToolCall(toolCalls, msg));
    // invalid values
    //      should return false given function arguments expected to be a string
    toolCalls.clear();
    OrderedJson invalidToolCall;
    invalidToolCall["id"] = "id2";
    invalidToolCall["type"] = "function";
    invalidToolCall["function"]["name"] = "valid_func-456";
    invalidToolCall["function"]["arguments"] = 1;
    toolCalls.push_back(invalidToolCall);
    EXPECT_FALSE(openaiInferInterface->ValidToolCall(toolCalls, msg));
    //      should return false given function arguments is null
    invalidToolCall["function"]["arguments"] = nullptr;
    toolCalls = OrderedJson::array({invalidToolCall});
    EXPECT_FALSE(openaiInferInterface->ValidToolCall(toolCalls, msg));
    //      should return false given wrong function name
    invalidToolCall["function"]["name"] = "invalid#func";
    toolCalls = OrderedJson::array({invalidToolCall});
    EXPECT_FALSE(openaiInferInterface->ValidToolCall(toolCalls, msg));
    //      should return false given function name not string
    invalidToolCall["function"]["name"] = 123;
    toolCalls = OrderedJson::array({invalidToolCall});
    EXPECT_FALSE(openaiInferInterface->ValidToolCall(toolCalls, msg));
    //      should return false given function name is null
    invalidToolCall["function"]["name"] = nullptr;
    toolCalls = OrderedJson::array({invalidToolCall});
    EXPECT_FALSE(openaiInferInterface->ValidToolCall(toolCalls, msg));
    //      should return false given function is null
    invalidToolCall.erase("function");
    toolCalls = OrderedJson::array({invalidToolCall});
    EXPECT_FALSE(openaiInferInterface->ValidToolCall(toolCalls, msg));
    //      should return false given type is null
    invalidToolCall.erase("type");
    toolCalls = OrderedJson::array({invalidToolCall});
    EXPECT_FALSE(openaiInferInterface->ValidToolCall(toolCalls, msg));
    //      should return false given id is null
    invalidToolCall.erase("id");
    toolCalls = OrderedJson::array({invalidToolCall});
    EXPECT_FALSE(openaiInferInterface->ValidToolCall(toolCalls, msg));
}

TEST_F(OpenAiInferTest, testValidMessagesArray)
{
    OrderedJson body;
    OrderedJson messges;
    std::string msg;
    // invalid values
    //      should return false given messages is null
    EXPECT_FALSE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
    //      should return false given messages is not arr
    body["messages"] = 1;
    EXPECT_FALSE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
    //      should return false given role is null
    OrderedJson roleMsg;
    body["messages"] = OrderedJson::array({roleMsg});
    EXPECT_FALSE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
    //      should return false given role is invalid
    roleMsg["role"] = "invalidRole";
    body["messages"] = OrderedJson::array({roleMsg});
    EXPECT_FALSE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
    //      should return false given tool_call_id is null
    roleMsg["role"] = "tool";
    body["messages"] = OrderedJson::array({roleMsg});
    EXPECT_FALSE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
    //      should return false given tool_calls is null
    roleMsg["role"] = "assistant";
    body["messages"] = OrderedJson::array({roleMsg});
    EXPECT_FALSE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
    //      should return false given content is null
    roleMsg["role"] = "system";
    body["messages"] = OrderedJson::array({roleMsg});
    EXPECT_FALSE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
    //      should return false given content len is 0
    roleMsg["role"] = "system";
    roleMsg["content"] = "";
    body["messages"] = OrderedJson::array({roleMsg});
    EXPECT_FALSE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
    // valid values
    roleMsg["content"] = "hello";
    body["messages"] = OrderedJson::array({roleMsg});
    EXPECT_TRUE(openaiInferInterface->ValidMessagesArray(body, messges, msg));
}

TEST_F(OpenAiInferTest, testFilterToolChoice)
{
    OrderedJson body;
    openaiInferInterface->inputParam->toolChoice = "none";
    openaiInferInterface->FilterToolChoice(body);
    EXPECT_EQ(body, nullptr);
    openaiInferInterface->inputParam->toolChoice = "auto";
    openaiInferInterface->inputParam->toolsObject = OrderedJson::parse(R"({
        "type": "function"
    })");
    openaiInferInterface->FilterToolChoice(body);
    EXPECT_EQ(body["type"], "function");
    openaiInferInterface->inputParam->toolChoice = "function";
    OrderedJson toolChoiceObject;
    toolChoiceObject["id"] = "id1";
    toolChoiceObject["type"] = "type1";
    toolChoiceObject["function"]["name"] = "valid_func";
    toolChoiceObject["function"]["arguments"] = "{}";
    openaiInferInterface->inputParam->toolChoiceObject = toolChoiceObject;
    OrderedJson toolsObject;
    toolsObject["function"]["name"] = "valid_func";
    openaiInferInterface->inputParam->toolsObject = OrderedJson::array({toolsObject});
    ;
    openaiInferInterface->FilterToolChoice(body);
    EXPECT_EQ(body.size(), 1);
    EXPECT_EQ(body[0]["function"]["name"], "valid_func");
}

TEST_F(OpenAiInferTest, testParseToolCall)
{
    // invalid values
    //      should return false given messages is null
    EXPECT_FALSE(openaiInferInterface->ParseToolCall(body, errorMsg));
    //      should return false when tools is empty
    body = OrderedJson::parse(R"(
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant, follow user instructions."
            }
        ]
    })");
    EXPECT_FALSE(openaiInferInterface->ParseToolCall(body, errorMsg));
    // valid values
    OrderedJson tool;
    tool["type"] = "function";
    tool["function"]["name"] = "valid_func_1";
    openaiInferInterface->inputParam->tools = OrderedJson::array({tool}).dump(4);
    EXPECT_TRUE(openaiInferInterface->ParseToolCall(body, errorMsg));
    // case when recompute is true
    body.clear();
    body["messages"] = "1,2,3";
    openaiInferInterface->isReCompute_ = true;
    EXPECT_TRUE(openaiInferInterface->ParseToolCall(body, errorMsg));
}

TEST_F(OpenAiInferTest, testBuildResponseJson)
{
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    // valid values
    EXPECT_TRUE(openaiInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    openaiInferInterface->respTokenMap[0] = {0};
    openaiInferInterface->inputParam->showDetails = true;
    EXPECT_TRUE(openaiInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    // invalid values
    //      should return false when ProcessResponseSingle fail
    MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseSingle, bool (*)(ResponseSPtr, const uint64_t &))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(openaiInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    //      should return false when ProcessResponseStream fail
    MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseStream,
               bool (*)(ResponseSPtr, const std::vector<BestNTokens> &, RespBodyQueue &, const uint64_t &))
        .stubs()
        .will(returnValue(false));
    openaiInferInterface->inputParam->streamMode = true;
    EXPECT_FALSE(openaiInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
}

TEST_F(OpenAiInferTest, testChangeUtf8Str)
{
    std::string input = "before";
    input += static_cast<char>(0x00);
    input += static_cast<char>(0x01);
    input += "middle";
    input += static_cast<char>(0x1F);
    input += "after";
    std::string result = openaiInferInterface->ChangeUtf8Str(input);
    EXPECT_EQ(result, "beforemiddleafter");
}

TEST_F(OpenAiInferTest, testSendStreamResponse)
{
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    openaiInferInterface->inputParam = std::make_shared<InferParam>();
    openaiInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "messages": [{"role": "user", "content": "mock test"}],
        "ignore_eos": true,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.95,
        "seed": 1234,
        "repetition_penalty": 1.03,
        "frequency_penalty": 1.05,
        "presence_penalty": 1.07,
        "skip_special_tokens": false,
        "best_of": 5,
        "use_beam_search": true,
        "stream": true
    })");
    EXPECT_EQ(openaiInferInterface->SetupInferParams(request, errorMsg), true);
    RespBodyQueue jsonStrings;
    auto stubs = MOCKER_CPP(&SingleReqInferInterfaceBase::GetAvailableOutputCache,
                            bool (*)(std::vector<SingleReqInferInterfaceBase::StreamCache> &))
                     .stubs();
    stubs.will(returnValue(false));
    openaiInferInterface->respTokenMap[0] = {0};
    openaiInferInterface->SendStreamResponse(jsonStrings);
    stubs.will(returnValue(true));
    openaiInferInterface->SendStreamResponse(jsonStrings);
    openaiInferInterface->request_->useBeamSearch.emplace(true);
    openaiInferInterface->isEnd = false;
    openaiInferInterface->SendStreamResponse(jsonStrings);
}

TEST_F(OpenAiInferTest, testEncodeSingleTokenLogrobs)
{
    OrderedJson singleToken;
    uint64_t baseIndex = 0;
    uint64_t seqId = 1000;
    openaiInferInterface->request_->topLogprobs.emplace(1);
    openaiInferInterface->logprobsTokensMap[seqId] = {100, 101};
    openaiInferInterface->logprobsMap[seqId] = {-0.123, -0.456};
    auto stubs =
        MOCKER_CPP(&SingleReqInferInterfaceBase::DecodeSingleToken,
                   bool (*)(std::vector<int64_t> &, std::string &, const uint32_t &, const uint32_t &, const bool &))
            .stubs();
    stubs.will(returnValue(true));
    EXPECT_TRUE(openaiInferInterface->EncodeSingleTokenLogrobs(singleToken, baseIndex, seqId));
    stubs.will(returnValue(false));
    EXPECT_FALSE(openaiInferInterface->EncodeSingleTokenLogrobs(singleToken, baseIndex, seqId));
}

TEST_F(OpenAiInferTest, testEncodeResponse)
{
    openaiInferInterface->probesMap[1001] = 0.85;
    openaiInferInterface->returnSeqCount_ = 1;
    openaiInferInterface->request_->logprobs.emplace(true);
    openaiInferInterface->request_->topLogprobs.emplace(1);
    openaiInferInterface->postTokenIdMap[1001] = {100};
    openaiInferInterface->pickedLogprobMap[1001] = {100};
    RespBodyQueue jsonStrs;
    MOCKER_CPP(&SingleReqInferInterfaceBase::DecodeSingleToken,
               bool (*)(std::vector<int64_t> &, std::string &, const uint32_t &, const uint32_t &, const bool &))
        .stubs()
        .will(returnValue(true));
    MOCKER_CPP(&SingleReqOpenAiInferInterface::EncodeSingleTokenLogrobs,
               bool (*)(OrderedJson &, const uint64_t &, const uint64_t,
                        const mindie_llm::SingleReqInferInterfaceBase::StreamCache *))
        .stubs()
        .will(returnValue(true));
    EXPECT_TRUE(openaiInferInterface->EncodeResponse(jsonStrs));
}

TEST_F(OpenAiInferTest, testEncodeStreamResponse)
{
    RespBodyQueue jsonStrs;
    request->stopStrings.reset();
    request->stopTokenIds.reset();
    openaiInferInterface->request_ = request;
    openaiInferInterface->returnSeqCount_ = 1;
    mindie_llm::SingleReqInferInterfaceBase::StreamCache cacheItem;
    cacheItem.probesMap[1001] = 0.85;
    cacheItem.eosMap[1001] = InferStatusType::ITERATION_CONTINUE;
    openaiInferInterface->streamCache.push_back(cacheItem);
    openaiInferInterface->skipCurrentRoundMap[1001] = true;
    EXPECT_TRUE(openaiInferInterface->EncodeStreamResponse(jsonStrs));
}

TEST_F(OpenAiInferTest, testBuildOpenAIReComputeBody)
{
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    request->seed = static_cast<uint64_t>(1234);
    request->temperature = 0.5f;
    request->topK = 10;
    request->topP = 0.95f;
    request->repetitionPenalty = 1.03f;

    auto inferParam = std::make_shared<InferParam>();
    inferParam->streamMode = true;

    openaiInferInterface->inputParam = inferParam;
    openaiInferInterface->request_ = request;
    openaiInferInterface->isReCompute_ = true;
    openaiInferInterface->model = "mockModel";

    std::vector<BestNTokens> tempTokens;
    std::string ret = openaiInferInterface->BuildVllmOpenAIReComputeBody(tempTokens);
    OrderedJson newReqJsonObj = OrderedJson::parse(ret);

    ASSERT_TRUE(newReqJsonObj.contains("seed"));
    EXPECT_EQ(newReqJsonObj["seed"], 1234);
    ASSERT_TRUE(newReqJsonObj.contains("temperature"));
    EXPECT_NEAR(newReqJsonObj["temperature"].get<double>(), 0.5, 1e-6);
    ASSERT_TRUE(newReqJsonObj.contains("top_k"));
    EXPECT_EQ(newReqJsonObj["top_k"], 10);
    ASSERT_TRUE(newReqJsonObj.contains("top_p"));
    EXPECT_NEAR(newReqJsonObj["top_p"].get<double>(), 0.95, 1e-6);
    ASSERT_TRUE(newReqJsonObj.contains("repetition_penalty"));
    EXPECT_NEAR(newReqJsonObj["repetition_penalty"].get<double>(), 1.03, 1e-6);
    ASSERT_TRUE(newReqJsonObj.contains("stream"));
    EXPECT_TRUE(newReqJsonObj["stream"].get<bool>());
    EXPECT_TRUE(newReqJsonObj.find("use_beam_search") == newReqJsonObj.end());
    ASSERT_TRUE(newReqJsonObj.contains("model"));
    EXPECT_EQ(newReqJsonObj["model"], "mockModel");
    EXPECT_TRUE(newReqJsonObj.find("max_tokens") == newReqJsonObj.end());
}
} // namespace mindie_llm