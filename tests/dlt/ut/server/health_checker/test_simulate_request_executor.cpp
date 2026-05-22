/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include <mockcpp/mockcpp.hpp>
#include <thread>
#include <chrono>

#define private public
#include "simulate_request_executor.h"
#include "infer_instances.h"
#include "endpoint_def.h"
#include "request_response/request_id.h"

using namespace mindie_llm;

class SimulateRequestExecutorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        GlobalMockObject::verify();
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

// ==================== Create 工厂方法测试 ====================

TEST_F(SimulateRequestExecutorTest, CreateReturnsValidInstance)
{
    auto executor = SimulateRequestExecutor::Create(InferReqType::REQ_STAND_INFER);
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(executor->reqType_, InferReqType::REQ_STAND_INFER);
}

TEST_F(SimulateRequestExecutorTest, CreateWithPrefillType)
{
    auto executor = SimulateRequestExecutor::Create(InferReqType::REQ_PREFILL);
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(executor->reqType_, InferReqType::REQ_PREFILL);
}

TEST_F(SimulateRequestExecutorTest, CreateWithDecodeType)
{
    auto executor = SimulateRequestExecutor::Create(InferReqType::REQ_DECODE);
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(executor->reqType_, InferReqType::REQ_DECODE);
}

TEST_F(SimulateRequestExecutorTest, CreateWithDefaultType)
{
    auto executor = SimulateRequestExecutor::Create();
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(executor->reqType_, InferReqType::REQ_STAND_INFER);
}

// ==================== CreateSimulateRequest 测试 ====================

TEST_F(SimulateRequestExecutorTest, CreateSimulateRequestBasic)
{
    auto executor = SimulateRequestExecutor::Create(InferReqType::REQ_STAND_INFER);
    ASSERT_NE(executor, nullptr);
    
    RequestSPtr request = executor->CreateSimulateRequest();
    ASSERT_NE(request, nullptr);
    
    // 验证请求参数
    EXPECT_FALSE(request->requestId.empty());  // 使用动态生成的 requestId
    EXPECT_TRUE(request->isSimulateRequest);   // 通过此字段标识虚推
    EXPECT_EQ(request->input_ids.size(), 1);
    EXPECT_EQ(request->input_ids[0], 1);
    EXPECT_EQ(request->input_token_num, 1);
    EXPECT_EQ(request->reqType, InferReqType::REQ_STAND_INFER);
    EXPECT_EQ(request->priority, 0);
    EXPECT_EQ(request->maxOutputLen, 1);
    EXPECT_EQ(request->temperature, 1.0f);
    EXPECT_EQ(request->topK, 1);
    EXPECT_EQ(request->topP, 1.0f);
    EXPECT_EQ(request->doSample, false);
    EXPECT_EQ(request->ignoreEos, false);
}

TEST_F(SimulateRequestExecutorTest, CreateSimulateRequestPrefillType)
{
    auto executor = SimulateRequestExecutor::Create(InferReqType::REQ_PREFILL);
    ASSERT_NE(executor, nullptr);
    
    RequestSPtr request = executor->CreateSimulateRequest();
    ASSERT_NE(request, nullptr);
    
    EXPECT_EQ(request->reqType, InferReqType::REQ_PREFILL);
}

// ==================== SetSimulateCallback 测试 ====================

TEST_F(SimulateRequestExecutorTest, SetSimulateCallbackSetsCallback)
{
    auto executor = SimulateRequestExecutor::Create();
    ASSERT_NE(executor, nullptr);
    
    RequestSPtr request = executor->CreateSimulateRequest();
    ASSERT_NE(request, nullptr);
    
    // 初始时没有回调
    EXPECT_EQ(request->serverResponseCallback_, nullptr);
    
    executor->SetSimulateCallback(request);
    
    // 设置后应该有回调
    EXPECT_NE(request->serverResponseCallback_, nullptr);
}

TEST_F(SimulateRequestExecutorTest, CallbackIgnoresNullResponse)
{
    auto executor = SimulateRequestExecutor::Create();
    RequestSPtr request = executor->CreateSimulateRequest();
    executor->SetSimulateCallback(request);
    
    // 调用回调，传入 nullptr，不应崩溃
    request->serverResponseCallback_(nullptr);
    
    // 队列应该为空
    EXPECT_TRUE(executor->responseQueue_.empty());
}

TEST_F(SimulateRequestExecutorTest, CallbackIgnoresWhenFinished)
{
    auto executor = SimulateRequestExecutor::Create();
    RequestSPtr request = executor->CreateSimulateRequest();
    executor->SetSimulateCallback(request);
    
    // 设置为已完成
    executor->isFinish_.store(true);
    
    // 创建一个有效的响应
    auto response = std::make_shared<Response>(request->requestId);
    response->isEos = true;
    
    // 调用回调，应该被忽略
    request->serverResponseCallback_(response);
    
    // 队列应该为空（被忽略）
    EXPECT_TRUE(executor->responseQueue_.empty());
}

TEST_F(SimulateRequestExecutorTest, CallbackIgnoresReleaseKvComplete)
{
    auto executor = SimulateRequestExecutor::Create();
    RequestSPtr request = executor->CreateSimulateRequest();
    executor->SetSimulateCallback(request);
    
    auto response = std::make_shared<Response>(request->requestId);
    response->inferStatusFlag = InferStatusType::RELEASE_KV_COMPLETE;
    
    request->serverResponseCallback_(response);
    
    // 应该被忽略
    EXPECT_TRUE(executor->responseQueue_.empty());
}

TEST_F(SimulateRequestExecutorTest, CallbackIgnoresIllegalInput)
{
    auto executor = SimulateRequestExecutor::Create();
    RequestSPtr request = executor->CreateSimulateRequest();
    executor->SetSimulateCallback(request);
    
    auto response = std::make_shared<Response>(request->requestId);
    response->inferStatusFlag = InferStatusType::ILLEGAL_INPUT;
    
    request->serverResponseCallback_(response);
    
    EXPECT_TRUE(executor->responseQueue_.empty());
}

TEST_F(SimulateRequestExecutorTest, CallbackIgnoresRecomputedTriggered)
{
    auto executor = SimulateRequestExecutor::Create();
    RequestSPtr request = executor->CreateSimulateRequest();
    executor->SetSimulateCallback(request);
    
    auto response = std::make_shared<Response>(request->requestId);
    response->transferStatusFlag = TransferStatusType::RECOMPUTED_TRIGGERED;
    
    request->serverResponseCallback_(response);
    
    EXPECT_TRUE(executor->responseQueue_.empty());
}

TEST_F(SimulateRequestExecutorTest, CallbackPushesValidResponse)
{
    auto executor = SimulateRequestExecutor::Create();
    RequestSPtr request = executor->CreateSimulateRequest();
    executor->SetSimulateCallback(request);
    
    auto response = std::make_shared<Response>(request->requestId);
    response->isEos = true;
    response->inferStatusFlag = InferStatusType::ITERATION_CONTINUE;
    response->transferStatusFlag = TransferStatusType::NOT_TRANSFER;
    
    request->serverResponseCallback_(response);
    
    // 响应应该被加入队列
    EXPECT_FALSE(executor->responseQueue_.empty());
    EXPECT_EQ(executor->responseQueue_.size(), 1);
}

// ==================== ParseTokensFromResponse 测试 ====================

TEST_F(SimulateRequestExecutorTest, ParseTokensFromResponseEmpty)
{
    auto executor = SimulateRequestExecutor::Create();
    
    auto response = std::make_shared<Response>("test_request_id");
    // responseContents 为空
    
    EXPECT_FALSE(executor->ParseTokensFromResponse(response));
}

TEST_F(SimulateRequestExecutorTest, ParseTokensFromResponseInvalidSeqId)
{
    auto executor = SimulateRequestExecutor::Create();
    
    auto response = std::make_shared<Response>("test_request_id");
    ResponseContent content;
    content.seqId = 0;  // 无效的 seqId
    content.outTokenIds = {1, 2, 3};
    response->responseContents.push_back(content);
    
    EXPECT_FALSE(executor->ParseTokensFromResponse(response));
}

TEST_F(SimulateRequestExecutorTest, ParseTokensFromResponseEmptyTokenIds)
{
    auto executor = SimulateRequestExecutor::Create();
    
    auto response = std::make_shared<Response>("test_request_id");
    ResponseContent content;
    content.seqId = 1;
    content.outTokenIds = {};  // 空 token
    response->responseContents.push_back(content);
    
    EXPECT_FALSE(executor->ParseTokensFromResponse(response));
}

TEST_F(SimulateRequestExecutorTest, ParseTokensFromResponseValid)
{
    auto executor = SimulateRequestExecutor::Create();
    
    auto response = std::make_shared<Response>("test_request_id");
    ResponseContent content;
    content.seqId = 1;
    content.outTokenIds = {100, 200, 300};
    response->responseContents.push_back(content);
    
    EXPECT_TRUE(executor->ParseTokensFromResponse(response));
}

TEST_F(SimulateRequestExecutorTest, ParseTokensFromResponseMultipleContents)
{
    auto executor = SimulateRequestExecutor::Create();
    
    auto response = std::make_shared<Response>("test_request_id");
    
    ResponseContent content1;
    content1.seqId = 1;
    content1.outTokenIds = {100};
    response->responseContents.push_back(content1);
    
    ResponseContent content2;
    content2.seqId = 2;
    content2.outTokenIds = {200};
    response->responseContents.push_back(content2);
    
    EXPECT_TRUE(executor->ParseTokensFromResponse(response));
}

TEST_F(SimulateRequestExecutorTest, ParseTokensFromResponsePartialInvalid)
{
    auto executor = SimulateRequestExecutor::Create();
    
    auto response = std::make_shared<Response>("test_request_id");
    
    // 第一个有效
    ResponseContent content1;
    content1.seqId = 1;
    content1.outTokenIds = {100};
    response->responseContents.push_back(content1);
    
    // 第二个无效（seqId = 0）
    ResponseContent content2;
    content2.seqId = 0;
    content2.outTokenIds = {200};
    response->responseContents.push_back(content2);
    
    // 只要有一个无效就返回 false
    EXPECT_FALSE(executor->ParseTokensFromResponse(response));
}

// ==================== 初始状态测试 ====================

TEST_F(SimulateRequestExecutorTest, InitialStateIsFinishFalse)
{
    auto executor = SimulateRequestExecutor::Create();
    EXPECT_FALSE(executor->isFinish_.load());
}

TEST_F(SimulateRequestExecutorTest, InitialStateQueueEmpty)
{
    auto executor = SimulateRequestExecutor::Create();
    EXPECT_TRUE(executor->responseQueue_.empty());
}

// ==================== isSimulateRequest 字段测试 ====================

TEST_F(SimulateRequestExecutorTest, SimulateRequestFieldIsSet)
{
    auto executor = SimulateRequestExecutor::Create();
    RequestSPtr request = executor->CreateSimulateRequest();
    
    // 验证虚推请求通过 isSimulateRequest 字段标识
    EXPECT_TRUE(request->isSimulateRequest);
    // requestId 使用动态生成，应该以 endpoint_common_ 前缀开头
    EXPECT_TRUE(request->requestId.find("endpoint_common_") == 0);
}

TEST_F(SimulateRequestExecutorTest, NormalRequestIsSimulateRequestFalse)
{
    // 验证正常请求的 isSimulateRequest 默认为 false
    auto normalRequest = std::make_shared<Request>();
    EXPECT_FALSE(normalRequest->isSimulateRequest);
}

// ==================== ISimulateExecutor 接口兼容性测试 ====================

TEST_F(SimulateRequestExecutorTest, ImplementsISimulateExecutor)
{
    auto executor = SimulateRequestExecutor::Create();
    
    // 应该可以转换为基类指针
    std::shared_ptr<ISimulateExecutor> basePtr = executor;
    EXPECT_NE(basePtr, nullptr);
}

// ==================== 生命周期安全测试 ====================

TEST_F(SimulateRequestExecutorTest, WeakPtrCallbackSafetyAfterDestroy)
{
    std::function<void(ResponseSPtr)> savedCallback;
    
    {
        auto executor = SimulateRequestExecutor::Create();
        RequestSPtr request = executor->CreateSimulateRequest();
        executor->SetSimulateCallback(request);
        
        // 保存回调
        savedCallback = request->serverResponseCallback_;
    }
    // executor 已销毁
    
    // 调用保存的回调，使用 weak_ptr 应该安全
    auto response = std::make_shared<Response>("test");
    savedCallback(response);  // 不应崩溃
    
    SUCCEED();
}

