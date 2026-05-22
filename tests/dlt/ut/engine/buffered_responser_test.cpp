/**
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
#include <boost/stacktrace.hpp>
#include "buffered_responser.h"

using namespace mindie_llm;
using namespace std;

void SignalHandler(int signal)
{
    std::cerr << "\n===== Crash Report =====" << std::endl;
    std::cerr << "Stack Trace: " << std::endl;
    std::cerr << boost::stacktrace::stacktrace() << std::endl;
    std::cerr << "========================\n" << std::endl;
}

class BufferedResponserTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        receivedResponses_.clear();
        responseSendTimes_.clear();
        config_.bufferResponseEnabled = true;
        config_.prefillExpectedTime = 1000;
        config_.decodeExpectedTime = 100;
        forwardCallback_ = [this](std::shared_ptr<Response> response) {
            string reqId = response->reqId;
            int64_t sendTime = chrono::time_point_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now())
                                   .time_since_epoch().count();
            responseSendTimes_[reqId] = sendTime;
            receivedResponses_.push_back(response);
        };
        bufferedResponser_ = std::make_shared<BufferedResponser>(forwardCallback_, config_);
    }

    void TearDown() override { bufferedResponser_.reset(); }

    ResponseSPtr CreateTestResponse(const string &reqId, bool isEnd = false)
    {
        auto resp = std::make_shared<Response>(reqId);
        resp->isEos = isEnd;
        return resp;
    }

    bool WaitForResponses(size_t expectedCount, int timeoutMs)
    {
        auto start = chrono::high_resolution_clock::now();
        while (true) {
            lock_guard<mutex> lock(receivedResponsesMtx_);
            if (receivedResponses_.size() >= expectedCount) {
                return true;
            }
            if (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count() >
                timeoutMs) {
                return false;
            }
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }

    const uint32_t changeNsToMs = 1000000;
    BufferResponseConfig config_;
    std::shared_ptr<BufferedResponser> bufferedResponser_;
    ForwardRespToManagerCall forwardCallback_;
    std::vector<std::shared_ptr<Response>> receivedResponses_;
    map<string, int64_t> responseSendTimes_;
    mutex receivedResponsesMtx_;
};

TEST_F(BufferedResponserTest, PrefillEndResponseTest)
{
    signal(SIGSEGV, SignalHandler);
    const string reqId = "test_req_001";
    auto arriveTime = chrono::high_resolution_clock::now();
    bufferedResponser_->RecordArriveTime(reqId, arriveTime);
    auto response = CreateTestResponse(reqId, true);
    bufferedResponser_->TryRespond(response);
    // 收到Prefill的结束响应时，立即发送
    ASSERT_TRUE(WaitForResponses(1, 20));
    lock_guard<mutex> lock(receivedResponsesMtx_);
    EXPECT_EQ(receivedResponses_[0]->reqId, reqId);
}

TEST_F(BufferedResponserTest, decodeEndResponseTest)
{
    signal(SIGSEGV, SignalHandler);
    const string reqId = "test_req_002";
    auto arriveTime = chrono::high_resolution_clock::now();
    bufferedResponser_->RecordArriveTime(reqId, arriveTime);
    auto resp1 = CreateTestResponse(reqId, false);
    auto resp2 = CreateTestResponse(reqId, false);
    bufferedResponser_->TryRespond(resp1);
    bufferedResponser_->TryRespond(resp2);
    auto respEnd = CreateTestResponse(reqId, true);
    bufferedResponser_->TryRespond(respEnd);
    // 收到decode的结束响应时，立即发送所有请求
    ASSERT_TRUE(WaitForResponses(3, 20));
    lock_guard<mutex> lock(receivedResponsesMtx_);
    EXPECT_EQ(receivedResponses_.size(), 3);
    EXPECT_EQ(receivedResponses_[0]->reqId, reqId);
}

TEST_F(BufferedResponserTest, BufferDisabledTest)
{
    signal(SIGSEGV, SignalHandler);
    config_.bufferResponseEnabled = false;
    bufferedResponser_ = make_shared<BufferedResponser>(forwardCallback_, config_);
    const string reqId = "test_req_003";
    auto arriveTime = chrono::high_resolution_clock::now();
    bufferedResponser_->RecordArriveTime(reqId, arriveTime);
    auto response = CreateTestResponse(reqId, false);
    bufferedResponser_->TryRespond(response);
    this_thread::sleep_for(chrono::milliseconds(10));
    lock_guard<mutex> lock(receivedResponsesMtx_);
    // 关闭特性时，不会由BufferedResponser进行发送
    EXPECT_TRUE(receivedResponses_.empty());
}

TEST_F(BufferedResponserTest, TimeoutSendTest)
{
    signal(SIGSEGV, SignalHandler);
    const string reqId = "test_req_004";
    auto arriveTime = chrono::high_resolution_clock::now();
    int64_t arriveNs = chrono::time_point_cast<chrono::nanoseconds>(arriveTime).time_since_epoch().count();
    bufferedResponser_->RecordArriveTime(reqId, arriveTime);
    auto prefillResponse = CreateTestResponse(reqId, false);
    bufferedResponser_->TryRespond(prefillResponse);
    ASSERT_TRUE(WaitForResponses(1, 1100));
    auto prefillTime = responseSendTimes_.find(reqId);
    ASSERT_TRUE(prefillTime != responseSendTimes_.end());
    int64_t prefillSendNs = prefillTime->second;
    double prefillDiffMs = static_cast<double>(prefillSendNs - arriveNs) / changeNsToMs;
    EXPECT_GE(prefillDiffMs, 950); // prefill响应等待时间需要大于0.95*1000=950ms
    auto decodeResponse = CreateTestResponse(reqId, false);
    bufferedResponser_->TryRespond(decodeResponse);
    ASSERT_TRUE(WaitForResponses(2, 150));
    auto decodeTime = responseSendTimes_.find(reqId);
    ASSERT_TRUE(decodeTime != responseSendTimes_.end());
    int64_t decodeSendNs = decodeTime->second;
    double totalDiffMs = static_cast<double>(decodeSendNs - arriveNs) / changeNsToMs;
    double decodeDiffMs = totalDiffMs - prefillDiffMs;
    EXPECT_GE(decodeDiffMs, 95); // decode响应等待时间需要大于0.95*100=95ms
    lock_guard<mutex> lock(receivedResponsesMtx_);
    EXPECT_EQ(receivedResponses_.size(), 2);
    EXPECT_EQ(receivedResponses_[1]->reqId, reqId);
}

// 多请求并发处理
TEST_F(BufferedResponserTest, ConcurrentRequestsTest)
{
    signal(SIGSEGV, SignalHandler);
    const string reqId1 = "test_req_005";
    const string reqId2 = "test_req_006";
    auto arriveTime1 = chrono::high_resolution_clock::now();
    auto arriveTime2 = chrono::high_resolution_clock::now();
    bufferedResponser_->RecordArriveTime(reqId1, arriveTime1);
    bufferedResponser_->RecordArriveTime(reqId2, arriveTime2);
    auto resp1 = CreateTestResponse(reqId1, false);
    auto resp2 = CreateTestResponse(reqId2, false);
    bufferedResponser_->TryRespond(resp1);
    bufferedResponser_->TryRespond(resp2);
    ASSERT_TRUE(WaitForResponses(2, 1100));
    lock_guard<mutex> lock(receivedResponsesMtx_);
    EXPECT_EQ(receivedResponses_.size(), 2);
    EXPECT_TRUE((receivedResponses_[0]->reqId == reqId1 &&
                 receivedResponses_[1]->reqId == reqId2) ||
                (receivedResponses_[0]->reqId == reqId2 &&
                 receivedResponses_[1]->reqId == reqId1));
}

TEST_F(BufferedResponserTest, PopFrontTest)
{
    signal(SIGSEGV, SignalHandler);
    ResponseSPtr testResp_ = std::make_shared<Response>("test_req_007");
    std::shared_ptr<ResponseBuffer> responseBuffer = std::make_unique<ResponseBuffer>(InferReqType::REQ_PREFILL, 0);
    responseBuffer->AddResponse(testResp_);
    auto result1 = responseBuffer->PopFront();
    ASSERT_NE(result1, nullptr);
    EXPECT_EQ(result1->reqId, "test_req_007");
    auto result2 = responseBuffer->PopFront();
    EXPECT_EQ(result2, nullptr);
}