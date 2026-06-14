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

#include <chrono>
#include <thread>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "load_balancer.h"
#include "llm_engine.h"
#include "scheduler.h"
#include "../crash_handler.h"

using namespace mindie_llm;
using namespace std;

namespace {
std::mutex mtx;
std::condition_variable cv;
int seqCount = 0;
} // namespace

/* 辅助函数 */
class EmptyScheduler : public IScheduler {
    explicit EmptyScheduler() {}
    void AddSeqGroup(SequenceGroupSPtr &seqGroup) override { (void)seqGroup; }
    SchedulerMetric CollectSchedulerMetric() override
    {
        SchedulerMetric metric;
        return metric;
    }
    std::pair<SequenceGroupMetaDatas, SchedulerOutputs> Schedule(bool needSync = false) override
    {
        (void)needSync;
        return {};
    };

    std::pair<SequenceGroupMetaDatas, SchedulerKVTransferOutput> ScheduleTransfer() { return {}; }

    size_t GetUnFinishedSeqGroups() { return 0; }

    std::unordered_set<SequenceId> &FetchFinishedSeqIds(ConcurrentDeque<SequenceId> &finishedSeqIds)
    {
        (void)finishedSeqIds;
        return seqIds_;
    }
    std::unordered_set<SequenceId> &FetchExceptionSeqIds(ConcurrentDeque<SequenceId> &exceptionSeqIds)
    {
        (void)exceptionSeqIds;
        return seqIds_;
    }
    std::unordered_set<RequestId> &FetchAbortedReqIds(ConcurrentDeque<RequestId> &abortedReqIds)
    {
        (void)abortedReqIds;
        return reqIds_;
    }
    void KVPulledReqEnterRunningQueue(ConcurrentDeque<RequestId> &pulledReqIds) { (void)pulledReqIds; }
    void NotifyMeKvPulledSeqIds(SequenceId seqId) { (void)seqId; }
    std::unordered_set<SequenceId> ClearAndReturnTerminatedSeqIds() { return {}; }
    void FetchSeqGeneratedTokens(ConcurrentDeque<std::pair<SequenceId, TokenId>> &seqIdToOutputTokenQueue)
    {
        (void)seqIdToOutputTokenQueue;
    }
    void MarkLastScheduleEmpty() {}
    void ClearLastScheduleEmpty() {}
    void PrepareNextSchedule(std::vector<ScheduledSequenceGroupSPtr> &scheduledSeqGroups) { (void)scheduledSeqGroups; }
    void ClearSeqGrp(SequenceGroupSPtr seqGroup, SequenceStatus finalStatus)
    {
        (void)seqGroup;
        (void)finalStatus;
    }
    void CollectAndClearAbortedParallelSeqGroups() override{};
    std::vector<SequenceGroupSPtr> &GetAbortedParallelSeqGroups() override { return abortedParallelSeqGroups_; }
    void SetPrefillPercentage(uint32_t prefillPercentage)
    {
        (void)prefillPercentage;
    }
    Role SwitchRole() {};
    std::shared_ptr<StagePolicy> GetStagePolicy() {};

private:
    std::unordered_set<SequenceId> seqIds_;
    std::unordered_set<RequestId> reqIds_;
    std::vector<SequenceGroupSPtr> abortedParallelSeqGroups_;
};
class MockScheduler : public EmptyScheduler {
    MockScheduler() : EmptyScheduler() {}
    void StopRunningRequest() override {}
    void AddSeqGroup(SequenceGroupSPtr &seqGroup) override
    {
        std::unique_lock<std::mutex> lock(mtx);
        (void)seqGroup;
        seqCount++;
        cv.notify_one();
    }
    SchedulerMetric CollectSchedulerMetric() override
    {
        SchedulerMetric metric;
        metric.blockInfo.freeNpuBlockNum_ = 160;
        return metric;
    }
};

class LoadBalancerTest : public ::testing::Test {
protected:
    void SetUp() override { seqCount = 0; }

    static void SetUpTestSuite()
    {
        mindie_llm::test::InitCrashHandler();
    }
};

TEST_F(LoadBalancerTest, SendMassiveRequestsAndVerifySchedulingCompletionInShortTime)
{
    size_t dpSize = 4;
    uint32_t maxPrefillBatchSize = 10;
    SchedulerConfig schedulerConfig;
    SchedulerConfigSPtr schedulerConfigPtr = std::make_shared<SchedulerConfig>(schedulerConfig);
    std::vector<std::shared_ptr<EnginePerDP>> enginePerDPs;
    for (size_t i = 0; i < dpSize; ++i) {
        EnginePerDPSPtr enginePerDP = std::make_shared<EnginePerDP>();
        enginePerDPs.emplace_back(enginePerDP);
        enginePerDP->scheduler = std::make_unique<MockScheduler>();
    }
    LoadBalancerPtr loadBalancer_ = MakeLoadBalancer(enginePerDPs, maxPrefillBatchSize);

    // 准备请求
    constexpr int reqNum = 10;
    size_t seqLen[reqNum] = {18, 3, 22, 15, 7, 11, 24, 9, 1, 17};
    // 构造输入
    std::vector<SequenceGroupSPtr> candidates(reqNum);
    for (int i = 0; i < reqNum; i++) {
        RequestId id(std::to_string(i));
        std::vector<SequenceSPtr> seq(1);
        std::vector<TokenId> inputs(seqLen[i], 5);
        seq[0] = std::make_shared<Sequence>(i, 1, inputs);
        candidates[i] = std::make_shared<SequenceGroup>(id, seq);
    }
    for (auto candidate : candidates) {
        loadBalancer_->AddSeqGroup(candidate);
    }
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait_for(lock, std::chrono::milliseconds(10), [] { return seqCount >= reqNum; });
    }
    EXPECT_EQ(seqCount, reqNum);
}

TEST_F(LoadBalancerTest, SendFewRequestsAndVerifyPeriodicTriggerScheduling)
{
    // 初始化准备
    size_t dpSize = 4;
    uint32_t maxPrefillBatchSize = 10;
    SchedulerConfig schedulerConfig;
    SchedulerConfigSPtr schedulerConfigPtr = std::make_shared<SchedulerConfig>(schedulerConfig);
    std::vector<std::shared_ptr<EnginePerDP>> enginePerDPs;
    for (size_t i = 0; i < dpSize; ++i) {
        EnginePerDPSPtr enginePerDP = std::make_shared<EnginePerDP>();
        enginePerDPs.emplace_back(enginePerDP);
        enginePerDP->scheduler = std::make_unique<MockScheduler>();
    }
    LoadBalancerPtr loadBalancer_ = MakeLoadBalancer(enginePerDPs, maxPrefillBatchSize);

    // 构造输入
    RequestId id("hi");
    std::vector<SequenceSPtr> seq(1);
    std::vector<TokenId> inputs(1, 5);
    seq[0] = std::make_shared<Sequence>(1, 1, inputs);
    SequenceGroupSPtr candidate = std::make_shared<SequenceGroup>(id, seq);

    // 运行检测
    seqCount = 0;
    loadBalancer_->AddSeqGroup(candidate);
    std::this_thread::sleep_for(std::chrono::microseconds(150)); // 定时器周期调度误差
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_EQ(seqCount, 1);
}

TEST_F(LoadBalancerTest, ValidateSequentialAlternationBetweenTriggeredAndPeriodicTasks)
{
    // 初始化准备
    size_t dpSize = 4;
    uint32_t maxPrefillBatchSize = 10;
    SchedulerConfig schedulerConfig;
    SchedulerConfigSPtr schedulerConfigPtr = std::make_shared<SchedulerConfig>(schedulerConfig);
    std::vector<std::shared_ptr<EnginePerDP>> enginePerDPs;
    for (size_t i = 0; i < dpSize; ++i) {
        EnginePerDPSPtr enginePerDP = std::make_shared<EnginePerDP>();
        enginePerDPs.emplace_back(enginePerDP);
        enginePerDP->scheduler = std::make_unique<MockScheduler>();
    }
    LoadBalancerPtr loadBalancer_ = MakeLoadBalancer(enginePerDPs, maxPrefillBatchSize);

    // 准备请求
    constexpr int reqNum = 10;
    size_t seqLen[reqNum] = {18, 3, 22, 15, 7, 11, 24, 9, 1, 17};
    // 构造输入
    std::vector<SequenceGroupSPtr> candidates(reqNum);
    for (int i = 0; i < reqNum; i++) {
        RequestId id(std::to_string(i));
        std::vector<SequenceSPtr> seq(1);
        std::vector<TokenId> inputs(seqLen[i], 5);
        seq[0] = std::make_shared<Sequence>(i, 1, inputs);
        candidates[i] = std::make_shared<SequenceGroup>(id, seq);
    }

    // 大量请求触发快速批处理
    for (auto candidate : candidates) {
        loadBalancer_->AddSeqGroup(candidate);
    }
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait_for(lock, std::chrono::milliseconds(10), [] { return seqCount >= reqNum; });
    }
    EXPECT_EQ(seqCount, reqNum);
    // 少量请求触发周期调度
    loadBalancer_->AddSeqGroup(candidates[0]);
    loadBalancer_->AddSeqGroup(candidates[1]);
    std::this_thread::sleep_for(std::chrono::microseconds(150)); // 定时器周期启动误差
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    int expectSeqCount = reqNum + 2;
    EXPECT_EQ(seqCount, expectSeqCount);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    // 大量请求触发快速批处理
    for (auto candidate : candidates) {
        loadBalancer_->AddSeqGroup(candidate);
    }
    expectSeqCount = reqNum * 2 + 2;
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait_for(lock, std::chrono::milliseconds(50), [expectSeqCount] { return seqCount >= expectSeqCount; });
    }
    EXPECT_EQ(seqCount, expectSeqCount);
}
