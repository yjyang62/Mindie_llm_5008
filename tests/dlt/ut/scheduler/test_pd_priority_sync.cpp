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
#include <memory>
#include <unistd.h>
#include <sys/wait.h>
#include <vector>
#include "scheduler.h"
#include "config_info.h"
#include "process_group.h"
#include "thread_group_cc.h"

using namespace mindie_llm;
using namespace std;

SchedulerConfigSPtr createSchedulerConfig(bool enableDistributed)
{
    auto config = std::make_shared<SchedulerConfig>();
    config->policyType = 0;
    config->maxSeqLen = 200;
    config->maxPrefillTokens = 200;
    config->maxPrefillBatchSize = 4;
    config->maxBatchSize = 4;
    config->cacheBlockSize = 128;
    config->cpuBlockNum = 300;
    config->npuBlockNum = 300;
    config->spSize = 1;
    config->isMultiNodeInfer = false;
    config->distributedEnable = enableDistributed;

    return config;
}

static SequenceGroupSPtr createDummySeqGroup(std::string &requestId, int promptLength, int blockSize)
{
    std::vector<TokenId> dummyInputs;
    for (int i = 0; i < promptLength; ++i) {
        dummyInputs.push_back(i);
    }

    SequenceSPtr seq = std::make_shared<Sequence>(std::stol(requestId), blockSize, dummyInputs);
    auto samplingParams = std::make_shared<SamplingParams>();
    samplingParams->bestOf = 1;

    std::vector<SequenceSPtr> seqs{seq};
    SequenceGroupSPtr seqGroup = std::make_shared<SequenceGroup>(requestId, seqs, samplingParams);
    seqGroup->seqId2ParallelSeqGroup_.Insert(seqGroup->firstSeq->seqId_, seqGroup);
    seqGroup->parentSeqId_ = seqGroup->firstSeq->seqId_;
    return seqGroup;
}

class PDPrioritySyncTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(PDPrioritySyncTest, ProcGrpSync)
{
    int process_num = 4;
    SchedulerConfigSPtr schedulerConfig = createSchedulerConfig(true);

    for (int i = 1; i < process_num; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            throw runtime_error("fork failed");
        } else if (pid == 0) {
            // create own process group and communicate
            ProcessGroup::GetInstance("127.0.0.1", 2222, "127.0.0.1", i, process_num, false);
            auto predictor = std::make_shared<LatencyPredictor>();
            std::shared_ptr<Scheduler> slaveScheduler = std::make_shared<Scheduler>(schedulerConfig, predictor, Role::PnD, i);
            PDPriorityType pdPriorityType = slaveScheduler->DecidePDPriority(true);
            ASSERT_TRUE(pdPriorityType == PDPriorityType::PREFILL_FIRST);
            return;
        }
    }
    ProcessGroup::GetInstance("127.0.0.1", 2222, "127.0.0.1", 0, process_num, true);
    auto predictor = std::make_shared<LatencyPredictor>();
    std::shared_ptr<Scheduler> schedulerMaster = std::make_shared<Scheduler>(schedulerConfig, predictor, Role::PnD, 0);
    PDPriorityType pdPriorityType = schedulerMaster->DecidePDPriority(true);
    ASSERT_TRUE(pdPriorityType == PDPriorityType::PREFILL_FIRST);
    std::cout << "master process gets pd priority " << (int)pdPriorityType << std::endl;
    wait(NULL);
    std::cout << "Child processes finished. parent ended!" << std::endl;
}

TEST_F(PDPrioritySyncTest, ThreadGrpSync)
{
    size_t localDPs = 4;
    ThreadGroupCC::GetInstance(localDPs);
    std::thread schedulerThread[localDPs];

    SchedulerConfigSPtr schedulerConfig = createSchedulerConfig(false);
    for (size_t dpRank = 0; dpRank < localDPs; ++dpRank) {
        auto predictor = std::make_shared<LatencyPredictor>();
        std::shared_ptr<Scheduler> scheduler = std::make_shared<Scheduler>(schedulerConfig, predictor, Role::PnD, dpRank);
        schedulerThread[dpRank] = std::thread([scheduler, dpRank]() {
            PDPriorityType pdPriorityType = scheduler->DecidePDPriority(true);
            ASSERT_TRUE(pdPriorityType == PDPriorityType::PREFILL_FIRST);
        });
    }

    // 等待子线程完成
    for (size_t dpRank = 0; dpRank < localDPs; ++dpRank) {
        schedulerThread[dpRank].join();
    }
}

TEST_F(PDPrioritySyncTest, TestPDPriorityEdgeCloud)
{
    SchedulerConfigSPtr schedulerConfig = createSchedulerConfig(true);
    auto predictor = std::make_shared<LatencyPredictor>();
    schedulerConfig->stageSelectPolicy = 3;
    schedulerConfig->layerwiseDisaggregated = true;
    std::shared_ptr<Scheduler> Scheduler_ = std::make_shared<Scheduler>(schedulerConfig, predictor, Role::PnD, 0);
    PDPriorityType pdPriorityType = Scheduler_->DecidePDPriority(false);
    EXPECT_EQ(pdPriorityType, PDPriorityType::PREFILL_FIRST);

    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);
    Scheduler_->running_.PushBack(seqGroup);
    pdPriorityType = Scheduler_->DecidePDPriority(false);
    EXPECT_EQ(pdPriorityType, PDPriorityType::DECODE_FIRST);
    
    Scheduler_->swapped_.PushBack(seqGroup);
    pdPriorityType = Scheduler_->DecidePDPriority(false);
    EXPECT_EQ(pdPriorityType, PDPriorityType::DECODE_FIRST);
}