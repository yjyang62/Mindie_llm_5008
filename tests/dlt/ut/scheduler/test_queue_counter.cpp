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
#include "queue_counter.h"
#include "self_attn_block_manager.h"


using namespace mindie_llm;

class QueueCounterTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto blockManager = std::make_shared<SelfAttnBlockManager>(BlockManagerConfig{});
        queueCounter_ = std::make_shared<QueueCounter>(std::make_shared<SchedulerConfig>(), blockManager);
    }

    std::shared_ptr<QueueCounter> queueCounter_;
};

// 测试获取token数量
TEST_F(QueueCounterTest, TestGetNumComputeNewUnCachedAndCachedTokens)
{
    RequestId sgId = "sg";
    std::vector<SequenceSPtr> seqs{std::make_shared<Sequence>(10, 10)};
    auto seqGroup = std::make_shared<SequenceGroup>(sgId, seqs);
    auto result = queueCounter_->GetNumComputeNewUnCachedAndCachedTokens(seqGroup, SequenceStatus::RUNNING);
    EXPECT_EQ(0, result.first);
    EXPECT_EQ(0, result.second);
}

TEST_F(QueueCounterTest, TestCountSequenceGroupSize)
{
    RequestId sgId = "sg";
    std::vector<SequenceSPtr> seqs{std::make_shared<Sequence>(10, 10)};
    auto seqGroup = std::make_shared<SequenceGroup>(sgId, seqs);
    EXPECT_EQ(1, queueCounter_->CountSequenceGroupSize(seqGroup));
}

// 测试获取token数量
TEST_F(QueueCounterTest, TestCountTokens)
{
    RequestId sgId = "sg";
    std::vector<SequenceSPtr> seqs{std::make_shared<Sequence>(10, 10)};
    auto seqGroup = std::make_shared<SequenceGroup>(sgId, seqs);
    EXPECT_EQ(0, queueCounter_->CountTokens(seqGroup, SequenceStatus::RUNNING));
}

// 测试获取计算阶段的token数量
TEST_F(QueueCounterTest, TestCountRunningTokens)
{
    RequestId sgId = "sg";
    std::vector<SequenceSPtr> seqs{std::make_shared<Sequence>(10, 10)};
    auto seqGroup = std::make_shared<SequenceGroup>(sgId, seqs);
    EXPECT_EQ(0, queueCounter_->CountRunningTokens(seqGroup));
}

// 测试获取block的数量
TEST_F(QueueCounterTest, TestGetNumRequiredBlocks)
{
    size_t seqLen = 10, blockSize = 2;
    EXPECT_EQ(5, queueCounter_->GetNumRequiredBlocks(seqLen, blockSize));
}

// 测试获取block的数量
TEST_F(QueueCounterTest, TestCountBlocks)
{
    RequestId sgId = "sg";
    std::vector<SequenceSPtr> seqs{std::make_shared<Sequence>(10, 10)};
    auto seqGroup = std::make_shared<SequenceGroup>(sgId, seqs);
    // 验证抛出 std::runtime_error
    EXPECT_THROW(
        queueCounter_->CountBlocks(seqGroup, SequenceStatus::WAITING),
        std::runtime_error
    );
}