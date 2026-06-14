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
#include <cmath>
#include <unordered_map>
#include <cassert>
#define private public
#include "self_attn_block_manager.h"
#include "sequence_group.h"
#include "prefix_cache_block_allocator.h"
#include "policy_helper.h"
#undef private

namespace mindie_llm {

class SelfAttnBlockManagerTest : public ::testing::Test {
protected:
    void SetUp() override {}
};
std::shared_ptr<BlockSpaceManager> CreateSelfAttnBlockManager(size_t blockSize = 8, size_t cpuBlockNum = 2,
                                                              size_t npuBlockNum = 2, bool enableCaching = true,
                                                              size_t reservedBlockNum = 0, size_t speculativeSlots = 0)
{
    BlockManagerConfig config{blockSize, cpuBlockNum, npuBlockNum, reservedBlockNum, speculativeSlots, enableCaching};
    SelfAttnBlockManager blockManager(config);
    return std::make_shared<SelfAttnBlockManager>(blockManager);
}

SequenceGroupSPtr CreateSequenceGroup(SequenceId seqId = 1, RequestId requestId = "rq_1", size_t blockSize = 8,
                                      std::vector<TokenId> inputs = {1, 2, 3, 4, 5, 6, 7, 8},
                                      std::shared_ptr<SamplingParams> sampling = std::make_shared<SamplingParams>())
{
    SequenceSPtr seqPtr = std::make_shared<Sequence>(seqId, static_cast<int>(blockSize), inputs);
    std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
    SequenceGroupSPtr seqGrpSPtr = std::make_shared<SequenceGroup>(requestId, seqs, sampling);
    seqGrpSPtr->seqId2ParallelSeqGroup_.Insert(seqGrpSPtr->firstSeq->seqId_, seqGrpSPtr);
    seqGrpSPtr->parentSeqId_ = seqGrpSPtr->firstSeq->seqId_;
    return seqGrpSPtr;
}

TEST_F(SelfAttnBlockManagerTest, should_return_blockId_balance_after_swap_and_free)
{
    // given
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager();

    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();
    PolicyHelper policyHelper{schedulerConfig, blockManager};

    SequenceGroupSPtr groupPtr = CreateSequenceGroup();

    // when
    auto ret = policyHelper.CanAppendSlots(groupPtr);
    policyHelper.AllocateAndSetRunning(groupPtr);

    std::vector<std::pair<BlockId, BlockId>> blockToCopy;
    policyHelper.AppendSlots(groupPtr, blockToCopy);
    size_t npuNumAfterAppend = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterAppend = blockManager->GetNumFreeCpuBlocks();

    policyHelper.SwapOut(groupPtr, blockToCopy);
    size_t npuNumAfterSwapOut = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterSwapOut = blockManager->GetNumFreeCpuBlocks();

    policyHelper.SwapIn(groupPtr, blockToCopy);
    size_t npuNumAfterSwapIn = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterSwapIn = blockManager->GetNumFreeCpuBlocks();

    std::vector<SequenceSPtr> seqs;
    seqs.emplace_back(groupPtr->firstSeq);
    blockManager->GetCommonComputedBlockIds(seqs);
    policyHelper.FreeSeqGroup(groupPtr);
    size_t npuNumAfterFreeSeqGroup = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterFreeSeqGroup = blockManager->GetNumFreeCpuBlocks();

    // then
    EXPECT_EQ(npuNumAfterAppend, 1);
    EXPECT_EQ(cpuNumAfterAppend, 2);

    EXPECT_EQ(npuNumAfterSwapOut, 2);
    EXPECT_EQ(cpuNumAfterSwapOut, 1);

    EXPECT_EQ(npuNumAfterSwapIn, 1);
    EXPECT_EQ(cpuNumAfterSwapIn, 2);

    EXPECT_EQ(npuNumAfterFreeSeqGroup, 2);
    EXPECT_EQ(cpuNumAfterFreeSeqGroup, 2);
}

TEST_F(SelfAttnBlockManagerTest, should_return_token_uncached_when_append_placeholder_token)
{
    // given
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager(8, 6, 6);

    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();
    PolicyHelper policyHelper{schedulerConfig, blockManager};

    SequenceGroupSPtr groupPtr = CreateSequenceGroup(1, "rq_1", 8, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    // when
    policyHelper.AllocateAndSetRunning(groupPtr);

    groupPtr->firstSeq->data_.promptTokenIds.emplace_back(PLACEHOLDER_TOKEN);
    std::vector<std::pair<BlockId, BlockId>> blockToCopy;
    auto ret = policyHelper.CanAppendSlots(groupPtr);
    policyHelper.AppendSlots(groupPtr, blockToCopy);

    blockManager->MarkBlocksAsComputed();

    size_t npuNumAfterAppend = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterAppend = blockManager->GetNumFreeCpuBlocks();

    size_t cachedTkoens = 0;
    cachedTkoens += blockManager->GetNumCachedTokens(groupPtr->firstSeq);

    // then
    EXPECT_EQ(npuNumAfterAppend, 4);
    EXPECT_EQ(cpuNumAfterAppend, 6);
    // 插入占位符的token不参与kv cache
    EXPECT_EQ(cachedTkoens, 8);
}

/* should_return_token_cached_when_append_valid_token 和
should_return_token_uncached_when_append_placeholder_token两个用例
对比在插入 占位符token 和 有效tokende 时候，插入占位符的token不参与kv cache。*/
TEST_F(SelfAttnBlockManagerTest, should_return_token_cached_when_append_valid_token)
{
    // given
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager(8, 6, 6);

    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();
    PolicyHelper policyHelper{schedulerConfig, blockManager};

    SequenceGroupSPtr groupPtr = CreateSequenceGroup(1, "rq_1", 8, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    // when
    policyHelper.AllocateAndSetRunning(groupPtr);

    groupPtr->firstSeq->data_.promptTokenIds.emplace_back(16);
    std::vector<std::pair<BlockId, BlockId>> blockToCopy;
    auto ret = policyHelper.CanAppendSlots(groupPtr);
    policyHelper.AppendSlots(groupPtr, blockToCopy);

    blockManager->MarkBlocksAsComputed();

    size_t npuNumAfterAppend = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterAppend = blockManager->GetNumFreeCpuBlocks();

    size_t cachedTkoens = 0;
    cachedTkoens += blockManager->GetNumCachedTokens(groupPtr->firstSeq);

    // then
    EXPECT_EQ(npuNumAfterAppend, 4);
    EXPECT_EQ(cpuNumAfterAppend, 6);
    // 有效token参与kv cache
    EXPECT_EQ(cachedTkoens, 16);
}

TEST_F(SelfAttnBlockManagerTest, should_return_blockId_except_when_batch_process_without_same_tokens)
{
    // given
    size_t cpuBlockNum = 256;
    size_t npuBlockNum = 128;
    size_t allocatedCnt = 0;
    size_t appendedCnt = 0;
    size_t allocatedFailCnt = 0;
    size_t appendedFailCnt = 0;
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager(8, cpuBlockNum, npuBlockNum);

    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();
    PolicyHelper policyHelper{schedulerConfig, blockManager};

    std::vector<SequenceGroupSPtr> groupPtrs;
    const int batchSize = 63;
    int sequenceCnt = batchSize * 20;
    // 提前准备好1024个groupPtr放到vector中
    for (int i = 1; i <= sequenceCnt; ++i) {
        std::vector<TokenId> inputs;
        for (int j = 1; j <= 15; ++j) {
            inputs.push_back((i - 1) * 15 + j);
        }
        SequenceGroupSPtr groupPtr = CreateSequenceGroup(i, "rq_" + std::to_string(i), 8, inputs);
        groupPtrs.push_back(groupPtr);
    }

    for (size_t i = 0; i < groupPtrs.size(); i += batchSize) {
        size_t end = std::min(i + batchSize, groupPtrs.size());
        std::vector<SequenceGroupSPtr> currentBatch(groupPtrs.begin() + i, groupPtrs.begin() + end);

        // 对当前批次执行alloc操作
        for (auto groupPtr : currentBatch) {
            if (blockManager->CanAllocate(groupPtr) == AllocStatus::OK) {
                policyHelper.AllocateAndSetRunning(groupPtr);
                allocatedCnt++;
            } else {
                allocatedFailCnt++;
            }
        }

        // then
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum - batchSize * 2);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum);

        // 对当前批次执行append操作
        for (auto groupPtr : currentBatch) {
            groupPtr->firstSeq->data_.promptTokenIds.emplace_back(16);
            if (policyHelper.CanAppendSlots(groupPtr) == true) {
                std::vector<std::pair<BlockId, BlockId>> blockToCopy;
                policyHelper.AppendSlots(groupPtr, blockToCopy);
                blockManager->MarkBlocksAsComputed();
                appendedCnt++;
            } else {
                appendedFailCnt++;
            }
            blockManager->GetNumCachedTokens(groupPtr->firstSeq);
        }

        // then
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum - batchSize * 2);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum);

        for (auto groupPtr : currentBatch) {
            std::vector<std::pair<BlockId, BlockId>> blockToCopy;
            policyHelper.SwapOut(groupPtr, blockToCopy);
        }
        // then SwapOut后的检查
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum - batchSize * 2);

        for (auto groupPtr : currentBatch) {
            std::vector<std::pair<BlockId, BlockId>> blockToCopy;
            policyHelper.SwapIn(groupPtr, blockToCopy);
        }
        // then SwapIn后的检查
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum - batchSize * 2);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum);

        // 对当前批次统一执行free操作
        for (auto groupPtr : currentBatch) {
            policyHelper.FreeSeqGroup(groupPtr);
        }

        // then
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum);
    }
    EXPECT_EQ(allocatedCnt, sequenceCnt);
    EXPECT_EQ(appendedCnt, sequenceCnt);
}

// block obj = ref cnt all
// blockid ref cnt == 有多少个blockobj使用 这个blockid
TEST_F(SelfAttnBlockManagerTest, should_return_blockId_except_when_batch_process_with_same_tokens)
{
    // given
    size_t cpuBlockNum = 256;
    size_t npuBlockNum = 128;
    size_t allocatedCnt = 0;
    size_t appendedCnt = 0;
    size_t allocatedFailCnt = 0;
    size_t appendedFailCnt = 0;
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager(8, cpuBlockNum, npuBlockNum);

    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();
    PolicyHelper policyHelper{schedulerConfig, blockManager};

    std::vector<SequenceGroupSPtr> groupPtrs;
    const int batchSize = 64;
    int sequenceCnt = batchSize * 16;

    // 提前准备好1024个groupPtr放到vector中
    for (int i = 0; i < sequenceCnt; i++) {
        std::vector<TokenId> inputs;
        // 判断是否是每batchSize中的25%
        if (i % 4 == 0) {
            for (int j = 1; j <= 15; ++j) {
                inputs.push_back(j + 0x12345678);
            }
        } else {
            for (int j = 1; j <= 15; ++j) {
                inputs.push_back((i - 1) * 15 + j);
            }
        }
        SequenceGroupSPtr groupPtr = CreateSequenceGroup(i, "rq_" + std::to_string(i), 8, inputs);
        groupPtrs.push_back(groupPtr);
    }

    for (size_t i = 0; i < groupPtrs.size(); i += batchSize) {
        size_t end = std::min(i + batchSize, groupPtrs.size());
        std::vector<SequenceGroupSPtr> currentBatch(groupPtrs.begin() + i, groupPtrs.begin() + end);

        // 对当前批次执行alloc操作
        for (auto groupPtr : currentBatch) {
            if (blockManager->CanAllocate(groupPtr) == AllocStatus::OK) {
                policyHelper.AllocateAndSetRunning(groupPtr);

                // 这里要更新一下时间 不然evitor会老化错误导致异常
                for (const auto &seq : groupPtr->GetSequences(SequenceStatus::RUNNING)) {
                    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    blockManager->AccessAllblocksInSeq(groupPtr->firstSeq, now);
                }

                allocatedCnt++;
            } else {
                allocatedFailCnt++;
            }
        }

        // then batchSize = 64,其中1/4 = 16个的seq token1-15是一样的，但是不是全满，所以只有token1-8是可以计算hash的
        // 所以npuBlockNum - batchSize * 2 + batchSize/4 - 1
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum - batchSize * 2 + batchSize / 4 - 1);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum);

        // 对当前批次执行append操作
        for (auto groupPtr : currentBatch) {
            groupPtr->firstSeq->data_.promptTokenIds.emplace_back(16);
            if (policyHelper.CanAppendSlots(groupPtr) == true) {
                std::vector<std::pair<BlockId, BlockId>> blockToCopy;
                policyHelper.AppendSlots(groupPtr, blockToCopy);
                blockManager->MarkBlocksAsComputed();
                appendedCnt++;
            } else {
                appendedFailCnt++;
            }
            blockManager->GetNumCachedTokens(groupPtr->firstSeq);
        }

        // then append之后的，之前只有8个token相同并且full，现在有16个token了。所以有 batchSize * 2 / 4 - 2个blockid复用
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum - batchSize * 2 + batchSize * 2 / 4 - 2);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum);

        for (auto groupPtr : currentBatch) {
            std::vector<std::pair<BlockId, BlockId>> blockToCopy;
            policyHelper.SwapOut(groupPtr, blockToCopy);
        }
        // then SwapOut后的检查
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum - batchSize * 2 + batchSize * 2 / 4 - 2);

        for (auto groupPtr : currentBatch) {
            std::vector<std::pair<BlockId, BlockId>> blockToCopy;
            policyHelper.SwapIn(groupPtr, blockToCopy);
        }
        // then SwapIn后的检查
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum - batchSize * 2 + batchSize * 2 / 4 - 2);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum);

        // 对当前批次统一执行free操作
        for (auto groupPtr : currentBatch) {
            policyHelper.FreeSeqGroup(groupPtr);
        }

        // then
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), npuBlockNum);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), cpuBlockNum);
    }
    EXPECT_EQ(allocatedCnt, sequenceCnt);
    EXPECT_EQ(appendedCnt, sequenceCnt);
}

// 测试swap out后allocater是否正确使用
TEST_F(SelfAttnBlockManagerTest, should_return_blockId_balance_after_swap_out_and_free)
{
    // given
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager();

    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();
    PolicyHelper policyHelper{schedulerConfig, blockManager};

    SequenceGroupSPtr groupPtr = CreateSequenceGroup();

    // when
    auto ret = policyHelper.CanAppendSlots(groupPtr);
    policyHelper.AllocateAndSetRunning(groupPtr);

    std::vector<std::pair<BlockId, BlockId>> blockToCopy;
    policyHelper.AppendSlots(groupPtr, blockToCopy);
    size_t npuNumAfterAppend = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterAppend = blockManager->GetNumFreeCpuBlocks();

    policyHelper.SwapOut(groupPtr, blockToCopy);
    size_t npuNumAfterSwapOut = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterSwapOut = blockManager->GetNumFreeCpuBlocks();

    std::vector<SequenceSPtr> seqs;
    seqs.emplace_back(groupPtr->firstSeq);
    blockManager->GetCommonComputedBlockIds(seqs);
    policyHelper.FreeSeqGroup(groupPtr);
    size_t npuNumAfterFreeSeqGroup = blockManager->GetNumFreeNpuBlocks();
    size_t cpuNumAfterFreeSeqGroup = blockManager->GetNumFreeCpuBlocks();

    // then
    EXPECT_EQ(npuNumAfterAppend, 1);
    EXPECT_EQ(cpuNumAfterAppend, 2);

    EXPECT_EQ(npuNumAfterSwapOut, 2);
    EXPECT_EQ(cpuNumAfterSwapOut, 1);

    EXPECT_EQ(npuNumAfterFreeSeqGroup, 2);
    EXPECT_EQ(cpuNumAfterFreeSeqGroup, 2);
}
void RunBatchTest(size_t batchSize, std::shared_ptr<BlockSpaceManager> blockManager,
                  std::shared_ptr<PolicyHelper> policyHelper)
{
    size_t allocatedCnt = 0;
    size_t appendedCnt = 0;
    size_t allocatedFailCnt = 0;
    size_t appendedFailCnt = 0;

    std::vector<SequenceGroupSPtr> groupPtrs;
    int sequenceCnt = batchSize * 16;

    // 提前准备好1024个groupPtr放到vector中
    for (int i = 0; i < sequenceCnt; i++) {
        std::vector<TokenId> inputs;
        // 判断是否是每batchSize中的25%
        if (i % 4 == 0) {
            for (int j = 1; j <= 15; ++j) {
                inputs.push_back(j + 0x12345678);
            }
        } else {
            for (int j = 1; j <= 15; ++j) {
                inputs.push_back((i - 1) * 15 + j);
            }
        }
        SequenceGroupSPtr groupPtr = CreateSequenceGroup(i, "rq_" + std::to_string(i), 8, inputs);
        groupPtrs.push_back(groupPtr);
    }

    for (size_t i = 0; i < groupPtrs.size(); i += batchSize) {
        size_t end = std::min(i + batchSize, groupPtrs.size());
        std::vector<SequenceGroupSPtr> currentBatch(groupPtrs.begin() + i, groupPtrs.begin() + end);

        // 对当前批次执行alloc操作
        for (auto groupPtr : currentBatch) {
            if (blockManager->CanAllocate(groupPtr) == AllocStatus::OK) {
                // 修改调用方式，使用指针调用成员函数
                policyHelper->AllocateAndSetRunning(groupPtr);
                allocatedCnt++;
            } else {
                allocatedFailCnt++;
            }
        }

        // then batchSize = 64,其中1/4 = 16个的seq token1-15是一样的，但是不是全满，所以只有token1-8是可以计算hash的
        // 所以npuBlockNum - batchSize * 2 + batchSize/4 - 1
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), 128 - batchSize * 2 + batchSize / 4 - 1);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), 256);

        // 对当前批次执行append操作
        for (auto groupPtr : currentBatch) {
            groupPtr->firstSeq->data_.promptTokenIds.emplace_back(16);
            // 修改调用方式，使用指针调用成员函数
            if (policyHelper->CanAppendSlots(groupPtr) == true) {
                std::vector<std::pair<BlockId, BlockId>> blockToCopy;
                // 修改调用方式，使用指针调用成员函数
                policyHelper->AppendSlots(groupPtr, blockToCopy);
                blockManager->MarkBlocksAsComputed();
                appendedCnt++;
            } else {
                appendedFailCnt++;
            }
            blockManager->GetNumCachedTokens(groupPtr->firstSeq);
        }

        // then append之后的，之前只有8个token相同并且full，现在有16个token了。所以有 batchSize * 2 / 4 - 2个blockid复用
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), 128 - batchSize * 2 + batchSize * 2 / 4 - 2);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), 256);

        for (auto groupPtr : currentBatch) {
            std::vector<std::pair<BlockId, BlockId>> blockToCopy;
            // 修改调用方式，使用指针调用成员函数
            policyHelper->SwapOut(groupPtr, blockToCopy);
        }
        // then SwapOut后的检查
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), 128);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), 256 - batchSize * 2 + batchSize * 2 / 4 - 2);

        for (auto groupPtr : currentBatch) {
            std::vector<std::pair<BlockId, BlockId>> blockToCopy;
            // 修改调用方式，使用指针调用成员函数
            policyHelper->SwapIn(groupPtr, blockToCopy);
        }
        // then SwapIn后的检查
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), 128 - batchSize * 2 + batchSize * 2 / 4 - 2);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), 256);

        // 对当前批次统一执行free操作
        for (auto groupPtr : currentBatch) {
            // 修改调用方式，使用指针调用成员函数
            policyHelper->FreeSeqGroup(groupPtr);
        }

        // then
        EXPECT_EQ(blockManager->GetNumFreeNpuBlocks(), 128);
        EXPECT_EQ(blockManager->GetNumFreeCpuBlocks(), 256);
    }
    EXPECT_EQ(allocatedCnt, sequenceCnt);
    EXPECT_EQ(appendedCnt, sequenceCnt);
}

TEST_F(SelfAttnBlockManagerTest, should_return_blockId_except_when_batch_process_with_same_tokens_in_range)
{
    std::vector<size_t> batchSizes = {4, 8, 12, 16, 20, 24, 28, 32, 64, 68, 72};
    size_t cpuBlockNum = 256;
    size_t npuBlockNum = 128;
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager(8, cpuBlockNum, npuBlockNum);
    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();

    std::shared_ptr<PolicyHelper> policyHelper = std::make_shared<PolicyHelper>(schedulerConfig, blockManager);
    for (size_t batchSize : batchSizes) {
        RunBatchTest(batchSize, blockManager, policyHelper);
    }
}

TEST_F(SelfAttnBlockManagerTest, should_return_token_cached_when_relpace_placeholder_token)
{
    // given
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager(8, 6, 6);

    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();
    PolicyHelper policyHelper{schedulerConfig, blockManager};

    SequenceGroupSPtr groupPtr1 =
        CreateSequenceGroup(1, "rq_1", 8, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

    SequenceGroupSPtr groupPtr2 =
        CreateSequenceGroup(2, "rq_2", 8, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    // when
    policyHelper.AllocateAndSetRunning(groupPtr1);
    policyHelper.AllocateAndSetRunning(groupPtr2);

    // rq_2 先插入PLACEHOLDER_TOKEN
    groupPtr2->firstSeq->data_.outputTokenIds.emplace_back(PLACEHOLDER_TOKEN);
    std::vector<std::pair<BlockId, BlockId>> blockToCopy;
    auto ret = policyHelper.CanAppendSlots(groupPtr2);
    policyHelper.AppendSlots(groupPtr2, blockToCopy);

    blockManager->MarkBlocksAsComputed();

    size_t npuNumAfterAppend = blockManager->GetNumFreeNpuBlocks();
    size_t cachedTkoensAfterAppend = blockManager->GetNumCachedTokens(groupPtr2->firstSeq);

    // rq_2 再替换PLACEHOLDER_TOKEN为有效值
    groupPtr2->firstSeq->data_.outputTokenIds[0] = 16;
    blockManager->ReplaceTrailingPlaceHolder(groupPtr2->seqs_[0], 1, 1);
    blockManager->MarkBlocksAsComputed();
    groupPtr2->seqs_[0]->data_.stage_ = SequenceStage::DECODE;

    // then
    size_t npuNumAfterReplace = blockManager->GetNumFreeNpuBlocks();
    size_t cachedTkoensAfterReplace = blockManager->GetNumCachedTokens(groupPtr2->firstSeq);
    // 插入占位符的token不参与kv cache
    EXPECT_EQ(npuNumAfterAppend, 3);
    EXPECT_EQ(cachedTkoensAfterAppend, 8);
    // 占位符变为有效的token后参与kv cache
    EXPECT_EQ(npuNumAfterReplace, 4);
    EXPECT_EQ(cachedTkoensAfterReplace, 16);
}

TEST_F(SelfAttnBlockManagerTest, should_return_token_cached_when_replace_two_placeholders)
{
    // given
    // 配置每个 block 存放的数量为 1
    std::shared_ptr<BlockSpaceManager> blockManager = CreateSelfAttnBlockManager(1, 32, 32);

    SchedulerConfigSPtr schedulerConfig = std::make_shared<SchedulerConfig>();
    PolicyHelper policyHelper{schedulerConfig, blockManager};

    SequenceGroupSPtr groupPtr1 =
        CreateSequenceGroup(1, "rq_1", 1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

    SequenceGroupSPtr groupPtr2 = CreateSequenceGroup(2, "rq_2", 1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});

    // when
    policyHelper.AllocateAndSetRunning(groupPtr1);
    policyHelper.AllocateAndSetRunning(groupPtr2);

    // rq_2 插入两个 PLACEHOLDER_TOKEN
    groupPtr2->firstSeq->data_.outputTokenIds.emplace_back(PLACEHOLDER_TOKEN);
    groupPtr2->firstSeq->data_.outputTokenIds.emplace_back(PLACEHOLDER_TOKEN);

    std::vector<std::pair<BlockId, BlockId>> blockToCopy;
    auto ret = policyHelper.CanAppendSlots(groupPtr2);
    policyHelper.AppendSlots(groupPtr2, blockToCopy);

    blockManager->MarkBlocksAsComputed();

    size_t npuNumAfterAppend = blockManager->GetNumFreeNpuBlocks();
    size_t cachedTkoensAfterAppend = blockManager->GetNumCachedTokens(groupPtr2->firstSeq);

    // rq_2 替换两个 PLACEHOLDER_TOKEN 为有效值
    groupPtr2->firstSeq->data_.outputTokenIds[0] = 15;
    groupPtr2->firstSeq->data_.outputTokenIds[1] = 16;
    blockManager->ReplaceTrailingPlaceHolder(groupPtr2->seqs_[0], 2, 2);
    blockManager->MarkBlocksAsComputed();
    groupPtr2->seqs_[0]->data_.stage_ = SequenceStage::DECODE;

    // then
    size_t npuNumAfterReplace = blockManager->GetNumFreeNpuBlocks();
    size_t cachedTkoensAfterReplace = blockManager->GetNumCachedTokens(groupPtr2->firstSeq);
    // 插入占位符的 token 不参与 kv cache
    EXPECT_EQ(npuNumAfterAppend, 32 - 16 - 2);
    EXPECT_EQ(cachedTkoensAfterAppend, 14);
    // 占位符变为有效的 token 后参与 kv cache
    EXPECT_EQ(npuNumAfterReplace, 16);
    EXPECT_EQ(cachedTkoensAfterReplace, 16);
}

} // namespace mindie_llm
