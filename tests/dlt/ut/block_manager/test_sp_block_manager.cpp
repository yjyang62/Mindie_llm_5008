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
#include <iostream>
#define private public
#include "block_manager_interface.h"
#include "self_attn_block_manager.h"
#include "block_table.h"
#include "cpu_npu_block_allocator.h"
#undef private

namespace mindie_llm {

SelfAttnBlockManagerSPtr CreateBlockManager(size_t numCpuBlocks = 20, size_t numNpuBlocks = 20, size_t rankSize = 2,
                                            size_t blockSize = 4,
                                            const std::vector<TokenId> &tokenIds = {1, 2, 3, 4, 5, 6, 7, 8},
                                            BlockAllocatorType allocType = BlockAllocatorType::HASHLESS)
{
    BlockManagerConfig config{blockSize, numCpuBlocks, numNpuBlocks, 0, 0, false, 2, 1};
    SelfAttnBlockManagerSPtr blockManager = std::make_shared<SelfAttnBlockManager>(config);
    return blockManager;
}
SequenceGroupSPtr CreateSequenceGroup(SequenceId seqId = 1, RequestId requestId = "rq_1", size_t blockSize = 8,
                                      std::vector<TokenId> inputs = {1, 2, 3, 4, 5, 6, 7, 8},
                                      std::shared_ptr<SamplingParams> sampling = nullptr)
{
    SequenceSPtr seqPtr = std::make_shared<Sequence>(seqId, static_cast<int>(blockSize), inputs);
    std::vector<std::shared_ptr<Sequence>> seqs = {seqPtr};
    return std::make_shared<SequenceGroup>(requestId, seqs, sampling);
}

class BlockManagerSpTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

extern void ValidateTokenIds(std::vector<TokenId> &expectedTokenIds, std::shared_ptr<BlockTable> blockTable);

// 申请16个token的sequence，block size 2，需要8个block，每个rank 4个block，应该返回OK
TEST_F(BlockManagerSpTest, should_return_OK_when_can_allocate_with_2_ranks)
{
    // given
    const size_t numCpuBlocks = 4;
    const size_t numNpuBlocks = 5;
    const size_t blockSize = 2;
    const size_t rankSize = 2;
    std::shared_ptr<SelfAttnBlockManager> blockManager =
        CreateBlockManager(numCpuBlocks, numNpuBlocks, rankSize, blockSize);

    std::vector<TokenId> inputs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    SequenceGroupSPtr sequenceGroup = CreateSequenceGroup(1, "rq_1", 8, inputs);

    // when
    AllocStatus ret = blockManager->CanAllocate(sequenceGroup);

    // then
    EXPECT_EQ(ret, AllocStatus::OK);
}

// 申请20个token的sequence，block size 2，需要10个block，每个rank 4个block，应该返回never
TEST_F(BlockManagerSpTest, should_return_never_when_can_allocate_with_2_ranks)
{
    // given
    const size_t numCpuBlocks = 4;
    const size_t numNpuBlocks = 4;
    const size_t blockSize = 2;
    const size_t rankSize = 2;
    std::shared_ptr<SelfAttnBlockManager> blockManager =
        CreateBlockManager(numCpuBlocks, numNpuBlocks, rankSize, blockSize);

    std::vector<TokenId> inputs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    SequenceGroupSPtr sequenceGroup = CreateSequenceGroup(1, "rq_1", 8, inputs);

    // when
    AllocStatus ret = blockManager->CanAllocate(sequenceGroup);

    // then
    EXPECT_EQ(ret, AllocStatus::NEVER);
}

// 申请16个token的sequence，block size 2，需要8个block，每个rank 4个block，应该返回OK
TEST_F(BlockManagerSpTest, should_return_OK_when_allocate_with_2_ranks)
{
    // given
    const size_t numCpuBlocks = 4;
    const size_t numNpuBlocks = 4;
    const size_t blockSize = 4;
    const size_t rankSize = 2;
    std::shared_ptr<SelfAttnBlockManager> blockManager =
        CreateBlockManager(numCpuBlocks, numNpuBlocks, rankSize, blockSize);

    std::vector<TokenId> inputs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    SequenceGroupSPtr sequenceGroup = CreateSequenceGroup(1, "rq_1", 8, inputs);

    // when
    AllocStatus ret = blockManager->CanAllocate(sequenceGroup);
    bool allocRet = blockManager->Allocate(sequenceGroup);
    std::vector<RankedBlockId> ids;
    blockManager->GetRankedBlockIds(1, ids);

    // then
    EXPECT_EQ(ret, AllocStatus::OK);
    EXPECT_EQ(allocRet, true);
    std::vector<RankedBlockId> expectIds = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

    EXPECT_EQ(ids, expectIds);
}

// block size 4 rank size 2
// given rank 0 [1 2 3 4] [5 6 7 8] rank 1 [9 10 11 12] [13 14 15 16]
// when append 17 18
// then result rank 0 [1 2 3 4] [5 6 7 8] [17] rank 1 [9 10 11 12] [13 14 15 16] [18]
TEST_F(BlockManagerSpTest, should_return_OK_when_full_block_append)
{
    // given
    const size_t numCpuBlocks = 4;
    const size_t numNpuBlocks = 4;
    const size_t blockSize = 4;
    const size_t rankSize = 2;
    std::shared_ptr<SelfAttnBlockManager> blockManager =
        CreateBlockManager(numCpuBlocks, numNpuBlocks, rankSize, blockSize);

    std::vector<TokenId> inputs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    SequenceGroupSPtr sequenceGroup = CreateSequenceGroup(1, "rq_1", 8, inputs);
    bool allocRet = blockManager->Allocate(sequenceGroup);
    sequenceGroup->firstSeq->status_ = SequenceStatus::RUNNING;
    // when
    sequenceGroup->firstSeq->data_.outputTokenIds.push_back(17);
    bool ret1 = blockManager->CanAppendSlotNew(sequenceGroup);
    blockManager->AppendSlotNew(sequenceGroup);

    sequenceGroup->firstSeq->data_.outputTokenIds.push_back(18);
    bool ret2 = blockManager->CanAppendSlotNew(sequenceGroup);
    blockManager->AppendSlotNew(sequenceGroup);

    // then
    EXPECT_EQ(ret1, true);
    EXPECT_EQ(ret2, true);
    EXPECT_EQ(blockManager->seqId2BlockTable_[1].currentSpRank_, 0);
    EXPECT_EQ(blockManager->seqId2BlockTable_[1].numFullSlotsPerRank_[0], 10);
    EXPECT_EQ(blockManager->seqId2BlockTable_[1].numFullSlotsPerRank_[1], 8);
    std::vector<RankedBlockId> ids;
    blockManager->GetRankedBlockIds(1, ids);
    std::vector<RankedBlockId> expectIds = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}};
    EXPECT_EQ(ids, expectIds);
}

// block size 4 rank size 2
// given rank 0 [1 2 3 4] [5 6 7] rank 1 [8 9 10 11] [12 13 14]
// when append 15 16
// then result rank 0 [1 2 3 4] [5 6 7 15] rank 1 [8 9 10 11] [12 13 14 16]
TEST_F(BlockManagerSpTest, should_return_OK_when_paritally_filled_block_append)
{
    // given
    const size_t numCpuBlocks = 4;
    const size_t numNpuBlocks = 4;
    const size_t blockSize = 4;
    const size_t rankSize = 2;
    std::shared_ptr<SelfAttnBlockManager> blockManager =
        CreateBlockManager(numCpuBlocks, numNpuBlocks, rankSize, blockSize);

    std::vector<TokenId> inputs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    SequenceGroupSPtr sequenceGroup = CreateSequenceGroup(1, "rq_1", 8, inputs);
    bool allocRet = blockManager->Allocate(sequenceGroup);
    std::vector<RankedBlockId> ids;
    sequenceGroup->firstSeq->status_ = SequenceStatus::RUNNING;
    
    // when
    sequenceGroup->firstSeq->data_.outputTokenIds.push_back(15);
    bool ret1 = blockManager->CanAppendSlotNew(sequenceGroup);
    blockManager->AppendSlotNew(sequenceGroup);

    sequenceGroup->firstSeq->data_.outputTokenIds.push_back(16);
    bool ret2 = blockManager->CanAppendSlotNew(sequenceGroup);
    blockManager->AppendSlotNew(sequenceGroup);

    // then
    EXPECT_EQ(ret1, true);
    EXPECT_EQ(ret2, true);
    EXPECT_EQ(blockManager->seqId2BlockTable_[1].currentSpRank_, 1);
    EXPECT_EQ(blockManager->seqId2BlockTable_[1].numFullSlotsPerRank_[0], 8);
    EXPECT_EQ(blockManager->seqId2BlockTable_[1].numFullSlotsPerRank_[1], 8);

    blockManager->GetRankedBlockIds(1, ids);
    std::vector<RankedBlockId> expectIds = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    EXPECT_EQ(ids, expectIds);
}

} // namespace mindie_llm