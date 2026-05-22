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
#include <numeric>
#define private public
#include "block_table.h"
#include "cpu_npu_block_allocator.h"
#include "math_utils.h"
#undef private

namespace mindie_llm {
struct BlockTableTestEnv {
    std::vector<TokenId> tokenIds_;
    std::shared_ptr<DeviceAwareBlockAllocator> blockAllocator_;
    std::shared_ptr<BlockTable> blockTable_;
    size_t blockSize_;
};

BlockTableTestEnv CreateBlockTableTestEnv(const std::vector<TokenId> &tokenIds = {1, 2, 3, 4, 5, 6, 7, 8},
                                          BlockAllocatorType allocType = BlockAllocatorType::HASHLESS,
                                          size_t numCpuBlocks = 20, size_t numNpuBlocks = 20, size_t rankSize = 2,
                                          size_t blockSize = 4)
{
    BlockTableTestEnv env;
    env.tokenIds_ = tokenIds;
    AllocatorConfig allocatorConfig = {allocType, numCpuBlocks, numNpuBlocks, blockSize, rankSize};
    env.blockAllocator_ = std::make_shared<CpuNpuBlockAllocator>(allocatorConfig);
    env.blockTable_ = std::make_shared<BlockTable>(blockSize, env.blockAllocator_, rankSize);
    env.blockSize_ = blockSize;
    return env;
}
class BlockTableSpTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

extern void ValidateTokenIds(std::vector<TokenId> &expectedTokenIds, std::shared_ptr<BlockTable> blockTable);

// pd分离时候 AllocateSmallRankFirst 输入 {1, 2, 3} 全部在rank0上
TEST_F(BlockTableSpTest, should_return_rank0_3_tokens_when_allocate_3_tokens_with_SmallRankFirst)
{
    // given
    BlockTableTestEnv env = CreateBlockTableTestEnv({1, 2, 3});

    // when
    env.blockTable_->AllocateSmallRankFirst(env.tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);

    // then
    EXPECT_EQ(env.blockTable_->GetBlockIds().size(), 1);

    EXPECT_EQ(env.blockTable_->GetBlockObjs().at(0)->GetRankIdx(), 0);
    EXPECT_EQ(env.blockTable_->GetBlockObjs().at(0)->GetBlockId(), 0);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[0], 3);

    ValidateTokenIds(env.tokenIds_, env.blockTable_);
}
// pd分离时候 AllocateSmallRankFirst 输入 {1, 2, 3, 4, 5} 在rank0上 {1, 2, 3, 4}, rank 1 {5}
TEST_F(BlockTableSpTest, should_return_rank0_4_tokens_rank1_1_token_when_allocate_5_tokens_with_SmallRankFirst)
{
    // given
    BlockTableTestEnv env = CreateBlockTableTestEnv({1, 2, 3, 4, 5});

    // when
    env.blockTable_->AllocateSmallRankFirst(env.tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);

    // then
    EXPECT_EQ(env.blockTable_->GetBlockIds().size(), 2);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[0], 4);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[1], 1);

    ValidateTokenIds(env.tokenIds_, env.blockTable_);
}
// 测试边界情况 - token数刚好是blockSize的整数倍
TEST_F(BlockTableSpTest, should_handle_exact_block_multiple_with_SmallRankFirst)
{
    // given - 256 tokens (64 blocks)
    std::vector<TokenId> tokens(256);
    std::iota(tokens.begin(), tokens.end(), 1);
    BlockTableTestEnv env = CreateBlockTableTestEnv(tokens, BlockAllocatorType::HASHLESS, 100, 100, 2, 4);

    // when
    env.blockTable_->AllocateSmallRankFirst(env.tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);

    // then
    EXPECT_EQ(env.blockTable_->GetBlockIds().size(), 64);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[0], 128);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[1], 128);
}

// 测试非整数倍blockSize的长token序列
TEST_F(BlockTableSpTest, should_handle_uneven_block_multiple_with_small_rank_first)
{
    std::vector<TokenId> tokens(128001);
    std::iota(tokens.begin(), tokens.end(), 1);
    BlockTableTestEnv env = CreateBlockTableTestEnv(tokens, BlockAllocatorType::HASHLESS, 1000, 1000, 2, 128);

    // when
    env.blockTable_->AllocateSmallRankFirst(env.tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);

    // then
    EXPECT_EQ(env.blockTable_->GetBlockIds().size(), 1001);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[0], 64000);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[1], 64001);
}

// 测试PD分离场景下的切分策略
TEST_F(BlockTableSpTest, should_allocate_1305_tokens_to_8_ranks_with_last_rank_getting_153_tokens)
{
    std::vector<TokenId> tokens(1305);
    std::iota(tokens.begin(), tokens.end(), 1);
    BlockTableTestEnv env = CreateBlockTableTestEnv(tokens, BlockAllocatorType::HASHLESS, 1000, 1000, 8, 128);

    // when
    env.blockTable_->AllocateSmallRankFirst(env.tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);

    // then
    EXPECT_EQ(env.blockTable_->GetBlockIds().size(), 11);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[0], 256);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[1], 256);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[2], 128);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[3], 128);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[4], 128);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[5], 128);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[6], 128);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[7], 153);
}

TEST_F(BlockTableSpTest, should_allocate_380_tokens_to_3_ranks_with_third_rank_getting_124_tokens)
{
    std::vector<TokenId> tokens(380);
    std::iota(tokens.begin(), tokens.end(), 1);
    BlockTableTestEnv env = CreateBlockTableTestEnv(tokens, BlockAllocatorType::HASHLESS, 1000, 1000, 8, 128);

    // when
    env.blockTable_->AllocateSmallRankFirst(env.tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);

    // then
    EXPECT_EQ(env.blockTable_->GetBlockIds().size(), 3);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[0], 128);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[1], 128);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[2], 124);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[3], 0);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[4], 0);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[5], 0);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[6], 0);
    EXPECT_EQ(env.blockTable_->numFullSlotsPerRank_[7], 0);
}
} // namespace mindie_llm