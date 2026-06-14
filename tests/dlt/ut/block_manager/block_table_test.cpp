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
#include "block_table.h"
#include "cpu_npu_block_allocator.h"
#include "math_utils.h"

namespace mindie_llm {
// validate tokenIds in blockObj from blockTable
void ValidateTokenIds(std::vector<TokenId> &expectedTokenIds, std::shared_ptr<BlockTable> blockTable)
{
    size_t tokenIdsNum{0};
    std::vector<TokenId> tokenIds{};
    for (const auto &blockObj : blockTable->GetBlockObjs()) {
        tokenIdsNum += blockObj->GetTokenIds().size();
        tokenIds.insert(tokenIds.end(), blockObj->GetTokenIds().begin(), blockObj->GetTokenIds().end());
    }
    EXPECT_EQ(tokenIdsNum, expectedTokenIds.size());
    EXPECT_EQ(tokenIds, expectedTokenIds);
}

class BlockTableTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        tokenIds_ = {1, 2, 3, 4, 5, 6};
        allocatorType_ = BlockAllocatorType::HASHLESS;
        numCpuBlocks_ = 20;
        numNpuBlocks_ = 20;
        blockSize_ = 4;
        AllocatorConfig allocatorConfig = {allocatorType_, numCpuBlocks_, numNpuBlocks_, blockSize_};
        blockAllocator_ = std::make_shared<CpuNpuBlockAllocator>(allocatorConfig);
        blockTable_ = std::make_shared<BlockTable>(blockSize_, blockAllocator_);
    }

    std::vector<TokenId> tokenIds_;
    BlockAllocatorType allocatorType_;
    size_t numCpuBlocks_;
    size_t numNpuBlocks_;
    size_t blockSize_;
    std::shared_ptr<DeviceAwareBlockAllocator> blockAllocator_;
    std::shared_ptr<BlockTable> blockTable_;
};

TEST_F(BlockTableTest, TestChunkTokensForAllocate)
{
    std::vector<std::vector<TokenId>> result = BlockTable::ChunkTokensForAllocate(tokenIds_, blockSize_);
    std::vector<std::vector<TokenId>> expectdeResult = {{1, 2, 3, 4}, {5, 6}};
    EXPECT_EQ(result, expectdeResult);
}

TEST_F(BlockTableTest, ShouldThrowWhenChunkSizeIsZero)
{
    blockSize_ = 0;
    EXPECT_THROW(BlockTable::ChunkTokensForAllocate(tokenIds_, blockSize_), std::runtime_error);
}

TEST_F(BlockTableTest, TestAllocate)
{
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);
    // validate fullslot num
    EXPECT_EQ(blockTable_->GetNumFullSlots(), tokenIds_.size());
    // validate block num
    size_t expectedBlockSize = CeilDiv(tokenIds_.size(), blockSize_);
    EXPECT_EQ(blockTable_->GetBlockIds().size(), expectedBlockSize);
    EXPECT_EQ(blockTable_->GetBlockObjs().size(), expectedBlockSize);
    // validate ids between blockids and blockObjs
    for (size_t i = 0; i < expectedBlockSize; i++) {
        EXPECT_EQ(blockTable_->GetBlockIds().at(i), blockTable_->GetBlockObjs().at(i)->GetBlockId());
    }
    // validate prev block
    BlockObjSPtr prevBlock = nullptr;
    for (const auto &blockObj : blockTable_->GetBlockObjs()) {
        EXPECT_EQ(blockObj->GetPrevBlock(), prevBlock);
        prevBlock = blockObj;
    }
    // validate tokenIds
    std::vector<TokenId> expectedTokenIds(tokenIds_);
    ValidateTokenIds(expectedTokenIds, blockTable_);
}

TEST_F(BlockTableTest, ShouldThrowWhenAllocateTwice)
{
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);
    EXPECT_THROW(blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE), std::runtime_error);
}

TEST_F(BlockTableTest, TestAppendWhenTokenIdsSizeGreaterThanEmptySlots)
{
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);
    std::vector<TokenId> appendTokenIds = {7, 8, 9};
    blockTable_->AppendTokenIds(appendTokenIds, INVALID_HASH_VALUE, 0);
    // validate fullslot num
    EXPECT_EQ(blockTable_->GetNumFullSlots(), tokenIds_.size() + appendTokenIds.size());
    // validate block num
    size_t expectedBlockSize = CeilDiv(tokenIds_.size() + appendTokenIds.size(), blockSize_);
    EXPECT_EQ(blockTable_->GetBlockIds().size(), expectedBlockSize);
    EXPECT_EQ(blockTable_->GetBlockObjs().size(), expectedBlockSize);
    for (size_t i = 0; i < expectedBlockSize; i++) {
        // validate ids between blockids and blockObjs
        EXPECT_EQ(blockTable_->GetBlockIds().at(i), blockTable_->GetBlockObjs().at(i)->GetBlockId());
    }
    // validate prev block
    BlockObjSPtr prevBlock = nullptr;
    for (const auto &blockObj : blockTable_->GetBlockObjs()) {
        EXPECT_EQ(blockObj->GetPrevBlock(), prevBlock);
        prevBlock = blockObj;
    }
    // validate tokenIds
    std::vector<TokenId> expectedTokenIds(tokenIds_);
    expectedTokenIds.insert(expectedTokenIds.end(), appendTokenIds.begin(), appendTokenIds.end());
    ValidateTokenIds(expectedTokenIds, blockTable_);
}

TEST_F(BlockTableTest, TestAppendWhenTokenIdsSizeLessThanEmptySlots)
{
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);
    std::vector<TokenId> appendTokenIds = {7};
    blockTable_->AppendTokenIds(appendTokenIds, INVALID_HASH_VALUE, 0);
    // validate fullslot num
    EXPECT_EQ(blockTable_->GetNumFullSlots(), tokenIds_.size() + appendTokenIds.size());
    // validate block num
    size_t expectedBlockSize = CeilDiv(tokenIds_.size() + appendTokenIds.size(), blockSize_);
    EXPECT_EQ(blockTable_->GetBlockIds().size(), expectedBlockSize);
    EXPECT_EQ(blockTable_->GetBlockObjs().size(), expectedBlockSize);
    // validate ids between blockids and blockObjs
    for (size_t i = 0; i < expectedBlockSize; i++) {
        EXPECT_EQ(blockTable_->GetBlockIds().at(i), blockTable_->GetBlockObjs().at(i)->GetBlockId());
    }
    // validate prev block
    BlockObjSPtr prevBlock = nullptr;
    for (const auto &blockObj : blockTable_->GetBlockObjs()) {
        EXPECT_EQ(blockObj->GetPrevBlock(), prevBlock);
        prevBlock = blockObj;
    }
    // validate tokenIds
    std::vector<TokenId> expectedTokenIds(tokenIds_);
    expectedTokenIds.insert(expectedTokenIds.end(), appendTokenIds.begin(), appendTokenIds.end());
    ValidateTokenIds(expectedTokenIds, blockTable_);
}

TEST_F(BlockTableTest, ShouldThrowWhenAppendToUnallocatedBlockTable)
{
    EXPECT_THROW(blockTable_->AppendTokenIds(tokenIds_, INVALID_HASH_VALUE, 0), std::runtime_error);
}

TEST_F(BlockTableTest, ShouldClearBlockTableWhenFree)
{
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE);
    blockTable_->Free();
    EXPECT_EQ(blockTable_->GetNumFullSlots(), 0);
    EXPECT_EQ(blockTable_->GetBlockIds().size(), 0);
    EXPECT_EQ(blockTable_->GetBlockObjs().size(), 0);
}

TEST_F(BlockTableTest, ShouldReturnOneWhenBlockIsFullGiventokenIdsSizeLessThanBlockSize)
{
    tokenIds_ = {1, 2, 3, 4};
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE); // EmptySlots = 0
    size_t tokenIdsSize = blockSize_ - 1;
    EXPECT_EQ(blockTable_->GetNumRelatedBlocks(tokenIdsSize, 0), 1);
}

TEST_F(BlockTableTest, ShouldReturnTwoWhenBlockIsFullGiventokenIdsSizeGreaterThanBlockSize)
{
    tokenIds_ = {1, 2, 3, 4};
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE); // EmptySlots = 0
    size_t tokenIdsSize = blockSize_ + 1;
    EXPECT_EQ(blockTable_->GetNumRelatedBlocks(tokenIdsSize, 0), 2);
}

TEST_F(BlockTableTest, ShouldReturnOneWhenBlockIsNotFullGiventokenIdsSizeLessThanEmptySlots)
{
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE); // EmptySlots = 2
    size_t tokenIdsSize = 1;
    EXPECT_EQ(blockTable_->GetNumRelatedBlocks(tokenIdsSize, 0), 0);
}

TEST_F(BlockTableTest, ShouldReturnTwoWhenBlockIsNotFullGiventokenIdsSizeGreaterThanEmptySlots)
{
    blockTable_->Allocate(tokenIds_, DeviceType::NPU, INVALID_HASH_VALUE); // EmptySlots = 2
    size_t tokenIdsSize = 3;
    EXPECT_EQ(blockTable_->GetNumRelatedBlocks(tokenIdsSize, 0), 1);
}

} // namespace mindie_llm