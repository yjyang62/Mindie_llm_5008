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
#include "hashless_block_allocator.h"

namespace mindie_llm {

class HashLessBlockAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        beginBlockId_ = 20;
        numBlocks_ = 10;
        blockSize_ = 1024;
        blockObjPool_ =
            std::make_shared<ObjPool<BlockObj>>(numBlocks_, []() { return std::make_shared<HashLessBlockObj>(); });
        allocator_ = std::make_shared<HashLessBlockAllocator>(beginBlockId_, numBlocks_, blockSize_, blockObjPool_);
    }

    BlockId beginBlockId_;
    size_t numBlocks_;
    size_t blockSize_;
    BlockObjPoolSPtr blockObjPool_;
    std::shared_ptr<HashLessBlockAllocator> allocator_;
};

// Unit Test function AllocateBlockId
TEST_F(HashLessBlockAllocatorTest, AllocateBlockIdTest)
{
    BlockId firstId = allocator_->AllocateBlockId();
    EXPECT_EQ(firstId, beginBlockId_);
    EXPECT_EQ(allocator_->GetNumFreeBlock(), numBlocks_ - 1);
    BlockId secondId = allocator_->AllocateBlockId();
    EXPECT_EQ(secondId, beginBlockId_ + 1);
    EXPECT_EQ(allocator_->GetNumFreeBlock(), numBlocks_ - 2);
}

// Test function AllocateMutableBlock
TEST_F(HashLessBlockAllocatorTest, AllocateMutableBlock)
{
    std::vector<TokenId> tokenIds = {1, 2, 3};
    BlockObjSPtr prevBlock = nullptr;
    BlockObjSPtr block = allocator_->AllocateMutableBlock(tokenIds, prevBlock);
    EXPECT_NE(block, nullptr);
    EXPECT_EQ(block->GetTokenIds(), tokenIds);
}

// Unit Test for function Free
TEST_F(HashLessBlockAllocatorTest, FreeBlockIdTest)
{
    BlockId blockId = allocator_->AllocateBlockId();
    BlockObjSPtr block = blockObjPool_->AcquireObj();
    block->SetBlockId(blockId);
    allocator_->Free(block, false);
    EXPECT_EQ(block, nullptr);
    EXPECT_EQ(allocator_->GetNumFreeBlock(), numBlocks_);
}

// Test function AllocateImmutableBlocks
TEST_F(HashLessBlockAllocatorTest, AllocateImmutableBlocksTest)
{
    std::vector<std::vector<TokenId>> tokenIds = {{1, 2, 3}, {4, 5, 6}};
    std::vector<BlockObjSPtr> blocks = allocator_->AllocateImmutableBlocks(tokenIds, nullptr);
    EXPECT_EQ(blocks.size(), tokenIds.size());
    for (size_t i = 0; i < blocks.size(); i++) {
        EXPECT_EQ(blocks[i]->GetTokenIds(), tokenIds[i]);
    }
}

// Test function Fork
TEST_F(HashLessBlockAllocatorTest, ForkTest)
{
    std::vector<TokenId> firstTokenIds = {1, 2, 3};
    std::vector<TokenId> secondTokenIds = {4, 5, 6};
    BlockObjSPtr firstBlock = allocator_->AllocateMutableBlock(firstTokenIds, nullptr);
    BlockObjSPtr secondBlock = allocator_->AllocateMutableBlock(secondTokenIds, firstBlock);
    BlockObjSPtr lastBlock = secondBlock;
    std::vector<BlockObjSPtr> forkedBlocks = allocator_->Fork(lastBlock);
    EXPECT_EQ(forkedBlocks[0]->GetTokenIds(), firstBlock->GetTokenIds());
    EXPECT_EQ(forkedBlocks[1]->GetTokenIds(), secondBlock->GetTokenIds());
}
} // namespace mindie_llm