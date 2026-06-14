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
#include "hashless_block_obj.h"

namespace mindie_llm {

class HashLessBlockTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        prevBlock = nullptr;
        tokenIds = {1, 2, 3, 4}; // token ids, set to {1, 2, 3, 4} for simple unitest
        blockSize = 8;           // size of the block, set to 8 for simple unittest
        allocator = nullptr;
        blockId = 0;
    }
    BlockObjSPtr prevBlock;
    std::vector<TokenId> tokenIds;
    size_t blockSize;
    BlockAllocatorSPtr allocator;
    BlockId blockId;
};

TEST_F(HashLessBlockTest, ConstructorTest)
{
    HashLessBlockObj blockObj;

    EXPECT_EQ(blockObj.GetPrevBlock(), nullptr);
    EXPECT_EQ(blockObj.GetTokenIds(), std::vector<TokenId>{});
    EXPECT_EQ(blockObj.GetBlockSize(), 0);
    EXPECT_EQ(blockObj.GetBlockId(), INVALID_BLOCKID);
}

TEST_F(HashLessBlockTest, InitBlockTest)
{
    HashLessBlockObj blockObj;
    BlockSharedAttr blockSharedAttr = {blockSize, allocator};
    blockObj.InitBlockObj(prevBlock, tokenIds, blockSharedAttr, blockId, INVALID_HASH_VALUE);

    BlockObjSPtr newPrevBlock = nullptr;
    std::vector<TokenId> newTokenIds = {9, 10, 11}; // init the block with new token ids {9, 10, 11}
    size_t newBlockSize = 16;                       // init the block with new block size 16
    BlockAllocatorSPtr newAllocator = nullptr;      // init the block with new allocator nullptr
    size_t newBlockId = 1;                          // init the block with new block id 1
    BlockSharedAttr newBlockSharedAttr = {newBlockSize, newAllocator};
    blockObj.InitBlockObj(newPrevBlock, newTokenIds, newBlockSharedAttr, newBlockId, INVALID_HASH_VALUE);

    EXPECT_EQ(blockObj.GetPrevBlock(), newPrevBlock);
    EXPECT_EQ(blockObj.GetTokenIds(), newTokenIds);
    EXPECT_EQ(blockObj.GetBlockSize(), newBlockSize);
    EXPECT_EQ(blockObj.GetBlockId(), newBlockId);
}

TEST_F(HashLessBlockTest, AppendTokenIdsTest)
{
    HashLessBlockObj blockObj;
    BlockSharedAttr blockSharedAttr = {blockSize, allocator};
    blockObj.InitBlockObj(prevBlock, tokenIds, blockSharedAttr, blockId, INVALID_HASH_VALUE);

    EXPECT_EQ(blockObj.GetTokenIds(), tokenIds);

    std::vector<TokenId> tokensIdsToAppend = {5, 6, 7, 8}; // append new token ids {5, 6, 7, 8} to the block
    blockObj.AppendTokenIdsNoCow(tokensIdsToAppend);

    std::vector<TokenId> tokenIds = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(blockObj.GetTokenIds(),
              tokenIds); // token ids in the block should be {1, 2, 3, 4, 5, 6, 7, 8} after appending
}

TEST_F(HashLessBlockTest, AppendTokenIdsNoCowTest)
{
    HashLessBlockObj blockObj;
    BlockSharedAttr blockSharedAttr = {blockSize, allocator};
    blockObj.InitBlockObj(prevBlock, tokenIds, blockSharedAttr, blockId, INVALID_HASH_VALUE);

    EXPECT_EQ(blockObj.GetTokenIds(), tokenIds);

    std::vector<TokenId> tokensIdsToAppend = {5, 6, 7, 8}; // append new token ids {5, 6, 7, 8} to the block
    blockObj.AppendTokenIdsNoCow(tokensIdsToAppend);

    std::vector<TokenId> tokenIds = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(blockObj.GetTokenIds(),
              tokenIds); // token ids in the block should be {1, 2, 3, 4, 5, 6, 7, 8} after appending
}

TEST_F(HashLessBlockTest, GetNumEmptySlotsTest)
{
    HashLessBlockObj blockObj;
    BlockSharedAttr blockSharedAttr = {blockSize, allocator};
    blockObj.InitBlockObj(prevBlock, tokenIds, blockSharedAttr, blockId, INVALID_HASH_VALUE);

    size_t numEmptySlots = blockSize - tokenIds.size(); // number of empty slots should be 8 - 4 = 4 after init
    EXPECT_EQ(blockObj.GetNumEmptySlots(), numEmptySlots);
}

TEST_F(HashLessBlockTest, GetNumTokensTotalTest)
{
    HashLessBlockObj blockObj;
    BlockSharedAttr blockSharedAttr = {blockSize, allocator};
    blockObj.InitBlockObj(prevBlock, tokenIds, blockSharedAttr, blockId, INVALID_HASH_VALUE);

    size_t numTokensTotal = 4; // number of tokens should be 0 + 4 = 4 after init
    EXPECT_EQ(blockObj.GetNumTokensTotal(), numTokensTotal);
}

TEST_F(HashLessBlockTest, IsFullTest)
{
    HashLessBlockObj blockObj;
    BlockSharedAttr blockSharedAttr = {blockSize, allocator};
    blockObj.InitBlockObj(prevBlock, tokenIds, blockSharedAttr, blockId, INVALID_HASH_VALUE);
    EXPECT_EQ(blockObj.IsFull(), false); // block should not be full after init with token ids {1, 2, 3, 4}

    std::vector<TokenId> tokensIdsToAppend = {5, 6, 7, 8}; // append new token ids {5, 6, 7, 8} to the block
    blockObj.AppendTokenIdsNoCow(tokensIdsToAppend);
    EXPECT_EQ(blockObj.IsFull(), true); // block should be full after appending {5, 6, 7, 8}
}
} // namespace mindie_llm