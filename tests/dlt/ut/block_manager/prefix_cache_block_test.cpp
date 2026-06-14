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
#include "prefix_cache_block.h"
#include "prefix_cache_block_allocator.h"

namespace mindie_llm {
class PrefixCacheBlockObjTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        prevBlock_ = nullptr;
        BlockId beginBlockId = 0;
        size_t numBlocks = 10;
        size_t blockSize = 4;
        uint32_t extraFactor = 4;
        BlockObjPoolSPtr blockObjPool = std::make_shared<ObjPool<BlockObj>>(
            extraFactor * numBlocks, []() { return std::make_shared<PrefixCachingBlockObj>(); });
        allocator_ = std::make_shared<PrefixCacheBlockAllocator>(beginBlockId, numBlocks, blockSize, blockObjPool);
        blockSharedAttr_.allocator = allocator_;
        objSPtr_->InitBlockObj(prevBlock_, tokenIds_, blockSharedAttr_, blockId_, INVALID_HASH_VALUE);
    }
    PrefixCachingBlockObjSPtr prevBlock_{nullptr};
    BlockId blockId_{0};
    size_t blockSize_{4};
    std::vector<TokenId> tokenIds_{1, 2};
    BlockAllocatorSPtr allocator_{nullptr};
    BlockSharedAttr blockSharedAttr_{blockSize_, allocator_};
    BlockObjSPtr objSPtr_ = std::make_shared<PrefixCachingBlockObj>();
};

TEST_F(PrefixCacheBlockObjTest, ShouldUpdateNumTotalTokensWhenInit)
{
    // Prevblock is null
    EXPECT_EQ(objSPtr_->GetNumTokensTotal(), tokenIds_.size());

    // Prevblock is not null
    std::vector<TokenId> newTokenIds{3, 4};
    BlockId newBlockId{1};
    PrefixCachingBlockObjSPtr newObjSPtr = std::make_shared<PrefixCachingBlockObj>();
    newObjSPtr->InitBlockObj(objSPtr_, newTokenIds, blockSharedAttr_, newBlockId, INVALID_HASH_VALUE);
    EXPECT_EQ(newObjSPtr->GetNumTokensTotal(), tokenIds_.size() + newTokenIds.size());
}

TEST_F(PrefixCacheBlockObjTest, ShouldUpdateNumTotalTokensWhenAppend)
{
    std::vector<TokenId> appendTokenIds{3};
    objSPtr_->AppendTokenIds(appendTokenIds);
    EXPECT_EQ(objSPtr_->GetNumTokensTotal(), tokenIds_.size() + appendTokenIds.size());
}

TEST_F(PrefixCacheBlockObjTest, ShouldThrowWhenBlockSlotsAreNotEnough)
{
    std::vector<TokenId> appendTokenIds{3, 4, 5};
    EXPECT_THROW(objSPtr_->AppendTokenIds(appendTokenIds), std::runtime_error);
}

TEST_F(PrefixCacheBlockObjTest, ShouldThrowWhenPrevBlockIsUnhashable)
{
    // Assume that the previous block is not full, i.e., unhashable
    std::vector<TokenId> newTokenIds{4, 5};
    BlockId newBlockId{1};
    PrefixCachingBlockObjSPtr newObjSPtr = std::make_shared<PrefixCachingBlockObj>();
    newObjSPtr->InitBlockObj(objSPtr_, newTokenIds, blockSharedAttr_, newBlockId, INVALID_HASH_VALUE);
    std::vector<TokenId> appendTokenIds{6, 7};
    newObjSPtr->AppendTokenIds(appendTokenIds);
    EXPECT_EQ(newObjSPtr->PrefixHash(), INVALID_HASH_VALUE);
}

TEST_F(PrefixCacheBlockObjTest, ShouldUnhashableWhenBlockIsNotFull)
{
    std::vector<TokenId> appendTokenIds{3};
    objSPtr_->AppendTokenIds(appendTokenIds);
    EXPECT_EQ(objSPtr_->PrefixHash(), INVALID_HASH_VALUE);
}

TEST_F(PrefixCacheBlockObjTest, ShouldHashableWhenBlockIsFull)
{
    std::vector<TokenId> appendTokenIds{3, 4};
    objSPtr_->AppendTokenIds(appendTokenIds);
    EXPECT_NE(objSPtr_->PrefixHash(), INVALID_HASH_VALUE);
}

TEST_F(PrefixCacheBlockObjTest, ShouldHashValueEqualWhenTokenIdsAreSame)
{
    std::vector<TokenId> tokenIds{1, 2, 3, 4};
    objSPtr_->InitBlockObj(prevBlock_, tokenIds, blockSharedAttr_, blockId_, INVALID_HASH_VALUE);
    BlockId newBlockId{1};
    PrefixCachingBlockObjSPtr newObjSPtr = std::make_shared<PrefixCachingBlockObj>();
    newObjSPtr->InitBlockObj(prevBlock_, tokenIds, blockSharedAttr_, newBlockId, INVALID_HASH_VALUE);
    EXPECT_EQ(objSPtr_->PrefixHash(), newObjSPtr->PrefixHash());
}

TEST_F(PrefixCacheBlockObjTest, ShouldHashValueNotEqualWhenTokenIdsAreDifferent)
{
    std::vector<TokenId> tokenIds{1, 2, 3, 4};
    objSPtr_->InitBlockObj(prevBlock_, tokenIds, blockSharedAttr_, blockId_, INVALID_HASH_VALUE);
    std::vector<TokenId> newTokenIds{1, 2, 3, 5};
    BlockId newBlockId{1};
    PrefixCachingBlockObjSPtr newObjSPtr = std::make_shared<PrefixCachingBlockObj>();
    newObjSPtr->InitBlockObj(prevBlock_, newTokenIds, blockSharedAttr_, newBlockId, INVALID_HASH_VALUE);
    EXPECT_NE(objSPtr_->PrefixHash(), newObjSPtr->PrefixHash());
}

} // namespace mindie_llm