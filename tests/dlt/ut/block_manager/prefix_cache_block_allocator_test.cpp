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
#include "block_manager_interface.h"
#define private public
#include "cpu_npu_block_allocator.h"
#include "device_aware_block_allocator.h"
#include "prefix_cache_block_allocator.h"
#include "prefix_cache_block_allocator_test.h"
#undef private

using namespace std;

namespace mindie_llm {

struct BlockIdInfo {
    bool isActive;
    bool isCached;
    bool isInEvictor;
    bool isComputed;
    int refCnt;
    bool operator==(const BlockIdInfo &other) const
    {
        return isActive == other.isActive && isCached == other.isCached && isInEvictor == other.isInEvictor &&
               isComputed == other.isComputed && refCnt == other.refCnt;
    }
};

BlockIdInfo GetBlockInnerInfo(const BlockId globalBlockId, DeviceAwareBlockAllocatorSPtr blockAllocator)
{
    auto cpuNpuAllocator = static_cast<CpuNpuBlockAllocator *>(blockAllocator.get());
    DeviceType deviceType = cpuNpuAllocator->GetDeviceTypeForBlockId(globalBlockId);
    auto allocator = cpuNpuAllocator->GetAllocator(deviceType);
    auto allocatorRaw = static_cast<PrefixCacheBlockAllocator *>(allocator.get());

    BlockIdInfo info;
    info.refCnt = allocatorRaw->refCounter_->GetRefCount(globalBlockId);
    info.isInEvictor = allocatorRaw->evictor_->ContainsBlock(globalBlockId);
    info.isActive = allocatorRaw->blockComputedAttr_.IsActive(globalBlockId);
    info.isComputed = allocatorRaw->blockComputedAttr_.IsComputed(globalBlockId);

    info.isCached = false;
    for (const auto &pair : allocatorRaw->cachedBlocks_) {
        if (pair.second == globalBlockId) {
            info.isCached = true;
            break;
        }
    }

    return info;
}

// groupCount 申请多少个token组，每个token组包含tokenSize个token
static TestFixture SetupAllocator(const size_t groupCount = 8, const size_t tokenSize = 8, const size_t numBlocks = 16,
                                  const size_t blockSize = 8, const BlockId beginBlockId = 0)
{
    uint32_t extraFactor = 4;
    BlockObjPoolSPtr blockObjPool = std::make_shared<ObjPool<BlockObj>>(
        extraFactor * numBlocks, []() { return std::make_shared<PrefixCachingBlockObj>(); });
    auto allocator = std::make_shared<PrefixCacheBlockAllocator>(beginBlockId, numBlocks, blockSize, blockObjPool);

    std::vector<std::vector<TokenId>> tokens;
    TokenId currentId = 1;
    for (size_t i = 0; i < groupCount; ++i) {
        std::vector<TokenId> group;
        for (size_t j = 0; j < tokenSize; ++j) {
            group.push_back(currentId++);
        }
        tokens.push_back(group);
    }

    return {allocator, tokens};
}

class PrefixCacheBlockAlloctorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(PrefixCacheBlockAlloctorTest, should_return_one_blockObj_one_blockId_when_alloc_a_mutable_obj)
{
    // given
    size_t groupCount = 1;
    size_t tokenSize = 7;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    // whend
    BlockObjSPtr objPtr = allocator->AllocateMutableBlock(tokens[0], nullptr, 0);

    // then
    EXPECT_EQ(objPtr->GetNumTokensTotal(), 7);
    EXPECT_EQ(objPtr->IsFull(), false);
    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 1);
    EXPECT_EQ(allocator->IsBlockComputed(objPtr->GetBlockId()), false);
    EXPECT_EQ(allocator->IsBlockCached(objPtr), false);
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 16 - 1);
}

TEST_F(PrefixCacheBlockAlloctorTest, should_return_balance_after_alloc_and_free_a_mutable_obj)
{
    // given
    size_t numBlocks = 16;
    size_t groupCount = 1;
    size_t tokenSize = 7;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    // when
    BlockObjSPtr objPtr = allocator->AllocateMutableBlock(tokens[0], nullptr, 0);
    allocator->Free(objPtr, false);

    // then：申请再释放后，block obj 和 block id的数量恢复到原有，保持平衡
    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 0);
    EXPECT_EQ(allocator->freeBlockIndices_.size(), numBlocks);
}

TEST_F(PrefixCacheBlockAlloctorTest, should_return_one_blockObj_one_blockId_when_alloc_a_imutable_obj)
{
    size_t numBlocks = 16;
    size_t groupCount = 1;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    BlockObjSPtr objPtr = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);

    EXPECT_EQ(objPtr->GetNumTokensTotal(), tokenSize);
    // imutable_obj这里必为为full
    EXPECT_EQ(objPtr->IsFull(), true);
    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 1);
    // imutable的blockId 会加到cache，但是copmputed标记不能设置
    EXPECT_EQ(allocator->IsBlockComputed(objPtr->GetBlockId()), false);
    EXPECT_EQ(allocator->IsBlockCached(objPtr), true);
    EXPECT_EQ(allocator->touchedBlocks_.size(), 1);
    EXPECT_EQ(allocator->freeBlockIndices_.size(), numBlocks - 1);
}

TEST_F(PrefixCacheBlockAlloctorTest, should_return_mark_computed_when_imutable_obj_touched)
{
    // given
    auto [allocator, tokens] = SetupAllocator(8);

    // when
    BlockObjSPtr objPtr = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);
    bool beforeTouchComputedFlag = allocator->IsBlockComputed(objPtr->GetBlockId());
    allocator->MarkBlocksAsComputed();

    // then
    EXPECT_EQ(beforeTouchComputedFlag, false);
    EXPECT_EQ(allocator->IsBlockComputed(objPtr->GetBlockId()), true);
}

TEST_F(PrefixCacheBlockAlloctorTest, should_return_2_block_when_allocate_twice)
{
    size_t groupCount = 2;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    BlockObjSPtr objPtr = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);
    BlockObjSPtr objPtr2 = allocator->AllocateImmutableBlock(tokens[1], nullptr, 0);

    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 2);
}

TEST_F(PrefixCacheBlockAlloctorTest, should_return_8_block_when_allocate_group_size8)
{
    size_t groupCount = 8;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);

    EXPECT_EQ(objPtrs.size(), 8);
    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 8);
}

TEST_F(PrefixCacheBlockAlloctorTest, should_return_reuse_1_use_9_block_when_reuse_no_touch)
{
    size_t groupCount = 8;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    // 用了8 +2 个block，8+1个blockid，剩余16 - 9 = 7个blockid
    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 8);

    BlockObjSPtr objPtr = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 8);
    BlockObjSPtr objPtr2 = allocator->AllocateImmutableBlock(tokens[1], nullptr, 0);
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 7);

    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 10);
}

TEST_F(PrefixCacheBlockAlloctorTest, should_return_use_8_reuse_8_block_when_reuse_no_touch)
{
    size_t groupCount = 8;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    // 用了16个block，8个blockid，剩余16 - 8 = 8个blockid
    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 8);

    vector<BlockObjSPtr> objPtrs2 = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);

    EXPECT_EQ(objPtrs.size(), 8);
    EXPECT_EQ(objPtrs2.size(), 8);
    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 16);
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 8);
}

class PrefixCacheBlockAlloctorSwapTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(PrefixCacheBlockAlloctorSwapTest, should_return_swap_out_swap_in_mmutable_blocks)
{
    size_t groupCount = 1;
    size_t tokenSize = 7;
    size_t numBlocks = 16;
    size_t cpuBeginBlockId = 16;
    auto [allocatorNpu, tokensNpu] = SetupAllocator(groupCount, tokenSize, numBlocks);
    auto [allocatorCpu, tokensCpu] = SetupAllocator(groupCount, tokenSize, numBlocks, 8, cpuBeginBlockId);

    BlockObjSPtr objPtr = allocatorNpu->AllocateMutableBlock(tokensNpu[0], nullptr, 0);
    vector<BlockObjSPtr> objPtrs = {objPtr};
    allocatorNpu->SwapOut(objPtrs);
    allocatorCpu->SwapIn(objPtrs);
    allocatorCpu->SwapOut(objPtrs);
    allocatorNpu->SwapIn(objPtrs);

    EXPECT_EQ(objPtrs.size(), 1);
    EXPECT_EQ(allocatorNpu->blockObjPool_->GetPoolSize() - allocatorNpu->blockObjPool_->GetFreeObjNum(), 1);
    EXPECT_EQ(allocatorNpu->freeBlockIndices_.size(), numBlocks - 1);
    EXPECT_EQ(allocatorCpu->blockObjPool_->GetPoolSize() - allocatorCpu->blockObjPool_->GetFreeObjNum(), 0);
    EXPECT_EQ(allocatorCpu->freeBlockIndices_.size(), numBlocks);
}

TEST_F(PrefixCacheBlockAlloctorSwapTest, should_return_swap_out_swap_in_immutable_blocks)
{
    size_t groupCount = 8;
    size_t tokenSize = 8;
    size_t numBlocks = 16;
    auto [allocatorNpu, tokensNpu] = SetupAllocator(groupCount, tokenSize, numBlocks);
    auto [allocatorCpu, tokensCpu] = SetupAllocator(groupCount, tokenSize, numBlocks);

    vector<BlockObjSPtr> objPtrs = allocatorNpu->AllocateImmutableBlocks(tokensNpu, nullptr, 0);
    allocatorNpu->SwapOut(objPtrs);
    allocatorCpu->SwapIn(objPtrs);
    allocatorCpu->SwapOut(objPtrs);
    allocatorNpu->SwapIn(objPtrs);

    EXPECT_EQ(objPtrs.size(), 8);
    EXPECT_EQ(allocatorNpu->blockObjPool_->GetPoolSize() - allocatorNpu->blockObjPool_->GetFreeObjNum(), 8);
    EXPECT_EQ(allocatorNpu->freeBlockIndices_.size(), numBlocks - 8);
    EXPECT_EQ(allocatorCpu->blockObjPool_->GetPoolSize() - allocatorCpu->blockObjPool_->GetFreeObjNum(), 0);
    EXPECT_EQ(allocatorCpu->freeBlockIndices_.size() + allocatorCpu->evictor_->GetNumblocks(), numBlocks);
}

class FindCachedBlocksPrefixTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(FindCachedBlocksPrefixTest, should_return_3_cached_prefix_block)
{
    // given
    size_t groupCount = 8;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    allocator->MarkBlocksAsComputed();

    std::vector<HashValue> blockHashes;

    size_t givenBlockHasheLen = 3;
    size_t expectMatchLen = givenBlockHasheLen;

    for (size_t i = 0; i < givenBlockHasheLen; i++) {
        blockHashes.emplace_back(objPtrs[i]->PrefixHash());
    }

    // When
    auto result = allocator->FindCachedBlocksPrefix(blockHashes);

    // Then
    EXPECT_EQ(result.size(), givenBlockHasheLen);
}

TEST_F(PrefixCacheBlockAlloctorTest, should_return_computed_false_when_allocate_2_same_block)
{
    size_t groupCount = 8;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    BlockObjSPtr objPtr1 = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);
    BlockObjSPtr objPtr2 = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);

    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 2);
    EXPECT_EQ(allocator->IsBlockComputed(objPtr1->GetBlockId()), false);
    EXPECT_EQ(allocator->IsBlockCached(objPtr1), true);
    EXPECT_EQ(allocator->IsBlockComputed(objPtr2->GetBlockId()), false);
    EXPECT_EQ(allocator->IsBlockCached(objPtr2), true);
}

TEST_F(FindCachedBlocksPrefixTest, should_return_unmatched_cached_prefix_block)
{
    // given
    size_t groupCount = 8;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);
    size_t givenBlockHasheLen = 3;
    size_t expectMatchLen = 0;
    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    // 三个不存在的hash前缀
    std::vector<HashValue> blockHashes = {123, 456, 789};
    allocator->MarkBlocksAsComputed();

    // When
    auto result = allocator->FindCachedBlocksPrefix(blockHashes);

    // Then
    EXPECT_EQ(result.size(), expectMatchLen);
}

TEST_F(FindCachedBlocksPrefixTest, should_return_8_cached_prefix_block)
{
    // given
    size_t groupCount = 8;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    allocator->MarkBlocksAsComputed();

    std::vector<HashValue> blockHashes;

    size_t givenBlockHasheLen = 8;
    size_t expectMatchLen = givenBlockHasheLen;

    for (size_t i = 0; i < givenBlockHasheLen; i++) {
        blockHashes.emplace_back(objPtrs[i]->PrefixHash());
    }

    // 0-7的前缀hash一致，第8个不一致
    blockHashes.emplace_back(123456);

    // When
    auto result = allocator->FindCachedBlocksPrefix(blockHashes);

    // Then
    EXPECT_EQ(result.size(), givenBlockHasheLen);
}

TEST_F(FindCachedBlocksPrefixTest, should_return_4_cached_prefix_block)
{
    // given
    size_t groupCount = 8;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);

    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    allocator->MarkBlocksAsComputed();

    std::vector<HashValue> blockHashes;

    size_t givenBlockHasheLen = 4;
    size_t expectMatchLen = givenBlockHasheLen;

    // 0-3的前缀hash一致，4-7个不一致
    for (size_t i = 0; i < givenBlockHasheLen; i++) {
        blockHashes.emplace_back(objPtrs[i]->PrefixHash());
    }
    blockHashes.emplace_back(123456);
    blockHashes.emplace_back(123457);
    blockHashes.emplace_back(123458);
    blockHashes.emplace_back(123459);
    // When
    auto result = allocator->FindCachedBlocksPrefix(blockHashes);

    // Then
    EXPECT_EQ(result.size(), givenBlockHasheLen);
}

class PrefixCacheEvictorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(PrefixCacheEvictorTest, should_return_evictor_saved_when_allocate_all_and_free)
{
    size_t groupCount = 8;
    size_t tokenSize = 8;
    // 一共只有8个block
    size_t numBlocks = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize, numBlocks);

    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    // tokens中的8个token申请之后，没有剩余的block了
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 0);

    for (auto objPtr : objPtrs) {
        // 校验是否为full，只有满的才会存到cache中，释放才会放到evictor
        EXPECT_EQ(objPtr->IsFull(), true);
        allocator->Free(objPtr, false);
    }

    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 0);
    for (BlockId blockId = 0; blockId < numBlocks; blockId++) {
        EXPECT_EQ(allocator->evictor_->ContainsBlock(blockId), true);
    }
}

TEST_F(PrefixCacheEvictorTest, should_not_return_evictor_empty_when_allocate_all_and_free_unfull_block)
{
    size_t groupCount = 8;
    // block空间可以放8个，但是只存储7个，每个block不满
    size_t tokenSize = 7;
    // 一共只有8个block
    size_t numBlocks = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize, numBlocks);

    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    // tokens中的8个token申请之后，没有剩余的block了
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 0);

    for (auto objPtr : objPtrs) {
        // 校验是否为full，只有满的才会存到cache中，释放才会放到evictor
        EXPECT_EQ(objPtr->IsFull(), false);
        allocator->Free(objPtr, false);
    }

    // 由于block不满，所以应该不会在evictor中
    EXPECT_EQ(allocator->blockObjPool_->GetPoolSize() - allocator->blockObjPool_->GetFreeObjNum(), 0);
    for (BlockId blockId = 0; blockId < numBlocks; blockId++) {
        EXPECT_EQ(allocator->evictor_->ContainsBlock(blockId), false);
    }
}

class PrefixCacheForkTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(PrefixCacheForkTest, should_return_full_block_chain_when_forking)
{
    // given
    size_t groupCount = 8;
    size_t tokenSize = 8;
    size_t numBlocks = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize, numBlocks);
    auto objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);

    // when
    auto forkedBlocks = allocator->Fork(objPtrs.back());

    // then 验证链结构
    ASSERT_EQ(forkedBlocks.size(), objPtrs.size());
    for (size_t i = 1; i < objPtrs.size(); ++i) {
        EXPECT_EQ(forkedBlocks[i]->GetPrevBlock(), forkedBlocks[i - 1]);
    }

    // 验证属性复制 主要是tokenId 、 refcnt、hash值
    for (size_t i = 0; i < objPtrs.size(); ++i) {
        EXPECT_EQ(forkedBlocks[i]->GetTokenIds(), objPtrs[i]->GetTokenIds());
        EXPECT_EQ(forkedBlocks[i]->ExtraHash(), objPtrs[i]->ExtraHash());
        EXPECT_EQ(allocator->refCounter_->GetRefCount(objPtrs[i]->GetBlockId()), 2);
    }
}

TEST_F(PrefixCacheForkTest, should_return_single_block_when_forking_single_block)
{
    // given
    size_t groupCount = 1;
    size_t tokenSize = 8;
    size_t numBlocks = 1;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize, numBlocks);
    BlockObjSPtr objPtr = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);

    // when
    auto forkedBlocks = allocator->Fork(objPtr);

    // then
    ASSERT_EQ(forkedBlocks.size(), 1);
    EXPECT_EQ(forkedBlocks[0]->GetTokenIds(), objPtr->GetTokenIds());
    EXPECT_EQ(forkedBlocks[0]->ExtraHash(), objPtr->ExtraHash());
    EXPECT_EQ(allocator->refCounter_->GetRefCount(objPtr->GetBlockId()), 2);
}

TEST_F(PrefixCacheForkTest, should_throw_runtime_error_when_forking_free_block)
{
    // given
    size_t groupCount = 1;
    size_t tokenSize = 8;
    size_t numBlocks = 1;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize, numBlocks);
    BlockObjSPtr objPtr = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);
    // 模拟设置引用计数被释放
    allocator->refCounter_->Decrease(objPtr->GetBlockId());

    // when/then 不能复制计数为0的block runtime_error("can't fork free'd block!")
    EXPECT_THROW(allocator->Fork(objPtr), std::runtime_error);
}

TEST_F(PrefixCacheForkTest, should_return_empty_vector_when_forking_null_block)
{
    // given
    size_t groupCount = 1;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);
    BlockObjSPtr lastBlockObj = nullptr;

    // when
    auto forkedBlocks = allocator->Fork(lastBlockObj);

    // then
    EXPECT_TRUE(forkedBlocks.empty());
}

TEST_F(PrefixCacheForkTest, should_return_correct_chain_structure_when_forking_chain)
{
    // given
    size_t groupCount = 3;
    size_t tokenSize = 8;
    size_t numBlocks = 3;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize, numBlocks);
    auto objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);

    // when
    auto forkedBlocks = allocator->Fork(objPtrs.back());

    // then
    ASSERT_EQ(forkedBlocks.size(), 3);
    EXPECT_EQ(forkedBlocks[0]->GetPrevBlock(), nullptr);
    EXPECT_EQ(forkedBlocks[1]->GetPrevBlock(), forkedBlocks[0]);
    EXPECT_EQ(forkedBlocks[2]->GetPrevBlock(), forkedBlocks[1]);
}

TEST_F(PrefixCacheForkTest, should_increase_ref_count_when_forking_block)
{
    // given
    size_t groupCount = 1;
    size_t tokenSize = 8;
    size_t numBlocks = 1;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize, numBlocks);
    auto objPtr = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);

    // when
    allocator->Fork(objPtr);

    // then
    EXPECT_EQ(allocator->refCounter_->GetRefCount(objPtr->GetBlockId()), 2);
}

class PrefixCacheCommonBlockTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(PrefixCacheCommonBlockTest, should_return_4_wehn_handle_different_sequence_lengths)
{
    // given
    std::vector<std::vector<BlockId>> input = {{0, 1, 2, 3, 4, 5}, // 长序列
                                               {0, 1, 2, 3},       // 短序列
                                               {0, 1, 2, 3, 6}};

    size_t groupCount = 4;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);
    auto objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);

    // when
    auto result = allocator->GetCommonComputedBlockIds(input);

    // then
    std::vector<BlockId> expectCommonComputedBlockIds = {0, 1, 2, 3};
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result, expectCommonComputedBlockIds);
}

TEST_F(PrefixCacheCommonBlockTest, should_return_0_when_handle_empty_sequence)
{
    // given
    std::vector<std::vector<BlockId>> input = {};

    size_t groupCount = 4;
    size_t tokenSize = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);
    auto objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);

    // when
    auto result = allocator->GetCommonComputedBlockIds(input);

    // then
    std::vector<BlockId> expectCommonComputedBlockIds = {};
    EXPECT_EQ(result.size(), 0);
    EXPECT_EQ(result, expectCommonComputedBlockIds);
}

TEST_F(PrefixCacheCommonBlockTest, should_return_1_stop_at_first_mismatch)
{
    // given
    std::vector<std::vector<BlockId>> input = {{0, 1, 2, 3, 4, 5},
                                               {0, 123456, 2, 3}, // 第二个元素不匹配
                                               {0, 1, 2, 3, 6}};

    size_t groupCount = 4;
    size_t tokenSize = 8;

    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize);
    auto objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);

    // when
    auto result = allocator->GetCommonComputedBlockIds(input);

    // then
    std::vector<BlockId> expectCommonComputedBlockIds = {0};
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result, expectCommonComputedBlockIds);
}

class PrefixCacheEvictAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};
TEST_F(PrefixCacheEvictAllocatorTest, should_return_valid_block_when_evict_success)
{
    // given
    size_t groupCount = 8;
    size_t tokenSize = 8;
    // 申请只有8个block的allocator
    size_t numBlocks = 8;
    auto [allocator, tokens] = SetupAllocator(groupCount, tokenSize, numBlocks);

    vector<BlockObjSPtr> objPtrs = allocator->AllocateImmutableBlocks(tokens, nullptr, 0);
    // tokens中的8个token申请之后，没有剩余的block了
    EXPECT_EQ(allocator->freeBlockIndices_.size(), 0);

    // 全部释放到evictor中
    for (auto objPtr : objPtrs) {
        allocator->Free(objPtr, false);
    }

    // when  改变token，确保不会复用之前的，这时候只能从evictor中申请
    tokens[0][0] = 12121212;
    size_t evictorNumPost = allocator->evictor_->GetNumblocks();
    auto objEvictorPtrs = allocator->AllocateImmutableBlock(tokens[0], nullptr, 0);
    size_t evictorNumAfter = allocator->evictor_->GetNumblocks();

    // then evictor中少一个
    EXPECT_EQ(evictorNumPost - evictorNumAfter, 1);
    EXPECT_EQ(allocator->GetPrefixCacheHitRate(), 0);
}
} // namespace mindie_llm