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
#define private public
#include "cpu_npu_block_allocator.h"
#include "block_allocator.h"
#include "block_obj.h"
#undef private

namespace mindie_llm {

class CpuNpuBlockAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        cpuBlocksNum_ = 10;
        npuBlocksNum_ = 10;
        blockSize_ = 4;
        AllocatorConfig allocatorConfig = {BlockAllocatorType::HASHLESS, cpuBlocksNum_, npuBlocksNum_, blockSize_};
        allocator_ = std::make_shared<CpuNpuBlockAllocator>(allocatorConfig);
    }

    size_t cpuBlocksNum_;
    size_t npuBlocksNum_;
    size_t blockSize_;
    std::shared_ptr<CpuNpuBlockAllocator> allocator_;
};

// Unit Test for function AllocateMutableBlock
TEST_F(CpuNpuBlockAllocatorTest, AllocateMutableBlockTest)
{
    std::vector<TokenId> tokenIds = {1, 2, 3, 4};
    BlockObjSPtr prevBlock = nullptr;
    HashValue extraHash = 0;

    BlockObjSPtr cpuBlock = allocator_->AllocateMutableBlock(DeviceType::CPU, tokenIds, prevBlock, extraHash);
    EXPECT_NE(cpuBlock, nullptr);
    EXPECT_EQ(allocator_->GetDeviceTypeForBlockId(cpuBlock->GetBlockId()), DeviceType::CPU);
    EXPECT_EQ(cpuBlock->GetTokenIds(), tokenIds);

    BlockObjSPtr npuBlock = allocator_->AllocateMutableBlock(DeviceType::NPU, tokenIds, cpuBlock, extraHash);
    EXPECT_NE(npuBlock, nullptr);
    EXPECT_EQ(allocator_->GetDeviceTypeForBlockId(npuBlock->GetBlockId()), DeviceType::NPU);
    EXPECT_EQ(npuBlock->GetTokenIds(), tokenIds);
    EXPECT_EQ(npuBlock->GetPrevBlock(), cpuBlock);
}

// Unit Test for function Free
TEST_F(CpuNpuBlockAllocatorTest, FreeTest)
{
    std::vector<TokenId> tokenIds = {1, 2, 3, 4};
    BlockObjSPtr prevBlock = nullptr;
    HashValue extraHash = 0;

    BlockObjSPtr cpuBlock = allocator_->AllocateMutableBlock(DeviceType::CPU, tokenIds, prevBlock, extraHash);
    EXPECT_NE(cpuBlock, nullptr);

    allocator_->Free(cpuBlock);
    EXPECT_EQ(cpuBlock, nullptr);
}

// Unit Test for function Swap
TEST_F(CpuNpuBlockAllocatorTest, SwapBlocksTest)
{
    std::vector<TokenId> tokenIds{1, 2, 3};
    BlockObjSPtr cpuBlock = allocator_->AllocateMutableBlock(DeviceType::CPU, tokenIds);
    BlockObjSPtr npuBlock = allocator_->AllocateMutableBlock(DeviceType::NPU, tokenIds);
    std::vector<BlockObjSPtr> cpuSwapBlocks = {cpuBlock};
    auto cpuToNpuMapping = allocator_->Swap(cpuSwapBlocks, DeviceType::CPU, DeviceType::NPU);

    // cpu初始分配1，cpu与npu块交换后，映射关系：{10，1}
    const int cpuBlockId = 10;
    EXPECT_EQ(cpuToNpuMapping[0].first, cpuBlockId);
    const int cpuSwapBlockId = 1;
    EXPECT_EQ(cpuToNpuMapping[0].second, cpuSwapBlockId);

    std::vector<BlockObjSPtr> npuSwapBlocks = {npuBlock};
    auto npuToNpuMapping = allocator_->Swap(npuSwapBlocks, DeviceType::NPU, DeviceType::CPU);

    // npu初始分配0，npu与cpu块交换后，映射关系：{0，11}
    const int npuBlockId = 0;
    EXPECT_EQ(npuToNpuMapping[0].first, npuBlockId);
    const int npuSwapBlockId = 11;
    EXPECT_EQ(npuToNpuMapping[0].second, npuSwapBlockId);
}

// Unit Test for function GetPrefixCacheHitRate
TEST_F(CpuNpuBlockAllocatorTest, PrefixCacheHitRateTest)
{
    std::vector<TokenId> tokenIds{1, 2, 3};
    BlockObjSPtr firstBlock = allocator_->AllocateMutableBlock(DeviceType::NPU, tokenIds);
    BlockObjSPtr secondBlock = allocator_->AllocateMutableBlock(DeviceType::NPU, tokenIds);
    allocator_->MarkBlocksAsAccessed(0, {firstBlock->GetBlockId(), secondBlock->GetBlockId()}, 100.0f);

    // GetPrefixCacheHitRate方法暂未实现，后续实现修改用例
    float hiteRate = allocator_->GetPrefixCacheHitRate();
    EXPECT_EQ(hiteRate, -1);
}

// Unit Test for function AllocateImmutableBlock
TEST_F(CpuNpuBlockAllocatorTest, PhysicalBlockIdTest)
{
    std::vector<TokenId> tokenIds{1, 2, 3};
    BlockObjSPtr cpuBlock = allocator_->AllocateMutableBlock(DeviceType::CPU, tokenIds);
    PhysicalBlockId firstPhysicalBlockId = allocator_->GetPhysicalBlockId(cpuBlock->GetBlockId());
    const size_t beginCpuBlockId = 10;
    // DeviceType为CPU时，物理块减10
    EXPECT_EQ(firstPhysicalBlockId, cpuBlock->GetBlockId() - beginCpuBlockId);

    BlockObjSPtr npuBlock = allocator_->AllocateMutableBlock(DeviceType::NPU, tokenIds);
    PhysicalBlockId secondPhysicalBlockId = allocator_->GetPhysicalBlockId(cpuBlock->GetBlockId());
    EXPECT_EQ(secondPhysicalBlockId, npuBlock->GetBlockId());
}
} // namespace mindie_llm