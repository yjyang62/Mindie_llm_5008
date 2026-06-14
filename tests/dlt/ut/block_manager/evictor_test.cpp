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

#include <climits>
#include "gtest/gtest.h"
#include "evictor.h"

using namespace mindie_llm;
using namespace std;

class EvictorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        evictor_ = MakeEvictor(EvictionPolicy::LRU);
    }

    EvictorPtr evictor_;
};

TEST_F(EvictorTest, ShouldContainsWhenAdd)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;

    evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);

    EXPECT_TRUE(evictor_->ContainsBlock(id));
}

TEST_F(EvictorTest, ShouldContainsWhenUpdate)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;

    evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);

    TimeStamp updateAccessedTime = 200.0f;
    evictor_->Update(id, updateAccessedTime);

    EXPECT_TRUE(evictor_->ContainsBlock(id));
}

TEST_F(EvictorTest, ShouldNotContainsWhenRemove)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;

    evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);

    evictor_->Remove(id);

    EXPECT_FALSE(evictor_->ContainsBlock(id));
}

TEST_F(EvictorTest, ShouldGetRightNumBlocksWhenAdd)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;

    evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);
    evictor_->Add(id, prefixHash, numHashedTokens + 1, lastAccessedTime);

    EXPECT_TRUE(evictor_->GetNumblocks() == 1);
}

TEST_F(EvictorTest, ShouldCleanWhenAddCleanupIfNecessary)
{
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;
    BlockId maxSize = 1000;

    for (BlockId id = 1; id <= maxSize; ++id) {
        evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);
    }

    EXPECT_EQ(maxSize, evictor_->GetNumblocks());
}

TEST_F(EvictorTest, ShouldGetRightNumBlocksWhenAddAndRemove)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;

    evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);
    EXPECT_TRUE(evictor_->GetNumblocks() == 1);

    evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);
    EXPECT_TRUE(evictor_->GetNumblocks() == 1);

    evictor_->Remove(id);
    EXPECT_TRUE(evictor_->GetNumblocks() == 0);
}

TEST_F(EvictorTest, ShouldThrowWhenEvictEmpty) { EXPECT_ANY_THROW(evictor_->Evict()); }

TEST_F(EvictorTest, ShouldGetResultWhenEvict)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;

    evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);

    EvictionResult result = evictor_->Evict();

    EXPECT_EQ(prefixHash, result.prefixHash);
    EXPECT_EQ(id, result.blockId);
}

// 最久未被访问的优先被驱逐
TEST_F(EvictorTest, ShouldEvictLRUWhenEvict)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    // 设置初始访问时间
    TimeStamp lastAccessedTime = 100.0f;
    int loopMax = 10;

    for (int loop = 0; loop < loopMax; ++loop) {
        evictor_->Add(id + loop, prefixHash, numHashedTokens, lastAccessedTime + loop);
    }

    EvictionResult result = evictor_->Evict();
    EXPECT_EQ(prefixHash, result.prefixHash);
    EXPECT_EQ(id, result.blockId);
}

// 最久未被访问的存在多个的时候，持有最多令牌数的优先被驱逐
TEST_F(EvictorTest, ShouldEvictLRUWhenThereAreMultiple)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;
    int loopMax = 10;

    for (int loop = 0; loop < loopMax; ++loop) {
        evictor_->Add(id + loop, prefixHash, numHashedTokens, lastAccessedTime + loop);
    }

    evictor_->Add(id + 1, prefixHash, numHashedTokens + 1, lastAccessedTime);

    EvictionResult result = evictor_->Evict();
    EXPECT_TRUE(result.prefixHash == 0x12345678);
    EXPECT_EQ(id + 1, result.blockId);
}

// 最久未被访问的存在多个的时候，存在更新场景下，最久未被访问的优先被驱逐
TEST_F(EvictorTest, ShouldEvictLRUWhenUpdate)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;
    int loopMax = 10;

    for (int loop = 0; loop < loopMax; ++loop) {
        evictor_->Add(id + loop, prefixHash, numHashedTokens, lastAccessedTime + loop);
    }

    TimeStamp now1 = 1;
    TimeStamp now2 = 2;
    evictor_->Update(id, lastAccessedTime + now1);
    evictor_->Update(id + 1, lastAccessedTime + now2);

    EvictionResult result = evictor_->Evict();

    BlockId targetBlockId = id + 2; // id和id + 1均被更新时间，LRU应该在id + 2
    EXPECT_EQ(targetBlockId, result.blockId);
}

// 最久未被访问的存在多个的时候，存在删除和更新场景下，最久未被访问的优先被驱逐，触发重建优先队列
TEST_F(EvictorTest, ShouldEvictLRUWhenUpdateAndRemove)
{
    BlockId id = 100;
    HashValue prefixHash = 0x12345678;
    size_t numHashedTokens = 10;
    TimeStamp lastAccessedTime = 100.0f;
    const int threshold = 100;

    evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);
    evictor_->Add(id + 1, prefixHash, numHashedTokens, lastAccessedTime + 1);

    for (size_t i = 0; i < evictor_->GetNumblocks() * threshold; ++i) {
        evictor_->Add(id, prefixHash, numHashedTokens, lastAccessedTime);
    }
    evictor_->Remove(id);
    evictor_->Add(id + 2, prefixHash, numHashedTokens, // id+2 更新第2个值和时间,
                  lastAccessedTime + 1);               // lastAccessedTime + 1时间触发变化

    EvictionResult result = evictor_->Evict();
    EXPECT_EQ(id + 1, result.blockId);
    EXPECT_TRUE(evictor_->GetNumblocks() == 1);
}
