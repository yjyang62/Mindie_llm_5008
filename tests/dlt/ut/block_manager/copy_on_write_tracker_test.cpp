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
#include "copy_on_write_tracker.h"

using namespace mindie_llm;
using namespace std;

class CopyOnWriteTrackerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        std::vector<BlockId> allBlockIndices;
        for (BlockId blockIdx = 0; (beginBlockId_ + blockIdx) <= endBlockId_; blockIdx++) {
            allBlockIndices.push_back(blockIdx + beginBlockId_);
        }

        refCounter_ = MakeRefCounterProtocol(allBlockIndices);
    }

    void TearDown() override {}

protected:
    BlockId beginBlockId_ = 100;
    BlockId endBlockId_ = 200;
    RefCounterProtocolSPtr refCounter_;
};

TEST_F(CopyOnWriteTrackerTest, ShouldAppendableWhenRefCountLessThan1)
{
    CopyOnWriteTracker cow(refCounter_);

    EXPECT_TRUE(cow.IsAppendable(beginBlockId_));
}

TEST_F(CopyOnWriteTrackerTest, ShouldNotAppendableWhenRefCountMoreThan1)
{
    EXPECT_EQ(1, refCounter_->Increase(beginBlockId_));
    EXPECT_EQ(2, refCounter_->Increase(beginBlockId_));
    EXPECT_EQ(3, refCounter_->Increase(beginBlockId_));

    CopyOnWriteTracker cow(refCounter_);
    EXPECT_FALSE(cow.IsAppendable(beginBlockId_));
}

TEST_F(CopyOnWriteTrackerTest, ShouldNotAppendableWhenRefCountMoreThan1AndLess1)
{
    EXPECT_EQ(1, refCounter_->Increase(beginBlockId_));
    EXPECT_EQ(2, refCounter_->Increase(beginBlockId_));
    EXPECT_EQ(3, refCounter_->Increase(beginBlockId_));
    EXPECT_EQ(2, refCounter_->Decrease(beginBlockId_));
    EXPECT_EQ(1, refCounter_->Decrease(beginBlockId_));
    EXPECT_EQ(1, refCounter_->GetRefCount(beginBlockId_));

    CopyOnWriteTracker cow(refCounter_);
    EXPECT_TRUE(cow.IsAppendable(beginBlockId_));
}

// 对于不存在的BlockId 能够抛出异常
TEST_F(CopyOnWriteTrackerTest, ShouldThrowErrorWhenRefCountInvalid)
{
    EXPECT_ANY_THROW(refCounter_->Increase(beginBlockId_ - 1));
    EXPECT_ANY_THROW(refCounter_->Decrease(beginBlockId_));
    EXPECT_ANY_THROW(refCounter_->GetRefCount(beginBlockId_ - 1));
}

TEST_F(CopyOnWriteTrackerTest, ShouldRecordCowWhenCow)
{
    CopyOnWriteTracker cow(refCounter_);

    cow.RecordCow(beginBlockId_, endBlockId_);

    auto result = cow.ClearCows();
    ASSERT_TRUE(result.size() == 1);
    EXPECT_EQ(std::make_pair(beginBlockId_, endBlockId_), result[0]);
}

// 建立多个RecordCow，能正确返回
TEST_F(CopyOnWriteTrackerTest, ShouldRecordCowRightWhenCow)
{
    CopyOnWriteTracker cow(refCounter_);
    std::vector<std::pair<BlockId, BlockId>> srcCows;

    for (BlockId srcBlockId = beginBlockId_; srcBlockId <= endBlockId_; ++srcBlockId) {
        for (BlockId targetBlockId = srcBlockId + 1; targetBlockId <= endBlockId_; ++targetBlockId) {
            cow.RecordCow(srcBlockId, targetBlockId);
            srcCows.emplace_back(srcBlockId, targetBlockId);
        }
    }

    auto result = cow.ClearCows();
    ASSERT_EQ(srcCows.size(), result.size());
    EXPECT_EQ(srcCows, result);
}

// ClearCows后继续RecordCow，能正确获取到对应数据
TEST_F(CopyOnWriteTrackerTest, ShouldRecordCowRightWhenMultipleClearCows)
{
    CopyOnWriteTracker cow(refCounter_);

    cow.RecordCow(beginBlockId_, endBlockId_);
    auto result = cow.ClearCows();
    EXPECT_TRUE(result.size() == 1);

    cow.RecordCow(beginBlockId_ + 1, endBlockId_);
    result = cow.ClearCows();
    EXPECT_TRUE(result.size() == 1);
}
