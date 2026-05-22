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
#include "block_tracker.h"
#undef private

// Unit Test for function Enable and Disable
namespace mindie_llm {

class SeqsBlocksComputedTrackerTest : public ::testing::Test {
private:
    size_t blockSize_;

public:
    void SetUp() override
    {
        blockSize_ = 2;
        tracker_ = std::make_shared<SeqsBlocksComputedTracker>(nullptr, blockSize_, true, 16);
    }
    void TearDown() override { tracker_.reset(); }
    std::shared_ptr<SeqsBlocksComputedTracker> tracker_;
};

class SeqsLastAccessBlocksTrackerTest : public ::testing::Test {
public:
    void SetUp() override { seqsLasttracker_ = std::make_shared<SeqsLastAccessBlocksTracker>(nullptr); }

    std::shared_ptr<SeqsLastAccessBlocksTracker> seqsLasttracker_;
};

// Unit Test for function AddSeq and RemoveSeq
TEST_F(SeqsLastAccessBlocksTrackerTest, AddSeqTest)
{
    seqsLasttracker_->AddSeq(0);
    EXPECT_EQ(seqsLasttracker_->seqIdToLastAccessTime_[0], -1);
    EXPECT_THROW(seqsLasttracker_->AddSeq(0), std::runtime_error);
    seqsLasttracker_->RemoveSeq(0);
    EXPECT_THROW(seqsLasttracker_->RemoveSeq(0), std::runtime_error);
}

// Unit Test for function UpdateSeqLastAccess and UpdateSeqBlocksLastAccess
TEST_F(SeqsLastAccessBlocksTrackerTest, UpdateSeqLastAccessTest)
{
    const TimeStamp timeStamp = 100.0f;
    seqsLasttracker_->AddSeq(1);
    seqsLasttracker_->UpdateSeqLastAccess(1, timeStamp);
    EXPECT_EQ(seqsLasttracker_->seqIdToLastAccessTime_[1], timeStamp);
    seqsLasttracker_->RemoveSeq(1);
    EXPECT_EQ(seqsLasttracker_->seqIdToLastAccessTime_.size(), 0);
    EXPECT_THROW(seqsLasttracker_->UpdateSeqLastAccess(1, timeStamp), std::runtime_error);
}
} // namespace mindie_llm
