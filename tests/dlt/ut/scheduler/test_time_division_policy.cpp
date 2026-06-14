/*
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
#include <mockcpp/mockcpp.hpp>
#define private public
#include "time_division_policy.h"

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

using namespace mindie_llm;

SequenceGroupSPtr createDummySeqGroup(std::string &requestId, int promptLength, int blockSize)
{
    std::vector<TokenId> dummyInputs;
    for (int i = 0; i < promptLength; ++i) {
        dummyInputs.push_back(i);
    }

    SequenceSPtr seq = std::make_shared<Sequence>(std::stol(requestId), blockSize, dummyInputs);
    auto samplingParams = std::make_shared<SamplingParams>();
    samplingParams->bestOf = 1;

    std::vector<SequenceSPtr> seqs{seq};
    SequenceGroupSPtr seqGroup = std::make_shared<SequenceGroup>(requestId, seqs, samplingParams);
    seqGroup->seqId2ParallelSeqGroup_.Insert(seqGroup->firstSeq->seqId_, seqGroup);
    seqGroup->parentSeqId_ = seqGroup->firstSeq->seqId_;
    return seqGroup;
}

class TimeDivisionPolicyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        timeDivisionPolicy_ = std::make_shared<TimeDivisionPolicy>();
    }
    void TearDown() override
    {
        // Reset the global mock object to clear any registered mocks.
        MOCKCPP_NS::GlobalMockObject::reset();
    }


    std::shared_ptr<TimeDivisionPolicy> timeDivisionPolicy_;
};

TEST_F(TimeDivisionPolicyTest, TestApply)
{
    ConcurrentDeque<SequenceGroupSPtr> waiting_;
    ConcurrentDeque<SequenceGroupSPtr> running_;
    ConcurrentDeque<SequenceGroupSPtr> swapped_;

    auto type = timeDivisionPolicy_->Apply(waiting_, running_, swapped_);
    EXPECT_EQ(type, PDPriorityType::PREFILL_FIRST);
}

TEST_F(TimeDivisionPolicyTest, TestMarkInferenceStartTimeStamp)
{
    PDPriorityType type = PDPriorityType::PREFILL_FIRST;
    timeDivisionPolicy_->MarkInferenceStartTimeStamp(type);
    EXPECT_GT(timeDivisionPolicy_->inferenceStartTimeRecords_.size(), 0);
}

TEST_F(TimeDivisionPolicyTest, TestMarkInferenceEndTimeStampEmpty)
{
    timeDivisionPolicy_->MarkInferenceEndTimeStamp();
}

TEST_F(TimeDivisionPolicyTest, TestMarkInferenceEndTimeStampPrefill)
{
    PDPriorityType type = PDPriorityType::PREFILL_FIRST;
    timeDivisionPolicy_->MarkInferenceStartTimeStamp(type);
    timeDivisionPolicy_->MarkInferenceEndTimeStamp();
    EXPECT_GT(timeDivisionPolicy_->inferenceTimeWindow_.size(), 0);
}

TEST_F(TimeDivisionPolicyTest, TestMarkInferenceEndTimeStampDecode)
{
    PDPriorityType type = PDPriorityType::DECODE_FIRST;
    timeDivisionPolicy_->MarkInferenceStartTimeStamp(type);
    timeDivisionPolicy_->MarkInferenceEndTimeStamp();
    EXPECT_GT(timeDivisionPolicy_->inferenceTimeWindow_.size(), 0);
}

TEST_F(TimeDivisionPolicyTest, TestSetPrefillPercentage)
{
    uint32_t prefillPercentage = 50;
    timeDivisionPolicy_->SetPrefillPercentage(prefillPercentage);
    EXPECT_EQ(timeDivisionPolicy_->prefillPercentage_, prefillPercentage);
}

TEST_F(TimeDivisionPolicyTest, TestGetFlexRole)
{
    ConcurrentDeque<SequenceGroupSPtr> waiting_;
    ConcurrentDeque<SequenceGroupSPtr> running_;
    ConcurrentDeque<SequenceGroupSPtr> swapped_;
    Role role = timeDivisionPolicy_->GetFlexRole(waiting_, running_, swapped_);
    EXPECT_EQ(role, Role::FlexP);
}

TEST_F(TimeDivisionPolicyTest, TestGetFlexRoleDecode)
{
    ConcurrentDeque<SequenceGroupSPtr> waiting_;
    ConcurrentDeque<SequenceGroupSPtr> running_;
    ConcurrentDeque<SequenceGroupSPtr> swapped_;

    MOCKER_CPP(&TimeDivisionPolicy::Apply, PDPriorityType (*)(ConcurrentDeque<SequenceGroupSPtr>,
        ConcurrentDeque<SequenceGroupSPtr>, ConcurrentDeque<SequenceGroupSPtr>))
        .stubs().will(returnValue(PDPriorityType::DECODE_FIRST));
    Role role = timeDivisionPolicy_->GetFlexRole(waiting_, running_, swapped_);
    EXPECT_EQ(role, Role::FlexD);
}

TEST_F(TimeDivisionPolicyTest, TestGetFlexRolePnD)
{
    ConcurrentDeque<SequenceGroupSPtr> waiting_;
    ConcurrentDeque<SequenceGroupSPtr> running_;
    ConcurrentDeque<SequenceGroupSPtr> swapped_;

    MOCKER_CPP(&TimeDivisionPolicy::Apply, PDPriorityType (*)(ConcurrentDeque<SequenceGroupSPtr>,
        ConcurrentDeque<SequenceGroupSPtr>, ConcurrentDeque<SequenceGroupSPtr>))
        .stubs().will(returnValue(PDPriorityType::MIX));

    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);
    waiting_.PushBack(seqGroup);
    Role role = timeDivisionPolicy_->GetFlexRole(waiting_, running_, swapped_);
    EXPECT_EQ(role, Role::FlexPnD);
}

TEST_F(TimeDivisionPolicyTest, TestUpdateSlideWindow)
{
    timeDivisionPolicy_->prefillInferenceTime_ = 2000000;
    timeDivisionPolicy_->decodeInferenceTime_ = 2000000;
    auto type = PDPriorityType::PREFILL_FIRST;
    uint64_t inferenceTime = 2000000;
    timeDivisionPolicy_->inferenceTimeWindow_.push(std::make_pair(type, inferenceTime));
    timeDivisionPolicy_->UpdateSlideWindow();
    EXPECT_EQ(timeDivisionPolicy_->inferenceTimeWindow_.size(), 0);
}

TEST_F(TimeDivisionPolicyTest, TestGetNowTimeStamp)
{
    uint64_t now = timeDivisionPolicy_->GetNowTimeStamp();
}

TEST_F(TimeDivisionPolicyTest, TestCheckRequestExistence)
{
    ConcurrentDeque<SequenceGroupSPtr> queue_;
    bool flag = timeDivisionPolicy_->CheckRequestExistence(queue_, [](const auto& sg) {return sg->isDecode_;});
    EXPECT_FALSE(flag);
    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);
    seqGroup->isDecode_ = true;
    queue_.PushBack(seqGroup);
    flag = timeDivisionPolicy_->CheckRequestExistence(queue_, [](const auto& sg) {return sg->isDecode_;});
    EXPECT_TRUE(flag);
}

TEST_F(TimeDivisionPolicyTest, TestHasPrefillReq)
{
    ConcurrentDeque<SequenceGroupSPtr> queue_;
    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);
    queue_.PushBack(seqGroup);
    bool flag = timeDivisionPolicy_->HasPrefillReq(queue_);
    EXPECT_TRUE(flag);
}

TEST_F(TimeDivisionPolicyTest, TestHasDecodeReq)
{
    ConcurrentDeque<SequenceGroupSPtr> waiting_;
    ConcurrentDeque<SequenceGroupSPtr> running_;
    ConcurrentDeque<SequenceGroupSPtr> swapped_;

    bool flag = timeDivisionPolicy_->HasDecodeReq(waiting_, running_, swapped_);
    EXPECT_FALSE(flag);
}

TEST_F(TimeDivisionPolicyTest, TestHasFlexLocalReq)
{
    ConcurrentDeque<SequenceGroupSPtr> waiting_;
    ConcurrentDeque<SequenceGroupSPtr> running_;
    ConcurrentDeque<SequenceGroupSPtr> swapped_;

    bool flag = timeDivisionPolicy_->HasFlexLocalReq(waiting_, running_, swapped_);
    EXPECT_FALSE(flag);
}