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
#include "scheduling_budget.h"

namespace mindie_llm {

class SchedulingBudgetTest : public ::testing::Test {
protected:
    std::shared_ptr<SchedulingBudget> budget;

    void SetUp() override
    {
        // Init SchedulingBudget(100 tokens, 10 seqs)
        budget = std::make_shared<SchedulingBudget>(100, 10);
    }
};

// --------------- Repeated Add Calls (Ignored) --------------- //

TEST_F(SchedulingBudgetTest, AddNumBatchedTokensRepeatedlyForSameRequestId)
{
    RequestId req = "reqRepeatedTokens";
    size_t tNumBatchedTokens = 30;
    size_t tNumCachedTokens = 5;

    // First add
    budget->AddNumBatchedTokens(req, tNumBatchedTokens, tNumCachedTokens);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(30));
    EXPECT_EQ(budget->numCachedTokens_, static_cast<size_t>(5));

    // Second addition for same ID is ignored
    budget->AddNumBatchedTokens(req, 20, 10);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(30));
    EXPECT_EQ(budget->numCachedTokens_, static_cast<size_t>(5));

    // Subtract
    budget->SubtractNumBatchedTokens(req, 30);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(0));
    EXPECT_EQ(budget->requestIdsNumBatchedTokens_.count(req), static_cast<size_t>(0));
}

TEST_F(SchedulingBudgetTest, AddNumSeqsRepeatedlyForSameRequestId)
{
    RequestId req = "reqRepeatedSeqs";

    budget->AddNumSeqs(req, 2);
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(2));

    // Second addition for same ID is ignored
    budget->AddNumSeqs(req, 3);
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(2));

    // Subtract
    budget->SubtractNumSeqs(req, 2);
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(0));
    EXPECT_EQ(budget->requestIdsNumCurSeqs_.count(req), static_cast<size_t>(0));
}

// -------------------- Edge Cases with Zero -------------------- //

TEST_F(SchedulingBudgetTest, CanScheduleZeroTokensButNonZeroSeqs)
{
    RequestId req = "reqZeroTokens";
    EXPECT_TRUE(budget->CanSchedule(0, 1));

    // Schedule tokens (zero here)
    budget->AddNumBatchedTokens(req, 0, 0);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(0));
    EXPECT_EQ(budget->numCachedTokens_, static_cast<size_t>(0));
    EXPECT_EQ(budget->requestIdsNumBatchedTokens_.count(req), static_cast<size_t>(1));

    // Add seqs
    budget->AddNumSeqs(req, 1);
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(1));
}

TEST_F(SchedulingBudgetTest, CanScheduleZeroSequencesButNonZeroTokens)
{
    RequestId req = "reqZeroSeqs";
    EXPECT_TRUE(budget->CanSchedule(10, 0));

    // Schedule tokens
    budget->AddNumBatchedTokens(req, 10, 0);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(10));
    EXPECT_EQ(budget->numCachedTokens_, static_cast<size_t>(0));
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(0));
}

// ------------------ Leftover Capacity & Mid-scheduling ------------------ //

TEST_F(SchedulingBudgetTest, CheckLeftoverCapacityAndMidScheduling)
{
    // First request
    RequestId req1 = "reqLeftover1";
    EXPECT_TRUE(budget->CanSchedule(50, 2));

    budget->AddNumBatchedTokens(req1, 50, 0);
    budget->AddNumSeqs(req1, 2);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(50));
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(2));

    // Exceed budget
    EXPECT_FALSE(budget->CanSchedule(60, 1));

    // 40 is feasible
    EXPECT_TRUE(budget->CanSchedule(40, 1));

    // Second request
    RequestId req2 = "reqLeftover2";
    budget->AddNumBatchedTokens(req2, 40, 0);
    budget->AddNumSeqs(req2, 1);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(90)); // 50 + 40
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(3));        // 2 + 1

    // 20 now exceeds total of 100
    EXPECT_FALSE(budget->CanSchedule(20, 1));
    EXPECT_TRUE(budget->CanSchedule(10, 1));
}

TEST_F(SchedulingBudgetTest, ReCheckAfterSubtracting)
{
    // Fill budget
    RequestId reqA = "reqRecheckA";
    budget->AddNumBatchedTokens(reqA, 50, 0);
    budget->AddNumSeqs(reqA, 5);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(50));
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(5));

    // Subtract usage
    budget->SubtractNumBatchedTokens(reqA, 50);
    budget->SubtractNumSeqs(reqA, 5);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(0));
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(0));

    // Full capacity available again
    EXPECT_TRUE(budget->CanSchedule(100, 10));

    // Use it up
    RequestId reqB = "reqRecheckB";
    budget->AddNumBatchedTokens(reqB, 100, 0);
    budget->AddNumSeqs(reqB, 10);
    EXPECT_EQ(budget->numBatchedTokens_, static_cast<size_t>(100));
    EXPECT_EQ(budget->numCurSeqs_, static_cast<size_t>(10));

    // No more room
    EXPECT_FALSE(budget->CanSchedule(1, 1));
}

} // namespace mindie_llm