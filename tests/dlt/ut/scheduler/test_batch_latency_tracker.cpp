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
#include "batch_latency_tracker.h"


using namespace mindie_llm;

class BatchLatencyTrackerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        batchLatencyTracker_ = std::make_shared<BatchLatencyTracker>(2);
    }

    std::shared_ptr<BatchLatencyTracker> batchLatencyTracker_;
};

// 测试AddDataPoint的size限制
TEST_F(BatchLatencyTrackerTest, TestAddDataPoint)
{
    batchLatencyTracker_->AddDataPoint(1);
    batchLatencyTracker_->AddDataPoint(2);
    batchLatencyTracker_->AddDataPoint(3);
    EXPECT_EQ(2, batchLatencyTracker_->queue_.size());
}

// 测试数据点的平均延迟时间计算
TEST_F(BatchLatencyTrackerTest, TestGetRecentAvgLatency)
{
    batchLatencyTracker_->AddDataPoint(1);
    batchLatencyTracker_->AddDataPoint(2);
    batchLatencyTracker_->AddDataPoint(3);
    EXPECT_EQ(2, batchLatencyTracker_->queue_.size());
    EXPECT_FLOAT_EQ(2.5, batchLatencyTracker_->GetRecentAvgLatency(2));
}