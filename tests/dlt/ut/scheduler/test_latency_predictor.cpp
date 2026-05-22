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
#include "latency_predictor.h"


using namespace mindie_llm;

class LatencyPredictorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        latencyPredictor_ = std::make_shared<LatencyPredictor>();
    }

    std::shared_ptr<LatencyPredictor> latencyPredictor_;
};

// 测试根据当前batch的信息，为prefill或decode预测添加一次数据
TEST_F(LatencyPredictorTest, TestUpdateBatchStats)
{
    uint32_t batchId = 1;
    uint32_t tokenNum1 = 1;
    float execTime1 = 2.0;

    latencyPredictor_->batchId_ = batchId;
    latencyPredictor_->batchStatsMap_.Insert(batchId, BatchStatsPtr(new BatchStats{ForwardMode::PREFILL, tokenNum1, 0, execTime1}));
    latencyPredictor_->UpdateBatchStats();
    EXPECT_FLOAT_EQ(tokenNum1, latencyPredictor_->prefillRegression_.sumX_);
    EXPECT_FLOAT_EQ(tokenNum1 * tokenNum1, latencyPredictor_->prefillRegression_.sumXX_);
}

// 测试根据当前batch的信息，预测一次prefill或decode的耗时
TEST_F(LatencyPredictorTest, TestPredictBatchExecTime)
{
    latencyPredictor_->prefillRegression_.slope_ = 1.0;
    latencyPredictor_->prefillRegression_.intercept_ = 1.0;
    BatchStats bs1{ForwardMode::PREFILL, 10};
    EXPECT_FLOAT_EQ(11.0, latencyPredictor_->PredictBatchExecTime(bs1));

    latencyPredictor_->decodeRegression_.coefficients_ = {{1.0, 1.0, 1.0}};
    BatchStats bs2{ForwardMode::DECODE, 2, 2};
    EXPECT_FLOAT_EQ(5.0, latencyPredictor_->PredictBatchExecTime(bs2));
}

// 测试获取decode的平均耗时
TEST_F(LatencyPredictorTest, TestGetDecodeRecentAvgLatency)
{
    latencyPredictor_->decodeLatency_.AddDataPoint(1);
    latencyPredictor_->decodeLatency_.AddDataPoint(2);
    latencyPredictor_->decodeLatency_.AddDataPoint(3);

    EXPECT_FLOAT_EQ(2.0, latencyPredictor_->GetDecodeRecentAvgLatency(3));
}
