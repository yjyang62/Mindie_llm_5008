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
#include "prefill_regression.h"


using namespace mindie_llm;

class PrefillRegressionTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        prefillRegression_ = std::make_shared<PrefillRegression>();
    }

    std::shared_ptr<PrefillRegression> prefillRegression_;
};

// 测试更新prefill耗时预测的拟合参数
TEST_F(PrefillRegressionTest, TestLinearRegression)
{
    prefillRegression_->count_ = 1;
    float tokenNum1 = 1.0, execTime1 = 2.0;
    prefillRegression_->LinearRegression(tokenNum1, execTime1);
    EXPECT_FLOAT_EQ(tokenNum1, prefillRegression_->sumX_);
    EXPECT_FLOAT_EQ(execTime1, prefillRegression_->sumY_);
    EXPECT_FLOAT_EQ(tokenNum1 * tokenNum1, prefillRegression_->sumXX_);
    EXPECT_FLOAT_EQ(tokenNum1 * execTime1, prefillRegression_->sumXY_);

    prefillRegression_->count_ = 2;
    float tokenNum2 = 3.0, execTime2 = 4.0;
    prefillRegression_->LinearRegression(tokenNum2, execTime2);
    EXPECT_FLOAT_EQ(tokenNum1 + tokenNum2, prefillRegression_->sumX_);
    EXPECT_FLOAT_EQ(execTime1 + execTime2, prefillRegression_->sumY_);
    EXPECT_FLOAT_EQ(tokenNum1 * tokenNum1 + tokenNum2 * tokenNum2, prefillRegression_->sumXX_);
    EXPECT_FLOAT_EQ(tokenNum1 * execTime1 + tokenNum2 * execTime2, prefillRegression_->sumXY_);
    EXPECT_FLOAT_EQ(1.0, prefillRegression_->slope_);
    EXPECT_FLOAT_EQ(1.0, prefillRegression_->intercept_);
}

// 测试为prefill耗时预测添加一个数据点
TEST_F(PrefillRegressionTest, TestAddDataPoint)
{
    float tokenNum1 = 1.0, execTime1 = 2.0;
    prefillRegression_->AddDataPoint(tokenNum1, execTime1);
    EXPECT_FLOAT_EQ(tokenNum1, prefillRegression_->sumX_);
    EXPECT_FLOAT_EQ(execTime1, prefillRegression_->sumY_);
    EXPECT_FLOAT_EQ(tokenNum1 * tokenNum1, prefillRegression_->sumXX_);
    EXPECT_FLOAT_EQ(tokenNum1 * execTime1, prefillRegression_->sumXY_);

    float tokenNum2 = 3.0, execTime2 = 4.0;
    prefillRegression_->AddDataPoint(tokenNum2, execTime2);
    EXPECT_FLOAT_EQ(tokenNum1 + tokenNum2, prefillRegression_->sumX_);
    EXPECT_FLOAT_EQ(execTime1 + execTime2, prefillRegression_->sumY_);
    EXPECT_FLOAT_EQ(tokenNum1 * tokenNum1 + tokenNum2 * tokenNum2, prefillRegression_->sumXX_);
    EXPECT_FLOAT_EQ(tokenNum1 * execTime1 + tokenNum2 * execTime2, prefillRegression_->sumXY_);
    EXPECT_FLOAT_EQ(1.0, prefillRegression_->slope_);
    EXPECT_FLOAT_EQ(1.0, prefillRegression_->intercept_);
}

// 测试预测一次prefill耗时
TEST_F(PrefillRegressionTest, TestPredict)
{
    prefillRegression_->slope_ = 1.0;
    prefillRegression_->intercept_ = 1.0;
    int tokenNum = 10;
    EXPECT_FLOAT_EQ(prefillRegression_->Predict(tokenNum), prefillRegression_->slope_ * tokenNum + prefillRegression_->intercept_);
}