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
#include <iostream>
#include <cmath>
#include "hit_rate_calculator.h"

namespace mindie_llm {
class ProbeHitRateCalculator : public HitRateCalculator {
public:
    void SetHitNum(uint64_t hitNum) { hitNum_ = hitNum; };
    void SetMissNum(uint64_t missNum) { missNum_ = missNum; };
    uint64_t GetHitNum() { return hitNum_; }
    uint64_t GetMissNum() { return missNum_; }
};

TEST(HitRateCalculatorTest, BasicTest)
{
    HitRateCalculator hrCaculator;
    bool hit;
    double expectedHitRate;

    hit = true;
    hrCaculator.Record(hit);
    expectedHitRate = 1.0;
    EXPECT_DOUBLE_EQ(hrCaculator.GetHitRate(), expectedHitRate);

    hit = false;
    hrCaculator.Record(hit);
    expectedHitRate = 0.5;
    EXPECT_DOUBLE_EQ(hrCaculator.GetHitRate(), expectedHitRate);

    hit = false;
    hrCaculator.Record(hit);
    expectedHitRate = 1.0 / 3.0;
    EXPECT_DOUBLE_EQ(hrCaculator.GetHitRate(), expectedHitRate);
}
TEST(HitRateCalculatorTest, ExtremeTest)
{
    ProbeHitRateCalculator hrCaculator;
    bool hit;
    double expectedHitRate;
    uint64_t expectedTotal;

    uint64_t mockHitNum = std::pow(2, 62); // counter limit
    hrCaculator.SetHitNum(mockHitNum);
    uint64_t mockMissNum = std::pow(2, 62); // counter limit
    hrCaculator.SetMissNum(mockMissNum);

    expectedHitRate = 0.5;
    expectedTotal = std::pow(2, 62) * 2;
    EXPECT_DOUBLE_EQ(hrCaculator.GetHitRate(), expectedHitRate);
    uint64_t totalNum = hrCaculator.GetHitNum() + hrCaculator.GetMissNum(); // sum under counter limits
    EXPECT_DOUBLE_EQ((double)totalNum, expectedTotal);
}
} // namespace mindie_llm
