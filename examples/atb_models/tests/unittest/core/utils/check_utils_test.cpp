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
#include <vector>
#include <climits>
#include "atb_speed/utils/check_util.h"

namespace atb_speed {
TEST(CheckUtils, TestCheckParam)
{
    int minParam = 0, maxParam = 10;
    int testParam = 5;
    EXPECT_EQ(CheckParamRange(testParam, minParam, maxParam), 5);
    EXPECT_THROW(CheckParamRange(-1, minParam, maxParam), std::runtime_error);
    EXPECT_THROW(CheckParamRange(11, minParam, maxParam), std::runtime_error);
    
    EXPECT_EQ(CheckNumHiddenLayersValid(100), 100);
    
    EXPECT_EQ(CheckPositive(maxParam), maxParam);
    EXPECT_THROW(CheckPositive(-1), std::runtime_error);

    std::vector<std::vector<int>> intParams;
    size_t layerNum = 5;
    size_t threshold = 4;
    for (size_t i = 0; i < layerNum - 1; ++i) {
        intParams.push_back({minParam, minParam, minParam, minParam});
    }
    EXPECT_THROW(CheckLinearParamsSufficient(intParams, layerNum, threshold), std::runtime_error);
    intParams.push_back({minParam, minParam, minParam, minParam});
    EXPECT_NO_THROW(CheckLinearParamsSufficient(intParams, layerNum, threshold));
    intParams = {{minParam, minParam, minParam, minParam}};
    EXPECT_THROW(CheckLinearParamsSufficient(intParams, layerNum, threshold), std::runtime_error);
}

TEST(CheckUtils, TestCheckLayerParam)
{
    int placeholder = 1;
    std::vector<std::vector<int>> intParams;
    size_t layerNum = 5;
    size_t threshold = 2;
    for (size_t i = 0; i < layerNum - 1; ++i) {
        intParams.push_back({placeholder, placeholder});
    }
    EXPECT_THROW(CheckLinearParamsSufficient(intParams, layerNum, threshold), std::runtime_error);
    intParams.push_back({placeholder});
    EXPECT_THROW(CheckLinearParamsSufficient(intParams, layerNum, threshold), std::runtime_error);
    intParams[layerNum - 1].push_back(placeholder);
    EXPECT_NO_THROW(CheckLinearParamsSufficient(intParams, layerNum, threshold));
    EXPECT_NO_THROW(CheckPackQuantParamsSufficient(intParams, layerNum));
    EXPECT_THROW(CheckLinearPackParamsSufficient(intParams, layerNum), std::runtime_error);
}

TEST(CheckUtils, TestCheckIntMulOverFlow)
{
    int a1 = 1;
    uint32_t b1 = 6;
    EXPECT_THROW(CheckIntMulOverFlow(a1, b1), std::runtime_error);
    EXPECT_EQ(CheckIntMulOverFlow(a1, 0), 0);
    int a2 = 5, b2 = 6;
    EXPECT_EQ(CheckIntMulOverFlow(a2, b2), 30);
    int a3 = INT_MAX - 1, a4 = INT_MIN + 1;
    uint32_t a5 = UINT32_MAX - 1;
    EXPECT_THROW(CheckIntMulOverFlow(a3, b2), std::overflow_error);
    EXPECT_THROW(CheckIntMulOverFlow(a3, -b2), std::overflow_error);
    EXPECT_THROW(CheckIntMulOverFlow(a4, b2), std::overflow_error);
    EXPECT_THROW(CheckIntMulOverFlow(a4, -b2), std::overflow_error);
    EXPECT_THROW(CheckIntMulOverFlow(a5, b1), std::overflow_error);
}
} // namespace atb_speed