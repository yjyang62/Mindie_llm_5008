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
 
#include "gtest/gtest.h"
#include "environment.h"

class RequestTest : public ::testing::Test {
public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}
};

TEST_F(RequestTest, Start)
{
    REQUIRE_SCENARIO(A2_SINGLE_MACHINE_PD_HYBRID_SCHEDULING|A2_TWO_MACHINE_PD_HYBRID_SCHEDULING);
    
    EXPECT_TRUE(true);
}