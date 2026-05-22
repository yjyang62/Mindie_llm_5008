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
#include "random_generator.h"

namespace mindie_llm {
class RandomGeneratorTest : public testing::Test {
protected:
    void SetUp() {}

    void TearDown() {}
};

TEST_F(RandomGeneratorTest, GetInstance)
{
    auto ret = RandomGenerator::GetInstance();
    EXPECT_TRUE(ret != nullptr);
}

TEST_F(RandomGeneratorTest, GetRand)
{
    auto ret = RandomGenerator::GetInstance()->GetRand();
    EXPECT_TRUE(ret > 0);
}

TEST_F(RandomGeneratorTest, GenerateHTTPRequestUUID)
{
    std::string requestId = GenerateHTTPRequestUUID();
    EXPECT_FALSE(requestId.empty());
}

} // namespace mindie_llm