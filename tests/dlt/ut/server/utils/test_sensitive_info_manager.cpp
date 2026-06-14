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
#include "SensitiveInfoManager.h"

namespace mindie_llm {
class SensitiveInfoManagerTest : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(SensitiveInfoManagerTest, CopySensitiveInfo)
{
    int len = 0;
    uint32_t maxLen = 10;
    uint32_t minLen = 1;
    SensitiveInfoManager manager(nullptr, len, maxLen, minLen);

    EXPECT_FALSE(manager.CopySensitiveInfo(nullptr, len));
    EXPECT_FALSE(manager.IsValid());
    EXPECT_TRUE(manager.GetSensitiveInfoContent() == nullptr);

    std::string contentStr = "sensitive";
    auto content = contentStr.c_str();
    EXPECT_FALSE(manager.CopySensitiveInfo(content, len));
    EXPECT_FALSE(manager.IsValid());
    EXPECT_TRUE(manager.GetSensitiveInfoContent() == nullptr);

    int lenLargerMaxLen = 11;
    EXPECT_FALSE(manager.CopySensitiveInfo(content, lenLargerMaxLen));
    EXPECT_FALSE(manager.IsValid());
    EXPECT_TRUE(manager.GetSensitiveInfoContent() == nullptr);

    int lenNotMatched = 2;
    EXPECT_TRUE(manager.CopySensitiveInfo(content, lenNotMatched));
    EXPECT_TRUE(manager.IsValid());
    EXPECT_TRUE(manager.GetSensitiveInfoContent() != content);

    int lenMatched = 10;
    EXPECT_TRUE(manager.CopySensitiveInfo(content, lenMatched));
    EXPECT_TRUE(manager.IsValid());
    EXPECT_STREQ(manager.GetSensitiveInfoContent(), content);

    manager.Clear();
    EXPECT_FALSE(manager.IsValid());
    EXPECT_TRUE(manager.GetSensitiveInfoContent() == nullptr);
}

}