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
#include "config_manager.h"
#include "base_config_manager.h"
#include "dt_tools.h"

using namespace mindie_llm;

namespace mindie_llm {
class ScheduleConfigManagerTest : public testing::Test {
protected:
    void SetUp() { jsonPath = GetCwdDirectory() + "/conf/config.json"; }
    void TearDown() {}

    std::string jsonPath;
};

TEST_F(ScheduleConfigManagerTest, testParamFuncsSuccess)
{
    ScheduleConfigManager scheduleConfigManager(jsonPath);
    scheduleConfigManager.InitFromJson();
    EXPECT_TRUE(scheduleConfigManager.CheckParam());
    auto scheduleConfig = scheduleConfigManager.GetParam();
    EXPECT_EQ(scheduleConfig.templateType, "Standard");
}

TEST_F(ScheduleConfigManagerTest, testParamFuncsFail)
{
    ScheduleConfigManager scheduleConfigManager("");
    EXPECT_FALSE(scheduleConfigManager.CheckParam());
    auto scheduleConfig = scheduleConfigManager.GetParam();
    EXPECT_FALSE(scheduleConfig.templateType == "Standard");
}

TEST_F(ScheduleConfigManagerTest, SetMaxPreemptCount)
{
    ScheduleConfigManager scheduleConfigManager(jsonPath);
    uint32_t value = 1;
    scheduleConfigManager.SetMaxPreemptCount(value);
    auto scheduleConfig = scheduleConfigManager.GetParam();
    EXPECT_EQ(scheduleConfig.maxPreemptCount, 1);
}
} // namespace mindie_llm