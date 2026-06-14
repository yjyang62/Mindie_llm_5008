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
 
#include <iostream>
#include "gtest/gtest.h"
#include "log_config.h"
#include "log.h"

namespace mindie_llm {
class TestLog : public ::testing::Test {
protected:
    LogConfig logConfig;

void SetUp() override {
    }
};

TEST_F(TestLog, TestCreateInstanceSuccess)
{
    Log::CreateInstance(LoggerType::MINDIE_LLM);
    auto llmLog = Log::GetInstance(LoggerType::MINDIE_LLM);
    ASSERT_NE(llmLog, nullptr);
}

TEST_F(TestLog, TestGetErrorCodeWithExistsErrorCode)
{
    std::ostringstream oss;
    std::string errorCode = "BACKEND_CONFIG_VAL_FAILED";
    Log::GetErrorCode(oss, errorCode);
    ASSERT_EQ(oss.str(), "[MIE05E020000] ");
}

TEST_F(TestLog, TestGetErrorCodeWithNotExistsErrorCode)
{
    std::ostringstream oss;
    std::string errorCode = "NON_EXISTENT_ERROR_CODE";
    Log::GetErrorCode(oss, errorCode);
    ASSERT_EQ(oss.str(), "");
}

TEST_F(TestLog, TestGetErrorCodeWithEmptyErrorCode)
{
    std::ostringstream oss;
    std::string errorCode = "";
    Log::GetErrorCode(oss, errorCode);
    ASSERT_EQ(oss.str(), "");
}
}