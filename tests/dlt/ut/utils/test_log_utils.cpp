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
 
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#include "log_utils.h"
#include "log_config.h"
#include "log.h"

namespace mindie_llm {

class TestLogUtils : public ::testing::Test {
protected:
    
    void SetUp() override
    {
        envLogFlag = false;
        envLogLevel = LogLevel::info;
        envLogPath = "";
    }

    LogConfig logConfig;
    bool envLogFlag;
    LogLevel envLogLevel;
    std::string envLogPath;
    uint32_t envFileSize;
    uint32_t envFilesNum;
    std::string rotateConfig;
};

TEST_F(TestLogUtils, TestSetMindieEnvFlagWithTrueSuccess)
{
    std::string envVar = "true";
    LogUtils::SetMindieLogParamBool(LoggerType::MINDIE_LLM, envLogFlag, envVar);
    ASSERT_TRUE(envLogFlag);
}

TEST_F(TestLogUtils, TestSetMindieEnvFlagWithZeroFail)
{
    std::string envVar = "0";
    LogUtils::SetMindieLogParamBool(LoggerType::MINDIE_LLM, envLogFlag, envVar);
    ASSERT_FALSE(envLogFlag);
}

TEST_F(TestLogUtils, TestSetMindieEnvFlagWithLlmTrueSuccess)
{
    std::string envVar = "llm:true";
    LogUtils::SetMindieLogParamBool(LoggerType::MINDIE_LLM, envLogFlag, envVar);
    ASSERT_TRUE(envLogFlag);
}

TEST_F(TestLogUtils, TestSetMindieEnvFlagWithLlmFalseFail)
{
    std::string envVar = "service:true;llm:false";
    LogUtils::SetMindieLogParamBool(LoggerType::MINDIE_LLM, envLogFlag, envVar);
    ASSERT_FALSE(envLogFlag);
}

TEST_F(TestLogUtils, TestSetMindieEnvFlagWithFalseFail)
{
    std::string envVar = "service:true; false";
    LogUtils::SetMindieLogParamBool(LoggerType::MINDIE_LLM, envLogFlag, envVar);
    ASSERT_FALSE(envLogFlag);
}

TEST_F(TestLogUtils, TestSetMindieEnvLevelWithDebug)
{
    std::string envVar = "debug";
    LogUtils::SetMindieLogParamLevel(LoggerType::MINDIE_LLM, envLogLevel, envVar);
    ASSERT_EQ(LogLevel::debug, envLogLevel);
}

TEST_F(TestLogUtils, TestSetMindieEnvLevelWithCritical)
{
    std::string envVar = "llm:critical";
    LogUtils::SetMindieLogParamLevel(LoggerType::MINDIE_LLM, envLogLevel, envVar);
    ASSERT_EQ(LogLevel::critical, envLogLevel);
}

TEST_F(TestLogUtils, TestSetMindieEnvLevelWithAllDebug)
{
    std::string envVar = "service:critical;debug";
    LogUtils::SetMindieLogParamLevel(LoggerType::MINDIE_LLM, envLogLevel, envVar);
    ASSERT_EQ(LogLevel::debug, envLogLevel);
}

TEST_F(TestLogUtils, TestSetMindieLogPathWithabc)
{
    std::string envVar = "abc";
    envLogPath = LogUtils::GetEnvParam(LoggerType::MINDIE_LLM, envVar);
    ASSERT_EQ("abc", envLogPath);
}

TEST_F(TestLogUtils, TestSetMindieLogPathWithLlmabcd)
{
    std::string envVar = "llm:abcd; service:service";
    envLogPath = LogUtils::GetEnvParam(LoggerType::MINDIE_LLM, envVar);
    ASSERT_EQ("abcd", envLogPath);
}

TEST_F(TestLogUtils, TestUpdateLogFileParamWith)
{
    std::string envVar = "llm: -fs 1 -r 10";
    rotateConfig = LogUtils::GetEnvParam(LoggerType::MINDIE_LLM, envVar);
    LogUtils::UpdateLogFileParam(rotateConfig, envFileSize, envFilesNum);
    constexpr uint32_t oneMb = 1 * 1024 * 1024;
    ASSERT_EQ(oneMb, envFileSize);
    constexpr uint32_t fileNum = 10;
    ASSERT_EQ(fileNum, envFilesNum);
}

}