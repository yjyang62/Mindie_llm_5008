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
#include "gtest/gtest.h"
#include "log_config.h"
#include "log.h"

namespace mindie_llm {

class TestLogConfig : public ::testing::Test {
protected:
    std::string logFilePath = "";
    LogConfig logConfig;
};

// ToFile测试
TEST_F(TestLogConfig, TestInitLogToFileWithMINDIELOGTOFILESuccess)
{
    // Setup
    logConfig.logToFile_ = 0;
    setenv("MINDIE_LOG_TO_FILE", "1", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);

    unsetenv("MINDIE_LOG_TO_FILE");
    // Verify
    ASSERT_TRUE(logConfig.logToFile_);
}

TEST_F(TestLogConfig, TestInitLogToFileWithMINDIELOGTOFILEFalse)
{
    // Setup
    logConfig.logToFile_ = 1;
    setenv("MINDIE_LOG_TO_FILE", "0", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);

    unsetenv("MINDIE_LOG_TO_FILE");
    // Verify
    ASSERT_FALSE(logConfig.logToFile_);
}

TEST_F(TestLogConfig, TestInitLogToFileWithMINDIELLMLOGTOFILESuccess)
{
    // Setup
    logConfig.logToFile_ = 0;
    setenv("MINDIE_LOG_TO_FILE", "1", 1);

    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);

    unsetenv("MINDIE_LOG_TO_FILE");
    // Verify
    ASSERT_TRUE(logConfig.logToFile_);
}

TEST_F(TestLogConfig, TestInitLogToFileWithMINDIELLMLOGTOFILEFail)
{
    // Setup
    logConfig.logToFile_ = 1;
    setenv("MINDIE_LOG_TO_FILE", "0", 1);

    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);

    unsetenv("MINDIE_LOG_TO_FILE");
    // Verify
    ASSERT_FALSE(logConfig.logToFile_);
}

TEST_F(TestLogConfig, TestInitLogToFileWithDefaultSuccess)
{
    // Setup

    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);

    // Verify
    ASSERT_TRUE(logConfig.logToFile_);
}

// ToStdOut测试
TEST_F(TestLogConfig, TestInitLogToStdoutWithMINDIELOGTOSTDOUTSuccess)
{
    // Setup
    logConfig.logToStdOut_ = 0;
    setenv("MINDIE_LOG_TO_STDOUT", "1", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_TO_STDOUT");
    // Verify
    ASSERT_TRUE(logConfig.logToStdOut_);
}

TEST_F(TestLogConfig, TestInitLogToStdoutWithMINDIELOGTOSTDOUTFail)
{
    // Setup
    logConfig.logToStdOut_ = 1;
    setenv("MINDIE_LOG_TO_STDOUT", "0", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_TO_STDOUT");
    // Verify
    ASSERT_FALSE(logConfig.logToStdOut_);
}

TEST_F(TestLogConfig, TestInitLogToStdoutWithMINDIELLMLOGTOSTDOUTSuccess)
{
    // Setup
    logConfig.logToStdOut_ = 0;
    setenv("MINDIE_LOG_TO_STDOUT", "1", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_TO_STDOUT");
    // Verify
    ASSERT_TRUE(logConfig.logToStdOut_);
}

TEST_F(TestLogConfig, TestInitLogToStdoutWithMINDIELOGLLMTOSTDOUTFail)
{
    // Setup
    logConfig.logToStdOut_ = 1;
    setenv("MINDIE_LOG_TO_STDOUT", "0", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_TO_STDOUT");
    // Verify
    ASSERT_FALSE(logConfig.logToStdOut_);
}

TEST_F(TestLogConfig, TestInitLogToStdoutWithDefaultFail)
{
    // Setup

    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);

    // Verify
    ASSERT_FALSE(logConfig.logToStdOut_);
}

// Level测试
TEST_F(TestLogConfig, TestInitLogLevelWithMINDIELOGLEVELSuccess)
{
    // Setup
    logConfig.logLevel_ = LogLevel::info;
    setenv("MINDIE_LOG_LEVEL", "debug", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_LEVEL");
    // Verify
    ASSERT_EQ(LogLevel::debug, logConfig.logLevel_);
}

TEST_F(TestLogConfig, TestInitLogLevelWithMINDIELLMLOGLEVELSuccess)
{
    // Setup
    logConfig.logLevel_ = LogLevel::info;
    setenv("MINDIE_LOG_LEVEL", "debug", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_LEVEL");
    // Verify
    ASSERT_EQ(LogLevel::debug, logConfig.logLevel_);
}

TEST_F(TestLogConfig, TestInitLogLevelWithDefaultSuccess)
{
    // Setup

    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);

    // Verify
    ASSERT_EQ(LogLevel::info, logConfig.logLevel_);
}

// FilePath测试

// Verbose测试
TEST_F(TestLogConfig, TestInitLogVerboseWithMINDIELOGVERBOSESuccess)
{
    // Setup
    logConfig.logVerbose_ = 0;
    setenv("MINDIE_LOG_VERBOSE", "1", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_VERBOSE");
    // Verify
    ASSERT_TRUE(logConfig.logVerbose_);
}

TEST_F(TestLogConfig, TestInitLogVerboseWithMINDIELOGVERBOSEFail)
{
    // Setup
    logConfig.logVerbose_ = 1;
    setenv("MINDIE_LOG_VERBOSE", "0", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_VERBOSE");
    // Verify
    ASSERT_FALSE(logConfig.logVerbose_);
}

TEST_F(TestLogConfig, TestInitLogVerboseWithDefaultSuccess)
{
    // Setup

    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);

    // Verify
    ASSERT_TRUE(logConfig.logVerbose_);
}

// Rotation测试
TEST_F(TestLogConfig, TestInitLogRotationParamWithMINDIELOGROTATESuccess)
{
    // Setup
    setenv("MINDIE_LOG_ROTATE", "-fs 1 -r 10", 1);
    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    unsetenv("MINDIE_LOG_ROTATE");
    // Verify
    constexpr uint32_t oneMb = 1 * 1024 * 1024;
    ASSERT_EQ(logConfig.logFileSize_, oneMb);
    constexpr uint32_t fileNum = 10;
    ASSERT_EQ(fileNum, logConfig.logFileCount_);
}

TEST_F(TestLogConfig, TestInitLogRotationParamWithDefaultSuccess)
{
    // Setup

    // Exercise
    logConfig.Init(LoggerType::MINDIE_LLM);
    // Verify
    constexpr uint32_t fileSize = 20 * 1024 * 1024;
    ASSERT_EQ(fileSize, logConfig.logFileSize_);
    constexpr uint32_t fileNum = 10;
    ASSERT_EQ(fileNum, logConfig.logFileCount_);
}

}