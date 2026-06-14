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
#include <string>
#include <climits>
#include <unistd.h>
#include <common_util.h>
#include <file_utils.h>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
using Json = nlohmann::json;

namespace mindie_llm {


// Test Suite for CanonicalPath function
class FileUtilsTest : public ::testing::Test {
public:
    std::string configPath_;
    std::string homePath_;
protected:
    void SetUp() override
    {
        std::string homePath;
        GetLlmPath(homePath);
        configPath_ = homePath + "/conf/config.json";
        homePath_ = homePath;
    }

    void TearDown() override
    {
    }
};

TEST_F(FileUtilsTest, TestIsFileValid)
{
    std::string errMsg;
    bool isValid = FileUtils::IsFileValid(configPath_, errMsg);
    EXPECT_TRUE(isValid);
}

TEST_F(FileUtilsTest, TestIsFileInValid)
{
    std::string errMsg;
    std::string configPath = "/home/test/noexist/config.json";
    bool isValid = FileUtils::IsFileValid(configPath, errMsg);
    EXPECT_FALSE(isValid);
}

TEST_F(FileUtilsTest, TestRegularFilePathInvalid)
{
    std::string errMsg;
    std::string configPath = "/home/test/noexist/config.json";
    std::string regularPath;
    bool isValid = FileUtils::RegularFilePath(configPath, errMsg, regularPath);
    EXPECT_FALSE(isValid);
}

TEST_F(FileUtilsTest, TestRegularFilePathIsNull)
{
    std::string errMsg;
    std::string configPath = "";
    std::string regularPath;
    bool isValid = FileUtils::RegularFilePath(configPath, errMsg, regularPath);
    EXPECT_FALSE(isValid);
}

TEST_F(FileUtilsTest, TestRegularFilePathV2FilePathIsNull)
{
    std::string errMsg;
    std::string configPath = "";
    std::string baseDir = "";
    std::string regularPath;
    bool isValid = FileUtils::RegularFilePath(configPath, baseDir, errMsg, false, regularPath);
    EXPECT_FALSE(isValid);
}

TEST_F(FileUtilsTest, TestRegularFilePathV2BaseDirIsNull)
{
    std::string errMsg;
    std::string configPath = "";
    std::string baseDir = "";
    std::string regularPath;
    bool isValid = FileUtils::RegularFilePath(configPath_, baseDir, errMsg, false, regularPath);
    EXPECT_FALSE(isValid);
}

TEST_F(FileUtilsTest, TestRegularFilePathV2RealPathInvalid)
{
    std::string errMsg;
    std::string configPath = "/home/test/noexist/config.json";
    std::string regularPath;
    bool isValid = FileUtils::RegularFilePath(configPath, homePath_, errMsg, false, regularPath);
    EXPECT_FALSE(isValid);
}

TEST_F(FileUtilsTest, TestGetModeString)
{
    const mode_t mode = 0b100'100'000;
    std::string errMsg;
    bool result = FileUtils::ConstrainPermission(configPath_, mode, errMsg);
    EXPECT_FALSE(result);
}
}