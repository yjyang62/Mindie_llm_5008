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

#include <libgen.h>
#include <filesystem>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "param_checker.h"
#include "file_utils.h"
#include "common_util.h"
#include "config_manager.h"
#include "base_config_manager.h"
#include "dt_tools.h"

using Json = nlohmann::json;
using namespace mindie_llm;
namespace fs = std::filesystem;

namespace mindie_llm {
class ParamCheckerTest : public testing::Test {
protected:
    void SetUp()
    {
        tempDir = fs::temp_directory_path() / "mindie_test";
        fs::create_directory(tempDir);
    }

    void TearDown()
    {
        fs::remove_all(tempDir);
        GlobalMockObject::verify();
    }

    void CreateLoraJson(const std::string& content)
    {
        auto jsonPath = tempDir / "lora_adapter.json";
        std::ofstream file(jsonPath);
        file << content;
        file.close();
    }

    Json json;
    fs::path tempDir;
};

TEST_F(ParamCheckerTest, IsWithinRange)
{
    std::string integerType = "int32_t";
    json = "1";
    bool ret = ParamChecker::IsWithinRange(integerType, json);
    EXPECT_EQ(ret, true);

    json = "-1";
    ret = ParamChecker::IsWithinRange(integerType, json);
    EXPECT_EQ(ret, true);
}

TEST_F(ParamCheckerTest, GetJsonData)
{
    std::string baseDir;
    Json jsonData;
    std::string jsonPath = GetCwdDirectory() + "/conf/config.json";
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
        .stubs().will(returnValue(true));
    bool ret = ParamChecker::GetJsonData(jsonPath, baseDir, jsonData);
    EXPECT_EQ(ret, true);
}

TEST_F(ParamCheckerTest, GetIntegerParamDefaultValueCase1)
{
    Json jsonData;
    std::string configName = "mockConfig";
    uint32_t defaultVal = 2;

    jsonData[configName] = 1;
    MOCKER(ParamChecker::IsWithinRange).stubs().will(returnValue(true));
    uint32_t ret = ParamChecker::GetIntegerParamDefaultValue(jsonData, configName, defaultVal);
    EXPECT_EQ(ret, 1);
}

TEST_F(ParamCheckerTest, GetIntegerParamDefaultValueCase2)
{
    Json jsonData;
    std::string configName = "mockConfig";
    uint32_t defaultVal = 2;

    uint32_t ret = ParamChecker::GetIntegerParamDefaultValue(jsonData, configName, defaultVal);
    EXPECT_EQ(ret, defaultVal);
}

TEST_F(ParamCheckerTest, GetIntegerParamDefaultValueCase3)
{
    Json jsonData;
    std::string configName = "mockConfig";
    uint32_t defaultVal = 2;

    jsonData[configName] = "1";
    uint32_t ret = ParamChecker::GetIntegerParamDefaultValue(jsonData, configName, defaultVal);
    EXPECT_EQ(ret, defaultVal);
}

TEST_F(ParamCheckerTest, GetIntegerParamDefaultValueCase4)
{
    Json jsonData;
    std::string configName = "mockConfig";
    uint32_t defaultVal = 2;

    jsonData[configName] = 1;
    MOCKER(ParamChecker::IsWithinRange).stubs().will(returnValue(false));
    uint32_t ret = ParamChecker::GetIntegerParamDefaultValue(jsonData, configName, defaultVal);
    EXPECT_EQ(ret, defaultVal);
}

TEST_F(ParamCheckerTest, IsWithinRangeCase)
{
    ASSERT_FALSE(ParamChecker::IsWithinRange("int32_t",
        nlohmann::json(std::to_string(std::numeric_limits<uint32_t>::max()))));
    ASSERT_TRUE(ParamChecker::IsWithinRange("int32_t",
        nlohmann::json(std::to_string(std::numeric_limits<int32_t>::min()))));
    ASSERT_FALSE(ParamChecker::IsWithinRange("uint32_t",
        nlohmann::json(std::to_string(std::numeric_limits<int32_t>::min()))));
    ASSERT_TRUE(ParamChecker::IsWithinRange("uint32_t", nlohmann::json("99999999")));
    int32_t testNumber = -99999999;
    ASSERT_TRUE(ParamChecker::IsWithinRange("int32_t", nlohmann::json(testNumber)));
}

TEST_F(ParamCheckerTest, CheckJsonArray)
{
    ASSERT_FALSE(ParamChecker::CheckJsonArray(nlohmann::json(std::vector<int>{1, 2, 3}), "string", "int32_t"));
    ASSERT_FALSE(ParamChecker::CheckJsonArray(nlohmann::json(
        std::vector<std::string>{"1", "2", "3"}), "integer", "int32_t"));
    ASSERT_FALSE(ParamChecker::CheckJsonArray(nlohmann::json(std::vector<int>{-1, -2, -3}), "integer", "uint32_t"));
    ASSERT_FALSE(ParamChecker::CheckJsonArray(nlohmann::json(
        std::vector<std::string>{"1", "2", "3"}), "bool", "uint32_t"));
    ASSERT_TRUE(ParamChecker::CheckJsonArray(nlohmann::json(std::vector<int>{1, 2, 3}), "integer", "uint32_t"));
}

TEST_F(ParamCheckerTest, CheckNpuRange)
{
    ASSERT_FALSE(ParamChecker::CheckNpuRange(nlohmann::json(1)));
    ASSERT_FALSE(ParamChecker::CheckNpuRange(nlohmann::json(std::vector<int>{1, 2, 3})));
    ASSERT_FALSE(ParamChecker::CheckNpuRange(nlohmann::json(std::vector<std::vector<int>>{{-1, -2, -3}})));
    ASSERT_TRUE(ParamChecker::CheckNpuRange(nlohmann::json(std::vector<std::vector<int>>{{1, 2, 3}})));
}

TEST_F(ParamCheckerTest, IsArrayValid)
{
    ASSERT_FALSE(ParamChecker::IsArrayValid("mock", nlohmann::json(1)));
    ASSERT_TRUE(ParamChecker::IsArrayValid("mock", nlohmann::json(std::vector<int>{1, 2, 3})));
}

TEST_F(ParamCheckerTest, CheckEngineName)
{
    int len1 = 100;
    ASSERT_FALSE(ParamChecker::CheckEngineName(std::string(len1, 'a')));
    int len2 = 50;
    ASSERT_FALSE(ParamChecker::CheckEngineName(std::string(len2, '*')));
    ASSERT_TRUE(ParamChecker::CheckEngineName(std::string(len2, 'a')));
}

TEST_F(ParamCheckerTest, CheckPolicyValue)
{
    int value = 100;
    ASSERT_FALSE(ParamChecker::CheckPolicyValue(value, "mock"));
    value = 1;
    ASSERT_TRUE(ParamChecker::CheckPolicyValue(value, "mock"));
}

TEST_F(ParamCheckerTest, CheckJsonParamType)
{
    std::vector<ParamSpec> paramSpecs;
    ParamSpec paramSpec;
    paramSpec.name = "mock";
    paramSpec.Type = "string";
    paramSpec.compulsory = true;
    paramSpecs.emplace_back(paramSpec);
    ASSERT_FALSE(ParamChecker::CheckJsonParamType(json, paramSpecs));

    json["mock"] = 1;
    ASSERT_FALSE(ParamChecker::CheckJsonParamType(json, paramSpecs));

    json["mock"] = "test";
    paramSpec.name = "num";
    paramSpec.Type = "int32_t";
    paramSpecs.emplace_back(paramSpec);
    json["num"] = "1";
    ASSERT_FALSE(ParamChecker::CheckJsonParamType(json, paramSpecs));

    json["num"] = std::numeric_limits<uint32_t>::max();
    ASSERT_FALSE(ParamChecker::CheckJsonParamType(json, paramSpecs));

    json["num"] = 1;
    paramSpec.name = "ids";
    paramSpec.Type = "array";
    paramSpecs.emplace_back(paramSpec);
    json["ids"] = 1;
    ASSERT_FALSE(ParamChecker::CheckJsonParamType(json, paramSpecs));

    json["ids"] = {1, 2, 3};
    paramSpec.name = "check";
    paramSpec.Type = "bool";
    paramSpecs.emplace_back(paramSpec);
    json["check"] = "true";
    ASSERT_FALSE(ParamChecker::CheckJsonParamType(json, paramSpecs));

    json["check"] = true;
    paramSpec.name = "obj";
    paramSpec.Type = "object";
    paramSpecs.emplace_back(paramSpec);
    json["obj"] = nullptr;
    ASSERT_FALSE(ParamChecker::CheckJsonParamType(json, paramSpecs));

    json["obj"] = {{"type", "mockType"}};
    ASSERT_TRUE(ParamChecker::CheckJsonParamType(json, paramSpecs));
}

TEST_F(ParamCheckerTest, ReadJsonFile_InvalidJsonFormat)
{
    auto invalidJsonFile = tempDir / "invalid.json";
    std::ofstream(invalidJsonFile) << "This is not valid JSON";
    
    std::string baseDir;
    Json jsonData;
    bool result = ParamChecker::ReadJsonFile(invalidJsonFile.string(), baseDir, jsonData, "");
    
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, ReadJsonFile_MissingConfigType)
{
    auto validJsonFile = tempDir / "valid.json";
    std::ofstream(validJsonFile) << R"({"other_config": {}})";
    std::string baseDir;
    Json jsonData;
    bool result = ParamChecker::ReadJsonFile(validJsonFile.string(), baseDir, jsonData, "missing_config");
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, IsWithinRange_UnsupportedType)
{
    Json jsonValue = "123";
    bool result = ParamChecker::IsWithinRange("int64_t", jsonValue);
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, IsWithinRange_InvalidArgument)
{
    Json jsonValue = "abc";
    bool result = ParamChecker::IsWithinRange("int32_t", jsonValue);
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, IsWithinRange_Int32PositiveOverflow)
{
    Json jsonValue = "2147483648";
    bool result = ParamChecker::IsWithinRange("int32_t", jsonValue);
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, IsWithinRange_Int32NegativeOverflow)
{
    Json jsonValue = "-2147483649";
    bool result = ParamChecker::IsWithinRange("int32_t", jsonValue);
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, IsWithinRange_UInt32Overflow)
{
    Json jsonValue = "4294967296";
    bool result = ParamChecker::IsWithinRange("uint32_t", jsonValue);
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, IsWithinRange_SizeTOverflow)
{
    std::string hugeNumber = "18446744073709551616";
    Json jsonValue = hugeNumber;
    
    bool result = ParamChecker::IsWithinRange("size_t", jsonValue);
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, CheckAndGetLoraJsonFile_InvalidJson)
{
    CreateLoraJson("{invalid json}");
    
    std::string baseDir = tempDir.string();
    nlohmann::json loraJsonData;
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
        .stubs().will(returnValue(true));
    bool result = ParamChecker::CheckAndGetLoraJsonFile(baseDir, loraJsonData);
    EXPECT_FALSE(result);
}

TEST_F(ParamCheckerTest, CheckAndGetLoraJsonFile_Success)
{
    CreateLoraJson(R"({
        "lora1": "/path/to/lora1",
        "lora2": "/path/to/lora2"
    })");
    
    std::string baseDir = tempDir.string();
    nlohmann::json loraJsonData;
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
        .stubs().will(returnValue(true));
        
    bool result = ParamChecker::CheckAndGetLoraJsonFile(baseDir, loraJsonData);
    EXPECT_TRUE(result);
    
    EXPECT_EQ(loraJsonData.size(), 2);
    EXPECT_EQ(loraJsonData["lora1"], "/path/to/lora1");
    EXPECT_EQ(loraJsonData["lora2"], "/path/to/lora2");
}

} // namespace mindie_llm