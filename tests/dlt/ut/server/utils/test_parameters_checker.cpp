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
#include <mockcpp/mockcpp.hpp>
#define private public
#include "parameters_checker.h"

using namespace mindie_llm;
using OrderedJson = nlohmann::ordered_json;

class ParametersCheckerTest : public testing::Test {
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(ParametersCheckerTest, OptionalInt32JsonCheck_FAIL)
{
    {
        OrderedJson json = R"({"test_key": 150})"_json;
        std::optional<int32_t> output;
        std::string error;
        auto validator = [](const int64_t& val, std::stringstream& err) {
            if (val > 100) err << "Value too large";
            return val <= 100;
        };
        
        bool result = ParametersChecker::OptionalInt32JsonCheck(
            json, "test_key", output, error, validator);
        
        EXPECT_FALSE(result);
        EXPECT_FALSE(output.has_value());
        EXPECT_EQ("Value too large", error);
    }
    {
        OrderedJson json = R"({"test_key": 3.14})"_json;
        std::optional<int32_t> output;
        std::string error;
        auto validator = [](const int64_t &val, std::stringstream &) { return true; };
        bool result = ParametersChecker::OptionalInt32JsonCheck(
            json, "test_key", output, error, validator);
        
        EXPECT_FALSE(result);
        EXPECT_FALSE(output.has_value());
        EXPECT_EQ("test_key must be integer type.", error);
    }
}

TEST_F(ParametersCheckerTest, OptionalInt32JsonCheck_SUCCESS)
{
    {
        OrderedJson json = R"({"test_key": 42})"_json;
        std::optional<int32_t> output;
        std::string error;
        auto validator = [](const int64_t& val, std::stringstream&) {
            return val >= 0 && val <= 100;
        };
        
        bool result = ParametersChecker::OptionalInt32JsonCheck(
            json, "test_key", output, error, validator);
        
        EXPECT_TRUE(result);
        ASSERT_TRUE(output.has_value());
        EXPECT_EQ(42, output.value());
        EXPECT_TRUE(error.empty());
    }
    {
        OrderedJson json = R"({"test_key": null})"_json;
        std::optional<int32_t> output = 99;
        std::string error;
        
        bool result = ParametersChecker::OptionalInt32JsonCheck(
            json, "test_key", output, error, {});
        
        EXPECT_TRUE(result);
        EXPECT_TRUE(output.has_value());
        EXPECT_TRUE(error.empty());
    }
}

TEST_F(ParametersCheckerTest, OptionalUInt64JsonCheck_FAIL)
{
    {
        OrderedJson json = R"({"test_key": 5})"_json;
        std::optional<uint64_t> output;
        std::string error;
        auto validator = [](const uint64_t& val, std::stringstream& err) {
            if (val < 10) err << "Value too small";
            return val >= 10;
        };
        
        bool result = ParametersChecker::OptionalUInt64JsonCheck(
            json, "test_key", output, error, validator);
        
        EXPECT_FALSE(result);
        EXPECT_FALSE(output.has_value());
        EXPECT_EQ("Value too small", error);
    }
    {
        OrderedJson json = R"({"test_key": -42})"_json;
        std::optional<uint64_t> output;
        std::string error;
        auto validator = [](const uint64_t &val, std::stringstream &) { return true; };
        bool result = ParametersChecker::OptionalUInt64JsonCheck(
            json, "test_key", output, error, validator);
        
        EXPECT_FALSE(result);
        EXPECT_FALSE(output.has_value());
        EXPECT_EQ("test_key must be unsigned type.", error);
    }
}
TEST_F(ParametersCheckerTest, OptionalUInt64JsonCheck_SUCCESS)
{
    {
        OrderedJson json = R"({"test_key": 42})"_json;
        std::optional<uint64_t> output;
        std::string error;
        auto validator = [](const uint64_t& val, std::stringstream&) {
            return val >= 10 && val <= 1000;
        };
        
        bool result = ParametersChecker::OptionalUInt64JsonCheck(
            json, "test_key", output, error, validator);
        
        EXPECT_TRUE(result);
        ASSERT_TRUE(output.has_value());
        EXPECT_EQ(42u, output.value());
        EXPECT_TRUE(error.empty());
    }
    {
        OrderedJson json = R"({"test_key": null})"_json;
        std::optional<uint64_t> output = 99;
        std::string error;
        
        bool result = ParametersChecker::OptionalUInt64JsonCheck(
            json, "test_key", output, error, {});
        
        EXPECT_TRUE(result);
        EXPECT_TRUE(output.has_value());
        EXPECT_TRUE(error.empty());
    }
}