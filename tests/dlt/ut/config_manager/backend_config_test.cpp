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
#include "mockcpp/mockcpp.hpp"
#include "file_utils.h"
#include "param_checker.h"
#include "common_util.h"
#include "param_checker.h"
#include "base_config_manager.h"
#include "dt_tools.h"

using namespace mindie_llm;

namespace mindie_llm {
bool operator==(const Error& lhs, const Error& rhs)
{
    std::string a = lhs.ToString();
    std::string b = rhs.ToString();
    return true;
}
class BackendConfigTest : public testing::Test {
protected:
    void SetUp()
    {
        jsonPath = GetCwdDirectory() + "/conf/config.json";
    }

    void TearDown()
    {
        GlobalMockObject::verify();
    }

    DeviceEle CreateDeviceEle(const std::string& deviceId,
                         const std::string& rankId = "0",
                         const std::string& deviceIp = "127.0.0.1")
                         {
    return {deviceId, rankId, deviceIp};
    }

    std::unique_ptr<BackendConfigManager> configManager;
    std::string jsonPath;
};

TEST_F(BackendConfigTest, TestInitFromJson_Fail)
{
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = {
        {"BackendConfig", {{"interNodeTLSEnabled", true}}}
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    configManager = std::make_unique<BackendConfigManager>(newJsonPath);
    MOCKER(BaseConfig::CheckSystemConfig).stubs().will(returnValue(false));
    EXPECT_FALSE(configManager->InitFromJson());
    GlobalMockObject::verify();
    MOCKER(ParamChecker::CheckJsonParamType).stubs().will(returnValue(false));
    EXPECT_FALSE(configManager->InitFromJson());
    GlobalMockObject::verify();
    MOCKER(ParamChecker::CheckJsonArray).stubs().will(returnValue(false));
    EXPECT_FALSE(configManager->InitFromJson());
}

TEST_F(BackendConfigTest, TestInitFromJson)
{
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = {
        {"BackendConfig", {{"interNodeTLSEnabled", true}}}
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    configManager = std::make_unique<BackendConfigManager>(newJsonPath);
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
        .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, bool, std::string&)>(FileUtils::RegularFilePath))
        .stubs().will(returnValue(true));
    MOCKER(ParamChecker::CheckPath).stubs().will(returnValue(true));
    EXPECT_TRUE(configManager->InitFromJson());
    EXPECT_TRUE(configManager->CheckParam());
    EXPECT_TRUE(configManager->CheckBackendInterTlsParam());
}

TEST_F(BackendConfigTest, CheckInterTlsParam)
{
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = {
        {"BackendConfig", {{"interNodeTLSEnabled", true}}}
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    configManager = std::make_unique<BackendConfigManager>(newJsonPath);
    MOCKER(ParamChecker::CheckPath).stubs().will(returnValue(true));
    EXPECT_TRUE(configManager->CheckInterTlsParam());
}

TEST_F(BackendConfigTest, CheckBackendInterTlsParam_OpenTls)
{
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = {
        {"BackendConfig", {{"interNodeTLSEnabled", true}}}
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    configManager = std::make_unique<BackendConfigManager>(newJsonPath);
    configManager->InitFromJson();
    EXPECT_FALSE(configManager->CheckBackendInterTlsParam());
}

TEST_F(BackendConfigTest, CheckBackendInterTlsParam_CloseTls)
{
    configManager = std::make_unique<BackendConfigManager>(jsonPath);
    configManager->InitFromJson();
    EXPECT_TRUE(configManager->CheckBackendInterTlsParam());
}

TEST_F(BackendConfigTest, UpdateMultiNodesInfer_ValidDeviceIds1)
{
    configManager = std::make_unique<BackendConfigManager>(jsonPath);
    configManager->InitFromJson();

    RanktableParam param;
    param.worldSize = 0;
    param.local.device = {};
    configManager->UpdateMultiNodesInfer(param);
    
    EXPECT_EQ(configManager->GetParam().worldSize, 0);
    EXPECT_TRUE(configManager->GetParam().npuDeviceIds[0].empty());
}

TEST_F(BackendConfigTest, UpdateMultiNodesInfer_ValidDeviceIds2)
{
    configManager = std::make_unique<BackendConfigManager>(jsonPath);
    configManager->InitFromJson();

    RanktableParam param;
    param.worldSize = 4;
    param.local.device = {
        CreateDeviceEle("0"), CreateDeviceEle("1"), CreateDeviceEle("2"), CreateDeviceEle("3")
    };
    configManager->UpdateMultiNodesInfer(param);
    
    EXPECT_EQ(configManager->GetParam().worldSize, 4);
    EXPECT_EQ(configManager->GetParam().npuDeviceIds[0].size(), 4);

    const auto& deviceIds = configManager->GetParam().npuDeviceIds[0];
    EXPECT_TRUE(deviceIds.find(0) != deviceIds.end());
    EXPECT_TRUE(deviceIds.find(1) != deviceIds.end());
    EXPECT_TRUE(deviceIds.find(2) != deviceIds.end());
    EXPECT_TRUE(deviceIds.find(3) != deviceIds.end());
}

TEST_F(BackendConfigTest, UpdateMultiNodesInfer_ValidDeviceIds3)
{
    configManager = std::make_unique<BackendConfigManager>(jsonPath);
    configManager->InitFromJson();

    RanktableParam param;
    param.worldSize = 3;
    param.local.device = {
        CreateDeviceEle("0"), CreateDeviceEle("abc"), CreateDeviceEle("2")
    };
    configManager->UpdateMultiNodesInfer(param);
    
    EXPECT_EQ(configManager->GetParam().worldSize, 3);
    EXPECT_EQ(configManager->GetParam().npuDeviceIds[0].size(), 1);
}

TEST_F(BackendConfigTest, UpdateMultiNodesInfer_ValidDeviceIds4)
{
    configManager = std::make_unique<BackendConfigManager>(jsonPath);
    configManager->InitFromJson();

    RanktableParam param;
    param.worldSize = 3;
    param.local.device = {
        CreateDeviceEle("0"), CreateDeviceEle("2"), CreateDeviceEle("99999999999999999999")
    };
    configManager->UpdateMultiNodesInfer(param);
    
    EXPECT_EQ(configManager->GetParam().worldSize, 3);
    EXPECT_EQ(configManager->GetParam().npuDeviceIds[0].size(), 2);
}

TEST_F(BackendConfigTest, CheckInterTlsParam_GetHomePathError)
{
    configManager = std::make_unique<BackendConfigManager>(jsonPath);
    MOCKER(GetHomePath)
        .stubs()
        .will(returnValue(Error(Error::Code::ERROR, "ERROR: Failed to get real path of home.")));
    nlohmann::json json = {};
    EXPECT_NO_THROW(configManager->CheckSystemConfig("", json, ""));
}

} // namespace mindie_llm