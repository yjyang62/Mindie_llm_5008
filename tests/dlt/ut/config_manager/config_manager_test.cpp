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

#include <thread>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#define private public
#include "common_util.h"
#include "dt_tools.h"
#include "config_manager.h"
#include "base_config_manager.h"
#include "mock_util.h"
#include "env_util.h"

using Json = nlohmann::json;
using namespace mindie_llm;
MOCKER_CPP_OVERLOAD_EQ(mindie_llm::ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(mindie_llm::Error)

namespace mindie_llm {
class ConfigManagerTest : public testing::Test {
protected:
    void SetUp()
    {
        jsonPath = GetCwdDirectory() + "/conf/config.json";
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetCwdDirectory() + "/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
    }

    void TearDown() { GlobalMockObject::verify(); }

    std::string jsonPath;
};

TEST_F(ConfigManagerTest, TestGetInterfaces)
{
    EXPECT_THROW(ConfigManager::GetInstance(), std::runtime_error);
    MOCKER_CPP(&GetConfigPath, Error (*)(std::string &)).stubs().will(returnValue(Error(Error::Code::INVALID_ARG)));
    EXPECT_FALSE(ConfigManager::CreateInstance(""));
    GlobalMockObject::verify();
    EXPECT_FALSE(ConfigManager::CreateInstance(jsonPath + "1"));
    EXPECT_TRUE(ConfigManager::CreateInstance(jsonPath));
    EXPECT_TRUE(ConfigManager::CreateInstance(jsonPath));
    auto backendConfig = ConfigManager::GetInstance().GetBackendConfig();
    EXPECT_EQ(backendConfig.backendName, "mindieservice_llm_engine");
    auto serverConfig = ConfigManager::GetInstance().GetServerConfig();
    EXPECT_FALSE(serverConfig.httpsEnabled);
    auto scheduleConfig = ConfigManager::GetInstance().GetScheduleConfig();
    EXPECT_EQ(scheduleConfig.templateType, "Standard");
    auto modelDeployConfig = ConfigManager::GetInstance().GetModelDeployConfig();
    EXPECT_EQ(modelDeployConfig[0].modelInstanceType, "StandardMock");
    auto ranktableParam = ConfigManager::GetInstance().GetRanktableParam();
    EXPECT_EQ(ranktableParam.serverCount, 2);
    auto logConfig = ConfigManager::GetInstance().GetLogConfig();
    EXPECT_FALSE(logConfig.logToStdOut_);
    auto loraConfig = ConfigManager::GetInstance().GetLoraConfig();
    EXPECT_EQ(loraConfig.size(), 1);
    EXPECT_EQ(loraConfig[0].loraName, "llama_65b");
}

TEST_F(ConfigManagerTest, TestCheckInterfaces) { EXPECT_TRUE(ConfigManager::GetInstance().CheckAndInitLogParam()); }

TEST_F(ConfigManagerTest, TestIsMultiNodeInfer) { EXPECT_TRUE(ConfigManager::GetInstance().IsMultiNodeInfer()); }

TEST_F(ConfigManagerTest, TestGetConfigJsonStr)
{
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        std::cout << "Error: Open json file failed" << std::endl;
    }
    Json jsonData;
    try {
        file >> jsonData;
        file.close();
    } catch (const std::exception &e) {
        file.close();
        std::cout << "Json file is invaild. Please check json format" << std::endl;
    }
    std::string ret = ConfigManager::GetInstance().GetConfigJsonStr();
    EXPECT_TRUE(ret != jsonData);
}

TEST_F(ConfigManagerTest, TestSetMaxPositionEmbeddings)
{
    uint32_t maxPositionEmbeddings = 1;
    ConfigManager::GetInstance().SetMaxPositionEmbeddings(maxPositionEmbeddings);
    auto modelDeployConfig = ConfigManager::GetInstance().GetModelDeployConfig();
    EXPECT_EQ(modelDeployConfig[0].maxPositionEmbeddings, 1);
}

TEST_F(ConfigManagerTest, TestCheckAllParam)
{
    MOCKER(ParamChecker::CheckPath).stubs().will(returnValue(true));
    std::string newJsonPath = GetCwdDirectory() + "/conf/interNodeTLSConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = {{"BackendConfig", {{"interNodeTLSEnabled", true}}}};
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    EXPECT_EQ(ConfigManager::GetInstance().CheckAllParam(), true);
}

TEST_F(ConfigManagerTest, should_retun_openAiSupport_change_to_false_when_config_change)
{
    // given 原始文件中openAiSupport 为 "vllm"
    EXPECT_EQ(ConfigManager::GetInstance().GetServerConfig().openAiSupportedvLLM, true);

    // when 文件发生变化，openAiSupport 从 "vllm" 变为 "openai"
    std::ifstream inFile(jsonPath);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(inFile, line)) {
        lines.push_back(line);
    }
    inFile.close();

    if (lines.size() > 39) {                           // 确保文件至少有40行
        lines[39] = "\"openAiSupport\" : \"openai\","; // 正确转义双引号
    }
    {
        std::ofstream outFile(jsonPath);
        for (const auto &l : lines) {
            outFile << l << "\n";
        }
        outFile.close();
    }
    // then 间隔20s, 配置发生变化, openAiSupportedvLLM 为 false
    sleep(20);
    EXPECT_EQ(ConfigManager::GetInstance().GetServerConfig().openAiSupportedvLLM, false);
    // then 恢复配置
    if (lines.size() > 39) {                         // 确保文件至少有40行
        lines[39] = "\"openAiSupport\" : \"vllm\","; // 正确转义双引号
    }
    {
        std::ofstream outFile(jsonPath);
        for (const auto &l : lines) {
            outFile << l << "\n";
        }
        outFile.close();
    }

    sleep(20);
    EXPECT_EQ(ConfigManager::GetInstance().GetServerConfig().openAiSupportedvLLM, true);
}

} // namespace mindie_llm