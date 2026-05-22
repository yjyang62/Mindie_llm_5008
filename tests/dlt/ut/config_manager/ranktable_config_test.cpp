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

#include <cstdlib>
#include <libgen.h>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "common_util.h"
#include "file_utils.h"
#include "config_manager.h"
#include "base_config_manager.h"
#include "env_util.h"
#include <filesystem>
#include "dt_tools.h"

using namespace mindie_llm;
namespace fs = std::filesystem;
using Json = nlohmann::json;

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

namespace mindie_llm {
class RanktableConfigTest : public testing::Test {
protected:
    static void SetUpTestSuite() {}

    static void TearDownTestSuite() {}

    void SetUp() override
    {
        EnvUtil& env = EnvUtil::GetInstance();
        env.SetEnvVar("RANK_TABLE_FILE", (GetCwdDirectory() + "/conf/ranktable.json"));
        env.SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        env.SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        env.SetEnvVar("HOST_IP", "127.0.0.2");
        chmod((GetCwdDirectory() + "/conf/ranktable.json").c_str(), S_IRUSR | S_IWUSR | S_IRGRP); // 640
        tempDir = fs::temp_directory_path() / "ranktable_test";
        fs::create_directory(tempDir);
    }

    void TearDown()
    {
        EnvUtil& env = EnvUtil::GetInstance();
        env.ClearEnvVar("RANK_TABLE_FILE");
        env.ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
        env.ClearEnvVar("MIES_CONTAINER_IP");
        env.ClearEnvVar("HOST_IP");
        fs::remove_all(tempDir);
        GlobalMockObject::verify();
    }

    void CreateRanktableFile(const std::string& content = "")
    {
        auto filePath = tempDir / "ranktable.json";
        std::ofstream file(filePath);
        if (!content.empty()) {
            file << content;
        }
        file.close();
        ranktableFilePath = filePath.string();
    }

    DeviceEle CreateValidDevice()
    {
        return DeviceEle{"0", "192.168.1.1", "0"};
    }

    ServerEle CreateValidServer(int deviceCount = 8)
    {
        ServerEle server;
        server.containerIp = "192.168.1.100";
        for (int i = 0; i < deviceCount; i++) {
            server.device.push_back(CreateValidDevice());
        }
        return server;
    }

    fs::path tempDir;
    std::string ranktableFilePath;
};

TEST_F(RanktableConfigTest, testInitFromJsonSuccess)
{
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    RanktableConfigManager ranktableConfigManager;
    bool ret = ranktableConfigManager.InitFromJson();
    ASSERT_EQ(ret, true);
    ranktableConfigManager.GetParam();
    ASSERT_TRUE(ranktableConfigManager.CheckParam());
    std::string containerIPAddress = ranktableConfigManager.GetContainerIPAddress();
    std::cout << "containerIPAddress: " << containerIPAddress << std::endl;
    ASSERT_EQ(containerIPAddress, "127.0.0.1");
    std::string hostIPAddress = ranktableConfigManager.GetHostIPAddress();
    std::cout << "hostIPAddress: " << hostIPAddress << std::endl;
    ASSERT_EQ(hostIPAddress, "127.0.0.2");
}

TEST_F(RanktableConfigTest, Constructor_CheckPermissionDisabled)
{
    CreateRanktableFile(R"({"server_count": 2})");
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(false));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(false));
    EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", ranktableFilePath);
    EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "0");
    RanktableConfigManager manager;
    uint32_t serverCount = 0;
    Json serverListData;
    EXPECT_FALSE(manager.ReadRanktableData(serverCount, serverListData));
    EXPECT_FALSE(manager.InitFromJson());
}

TEST_F(RanktableConfigTest, Constructor_InvalidFilePath)
{
    std::string invalidPath = "/invalid/path/with:colon";
    EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", invalidPath);
    RanktableConfigManager manager;
    EXPECT_FALSE(manager.InitFromJson());
}

TEST_F(RanktableConfigTest, ReadRanktableData_OpenFileFailed)
{
    EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", (GetCwdDirectory() + "/conf/ksfa"));
    RanktableConfigManager manager;
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    uint32_t serverCount = 0;
    Json serverListData;
    bool result = manager.ReadRanktableData(serverCount, serverListData);
    EXPECT_FALSE(result);
}

TEST_F(RanktableConfigTest, ReadRanktableData_InvalidServerCount)
{
    CreateRanktableFile(R"({
        "server_count": "invalid_number",
        "server_list": []
    })");
    EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", ranktableFilePath);
    RanktableConfigManager manager;
    uint32_t serverCount = 0;
    Json serverListData;
    bool result = manager.ReadRanktableData(serverCount, serverListData);
    
    EXPECT_FALSE(result);
}

TEST_F(RanktableConfigTest, ReadRanktableData_ServerCountOutOfRange)
{
    CreateRanktableFile(R"({
        "server_count": "4294967296",
        "server_list": []
    })");
    EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", ranktableFilePath);
    RanktableConfigManager manager;
    uint32_t serverCount = 0;
    Json serverListData;
    bool result = manager.ReadRanktableData(serverCount, serverListData);
    
    EXPECT_FALSE(result);
}

TEST_F(RanktableConfigTest, ReadRanktableData_InvalidJsonFormat)
{
    CreateRanktableFile("invalid json");
    EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", ranktableFilePath);
    RanktableConfigManager manager;
    uint32_t serverCount = 0;
    Json serverListData;
    bool result = manager.ReadRanktableData(serverCount, serverListData);
    EXPECT_FALSE(result);
}

TEST_F(RanktableConfigTest, GetContainerIPAddress_EnvEmpty)
{
    EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "1.1.1.1");
    RanktableConfigManager manager;
    std::string ip = manager.GetContainerIPAddress();
    EXPECT_EQ(ip, "1.1.1.1");
}

TEST_F(RanktableConfigTest, CheckDeviceId_OutOfRange)
{
    RanktableConfigManager manager;
    EXPECT_FALSE(manager.CheckDeviceId("64"));
    EXPECT_FALSE(manager.CheckDeviceId("100"));
    EXPECT_FALSE(manager.CheckDeviceId("-1"));
    EXPECT_FALSE(manager.CheckDeviceId("-10"));
}

TEST_F(RanktableConfigTest, CheckDeviceId_InvalidArgument)
{
    RanktableConfigManager manager;
    EXPECT_FALSE(manager.CheckDeviceId("abc"));
    EXPECT_FALSE(manager.CheckDeviceId("123abc"));
    EXPECT_FALSE(manager.CheckDeviceId(""));
}

TEST_F(RanktableConfigTest, CheckDeviceId_OutOfRangeException)
{
    RanktableConfigManager manager;
    EXPECT_FALSE(manager.CheckDeviceId("4294967296"));
    EXPECT_FALSE(manager.CheckDeviceId("9999999999"));
    EXPECT_FALSE(manager.CheckDeviceId("-2147483649"));
}

TEST_F(RanktableConfigTest, CheckDeviceIp_InvalidIP)
{
    RanktableConfigManager manager;
    EXPECT_FALSE(manager.CheckDeviceIp("256.0.0.1"));
    EXPECT_FALSE(manager.CheckDeviceIp("192.168.1"));
    EXPECT_FALSE(manager.CheckDeviceIp("192.168.1.1.1"));
    EXPECT_FALSE(manager.CheckDeviceIp("abc.def.ghi.jkl"));
    EXPECT_FALSE(manager.CheckDeviceIp(""));
}

TEST_F(RanktableConfigTest, CheckRankId_OutOfRange)
{
    RanktableConfigManager manager;
    EXPECT_FALSE(manager.CheckRankId("512"));
    EXPECT_FALSE(manager.CheckRankId("1000"));
    EXPECT_FALSE(manager.CheckRankId("-1"));
    EXPECT_FALSE(manager.CheckRankId("-10"));
}

TEST_F(RanktableConfigTest, CheckRankId_InvalidArgument)
{
    RanktableConfigManager manager;
    EXPECT_FALSE(manager.CheckRankId("abc"));
    EXPECT_FALSE(manager.CheckRankId(""));
}

TEST_F(RanktableConfigTest, CheckRankId_OutOfRangeException)
{
    RanktableConfigManager manager;
    EXPECT_FALSE(manager.CheckRankId("4294967296"));
    EXPECT_FALSE(manager.CheckRankId("9999999999"));
    EXPECT_FALSE(manager.CheckRankId("-2147483649"));
}

} // namespace mindie_llm
