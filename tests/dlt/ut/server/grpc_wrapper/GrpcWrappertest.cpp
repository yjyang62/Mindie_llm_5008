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
#include <libgen.h>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#define private public
#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "grpc_handler.h"
#include "grpc_wrapper.h"
#include "env_util.h"
#include "mock_util.h"

using namespace mindie_llm;

MOCKER_CPP_OVERLOAD_EQ(ServerConfig)

namespace mindie_llm {
class GrpcWrapperMethodsTest : public testing::Test {
protected:
    void SetUp()
    {
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_grpc.json");
        MockServerConfig();
    }
    void TearDown()
    {
        EnvUtil::GetInstance().ClearEnvVar("RANK_TABLE_FILE");
        EnvUtil::GetInstance().ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
        EnvUtil::GetInstance().ClearEnvVar("MIES_CONTAINER_IP");
        EnvUtil::GetInstance().ClearEnvVar("HOST_IP");
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

    std::string GetParentDirectory()
    {
        char buffer[1024];
        try {
            return std::filesystem::current_path().string();
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error getting current directory: " << e.what() << std::endl;
            return "";
        }
    }
    
    ServerConfig serverConfig_;

    void MockServerConfig()
    {
        serverConfig_.ipAddress = "127.0.0.1";
        serverConfig_.interCommTLSEnabled = true;

        MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig_));
    }
};

TEST_F(GrpcWrapperMethodsTest, GetInstance)
{
    EXPECT_TRUE(&GrpcWrapper::GetInstance() != nullptr);
}

TEST_F(GrpcWrapperMethodsTest, StartAndStop)
{
    GrpcWrapper::GetInstance().Stop();
    MOCKER_CPP(&GrpcHandler::InitGrpcService, bool (*)())
    .stubs()
    .will(returnValue(false));
    EXPECT_EQ(GrpcWrapper::GetInstance().Start(), 1);
    MOCKER_CPP(&GrpcHandler::InitDmiBusiness, bool (*)())
    .stubs()
    .will(returnValue(false));
    EXPECT_EQ(GrpcWrapper::GetInstance().Start(), 1);
    GlobalMockObject::verify();
    MockServerConfig();
    EXPECT_EQ(GrpcWrapper::GetInstance().Start(), 0);
    EXPECT_TRUE(GrpcWrapper::GetInstance().started_.load());
    EXPECT_EQ(GrpcWrapper::GetInstance().Start(), 0);
    GrpcWrapper::GetInstance().Stop();
    EXPECT_FALSE(GrpcWrapper::GetInstance().started_.load());
}

} // namespace llm