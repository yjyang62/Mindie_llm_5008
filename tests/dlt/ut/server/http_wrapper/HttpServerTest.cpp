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
#include <sys/socket.h>
#include <netinet/in.h>
#include <libgen.h>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#define private public
#include "memory_utils.h"
#include "http_server.h"
#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "base_config_manager.h"
#include "mock_util.h"
#include "env_util.h"
#include "https_server_helper.h"
#include "http_handler.h"

using namespace mindie_llm;

MOCKER_CPP_OVERLOAD_EQ(ServerConfig)

class HttpServerTest : public testing::Test {
protected:
    void SetUp() override
    {
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_http.json");

        serverConfig_.allowAllZeroIpListening = true;
        serverConfig_.httpsEnabled = false;
        serverConfig_.ipAddress = "127.0.0.1";
        serverConfig_.managementIpAddress = "127.0.0.2";
        serverConfig_.port = 8081;
        serverConfig_.managementPort = 8082;
        serverConfig_.metricsPort = 8083;
        MOCKER_CPP(&HttpHandler::BusinessInitialize, int (*)(HttpsServerHelper&))
        .stubs()
        .will(returnValue(0));
        MOCKER_CPP(&HttpHandler::ManagementInitialize, int (*)(HttpsServerHelper&))
        .stubs()
        .will(returnValue(0));
        MOCKER_CPP(&HttpHandler::InitializeMetricsResource, void (*)(HttpsServerHelper&))
        .stubs();
        MOCKER_CPP(&HttpsServerHelper::listen, bool (*)(const std::string&, int, int))
        .stubs()
        .will(returnValue(true));
        MOCKER_CPP(&HttpsServerHelper::is_running, bool (*)())
        .stubs()
        .will(returnValue(true));
        MOCKER_CPP(&HttpsServerHelper::wait_until_ready, void (*)())
        .stubs();
    }

    void TearDown() override
    {
        EXPECT_EQ(HttpServer::HttpServerDeInit(), 0);
        EnvUtil::GetInstance().ClearEnvVar("RANK_TABLE_FILE");
        EnvUtil::GetInstance().ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
        EnvUtil::GetInstance().ClearEnvVar("MIES_CONTAINER_IP");
        EnvUtil::GetInstance().ClearEnvVar("HOST_IP");
        GlobalMockObject::verify();
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

        std::unique_ptr<char, decltype(&std::free)> temp(strdup(buffer), &std::free);
        
        if (temp == nullptr) {
            std::cerr << "Memory allocation failed" << std::endl;
            return "";
        }

        char* parent = dirname(temp.get());
        std::string result(parent);
        return result;
    }

    ServerConfig serverConfig_;
};


TEST_F(HttpServerTest, HttpServerInit_AllowAllZeroIp)
{
    serverConfig_.allowAllZeroIpListening = false;
    serverConfig_.ipAddress = "0.0.0.0";
    serverConfig_.managementIpAddress = "0.0.0.0";
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    EXPECT_EQ(HttpServer::HttpServerInit(), 1);
}

TEST_F(HttpServerTest, HttpServerInit_IsPortUsed)
{
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    MOCKER_CPP(bind, int (*)(int, const struct sockaddr*, socklen_t))
        .stubs()
        .will(returnValue(-1));
    EXPECT_EQ(HttpServer::HttpServerInit(), 1);
    MOCKER_CPP(memset_s, int (*)(void*, size_t, int, size_t))
        .stubs()
        .will(returnValue(-1));
    EXPECT_EQ(HttpServer::HttpServerInit(), 1);
    MOCKER_CPP(socket, int (*)(int, int, int))
        .stubs()
        .will(returnValue(-1));
    EXPECT_EQ(HttpServer::HttpServerInit(), 1);
}

TEST_F(HttpServerTest, HttpServerInit_Condition1)
{
    serverConfig_.ipAddress = "127.0.0.1";
    serverConfig_.managementIpAddress = "127.0.0.1";
    serverConfig_.port = 8081;
    serverConfig_.managementPort = 8081;
    serverConfig_.metricsPort = 8081;
    
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    
    EXPECT_EQ(HttpServer::HttpServerInit(), 0);
}

TEST_F(HttpServerTest, HttpServerInit_Condition2)
{
    serverConfig_.ipAddress = "127.0.0.1";
    serverConfig_.managementIpAddress = "127.0.0.1";
    serverConfig_.port = 8081;
    serverConfig_.managementPort = 8081;
    serverConfig_.metricsPort = 8082;
    
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    
    EXPECT_EQ(HttpServer::HttpServerInit(), 0);
}

TEST_F(HttpServerTest, HttpServerInit_Condition3)
{
    serverConfig_.port = 8081;
    serverConfig_.managementPort = 8082;
    serverConfig_.metricsPort = 8082;
    
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    
    EXPECT_EQ(HttpServer::HttpServerInit(), 0);
}

TEST_F(HttpServerTest, HttpServerInit_Condition4)
{
    serverConfig_.port = 8081;
    serverConfig_.managementPort = 8082;
    serverConfig_.metricsPort = 8083;
    
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    
    EXPECT_EQ(HttpServer::HttpServerInit(), 0);
}

TEST_F(HttpServerTest, HttpServerInit_Condition5)
{
    serverConfig_.ipAddress = "127.0.0.1";
    serverConfig_.managementIpAddress = "127.0.0.2";
    serverConfig_.managementPort = 8082;
    serverConfig_.metricsPort = 8082;
    serverConfig_.port = 8082;
    
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    
    EXPECT_EQ(HttpServer::HttpServerInit(), 0);
}

TEST_F(HttpServerTest, HttpServerInit_Condition6)
{
    serverConfig_.ipAddress = "127.0.0.1";
    serverConfig_.managementIpAddress = "127.0.0.2";
    serverConfig_.managementPort = 8082;
    serverConfig_.metricsPort = 8083;
    serverConfig_.port = 8082;
    
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    
    EXPECT_EQ(HttpServer::HttpServerInit(), 0);
}

TEST_F(HttpServerTest, HttpServerInit_ElseCondition)
{
    serverConfig_.managementIpAddress = "127.0.0.1";
    serverConfig_.port = 8081;
    serverConfig_.managementPort = 8082;
    serverConfig_.metricsPort = 8081;
    
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
        .stubs()
        .will(returnValue(serverConfig_));
    EXPECT_EQ(HttpServer::HttpServerInit(), 1);
}