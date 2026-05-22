/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
#include <string>
#include "mockcpp/mockcpp.hpp"
#define private public
#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "grpc_handler.h"
#include "common_util.h"
#include "grpc_communication_mng.h"
#include "env_util.h"
#include "mock_util.h"
#include "single_req_vllm_openai_infer_interface.h"

using namespace mindie_llm;

MOCKER_CPP_OVERLOAD_EQ(ModelDeployConfig)
MOCKER_CPP_OVERLOAD_EQ(ScheduleConfig)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(BackendConfig)

namespace mindie_llm {

class GrpcHandlerTest : public testing::Test {
protected:
    void SetUp()
    {
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_grpc.json");
        MockAllConfig();
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

    ServerConfig serverConfig_;
    BackendConfig backendConfig_;
    ModelDeployConfig modelDeployConfig_;
    ScheduleConfig scheduleConfig_;

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

    void MockAllConfig()
    {
        MockServerConfig();
        MockBackendConfig();
        MockModelDeployConfig();
        MockScheduleConfig();
    }

    void MockServerConfig()
    {
        serverConfig_.allowAllZeroIpListening = false;
        serverConfig_.httpsEnabled = false;
        serverConfig_.ipAddress = "127.0.0.1";
        serverConfig_.managementIpAddress = "127.0.0.2";
        serverConfig_.port = 1025;
        serverConfig_.managementPort = 1026;
        serverConfig_.metricsPort = 1027;
        serverConfig_.maxLinkNum = 1000;
        serverConfig_.fullTextEnabled = false;
        serverConfig_.tlsCaPath = "../../config_manager/conf/cert/";
        serverConfig_.tlsCaFile = {"ca.pem"};
        serverConfig_.tlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.tlsPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.tlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.tlsCrlFiles = {"server_crl.pem"};
        serverConfig_.managementTlsCaFile = {"management_ca.pem"};
        serverConfig_.managementTlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.managementTlsPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.managementTlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.managementTlsCrlFiles = {"server_crl.pem"};
        serverConfig_.inferMode = "standard";
        serverConfig_.interCommTLSEnabled = true;
        serverConfig_.interCommPort = 1121;
        serverConfig_.interCommTlsCaPath = "../../config_manager/conf/cert/";
        serverConfig_.interCommTlsCaFiles = {"ca.pem"};
        serverConfig_.interCommTlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.interCommPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.interCommTlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.interCommTlsCrlFiles = {"server_crl.pem"};
        serverConfig_.tokenTimeout = 5;
        serverConfig_.e2eTimeout = 5;
        serverConfig_.distDPServerEnabled = false;
        MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig_));
    }
    
    void MockBackendConfig()
    {
        backendConfig_.backendName = "mindieservice_llm_engine";
        backendConfig_.modelInstanceNumber = 2;
        backendConfig_.npuDeviceIds = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};
        backendConfig_.tokenizerProcessNumber = 2;
        backendConfig_.multiNodesInferEnabled = true;
        backendConfig_.multiNodesInferPort = 1120;
        backendConfig_.interNodeTLSEnabled = false;
        backendConfig_.interNodeTlsCaPath = "../../config_manager/conf/";
        backendConfig_.interNodeTlsCaFiles = "ca.pem";
        backendConfig_.interNodeTlsCert = "../../config_manager/conf/certs/server.pem";
        backendConfig_.interNodeTlsPk = "../../config_manager/conf/server.key.pem";
        backendConfig_.interNodeTlsCrlPath = "../../config_manager/conf/certs/";
        backendConfig_.interNodeTlsCrlFiles = "server_crl.pem";
        backendConfig_.interNodeTlsCaFilesVec = {"ca.pem"};
        backendConfig_.interNodeTlsCrlFilesVec = {"ca.pem"};

        MOCKER_CPP(GetBackendConfig, const BackendConfig& (*)())
            .stubs()
            .will(returnValue(backendConfig_));
    }

    void MockModelDeployConfig()
    {
        modelDeployConfig_.modelInstanceType = "StandardMock";
        modelDeployConfig_.modelName = "llama_65b";
        modelDeployConfig_.modelWeightPath = "../../config_manager/conf";
        modelDeployConfig_.worldSize = 8;
        modelDeployConfig_.npuDeviceIds = {0, 1, 2, 3, 4, 5, 6, 7};
        modelDeployConfig_.npuMemSize = -1;
        modelDeployConfig_.cpuMemSize = 5;
        modelDeployConfig_.backendType = "atb";
        modelDeployConfig_.trustRemoteCode = false;
        modelDeployConfig_.maxSeqLen = 2560;
        modelDeployConfig_.maxInputTokenLen = 2048;
        modelDeployConfig_.truncation = false;
        modelDeployConfig_.loraModules["llama_65b"] = "../../config_manager/conf";

        std::vector<ModelDeployConfig> modelConfig = {modelDeployConfig_};

        MOCKER_CPP(GetModelDeployConfig, const std::vector<ModelDeployConfig> & (*)())
            .stubs()
            .will(returnValue(modelConfig));
    }

    void MockScheduleConfig()
    {
        
        scheduleConfig_.templateType = "Standard";
        scheduleConfig_.templateName = "Standard_LLM";
        scheduleConfig_.cacheBlockSize = 128;
        scheduleConfig_.maxPrefillBatchSize = 50;
        scheduleConfig_.maxPrefillTokens = 8192;
        scheduleConfig_.prefillTimeMsPerReq = 150;
        scheduleConfig_.prefillPolicyType = 0;
        scheduleConfig_.bufferResponseEnabled = false;
        scheduleConfig_.decodeTimeMsPerReq = 50;
        scheduleConfig_.decodePolicyType = 0;
        scheduleConfig_.policyType = 0;
        scheduleConfig_.enableSplit = true;
        scheduleConfig_.splitType = true;
        scheduleConfig_.splitStartType = true;
        scheduleConfig_.splitChunkTokens = 1;
        scheduleConfig_.splitStartBatchSize = 100;
        scheduleConfig_.enablePrefixCache = false;
        scheduleConfig_.maxBatchSize = 200;
        scheduleConfig_.maxIterTimes = 512;
        scheduleConfig_.maxPreemptCount = 0;
        scheduleConfig_.supportSelectBatch = true;
        scheduleConfig_.maxQueueDelayMicroseconds = 5000;
        scheduleConfig_.decodeExpectedTime = 5;
        scheduleConfig_.prefillExpectedTime = 5;
        scheduleConfig_.stageSelectPolicy = 1;
        scheduleConfig_.dynamicBatchSizeEnable = true;
        scheduleConfig_.maxNumPartialPrefills = 5;
        scheduleConfig_.maxLongPartialPrefills = 5;
        scheduleConfig_.longPrefillTokenThreshold = 5;
        MOCKER_CPP(GetScheduleConfig, const ScheduleConfig& (*)())
            .stubs()
            .will(returnValue(scheduleConfig_));
    }
};

TEST_F(GrpcHandlerTest, GetInstance) { EXPECT_TRUE(&GrpcHandler::GetInstance() != nullptr); }

using DecodeRequestHandler = std::function<void(const prefillAndDecodeCommunication::DecodeParameters& request,
                                                prefillAndDecodeCommunication::DecodeRequestResponse& response)>;

using KVReleaseHandler = std::function<void(const std::string& requestID)>;

TEST_F(GrpcHandlerTest, InitDmiBusiness)
{
    MOCKER_CPP(&GrpcCommunicationMng::RegisterKvReleaseHandler, bool (*)(KVReleaseHandler))
    .stubs()
    .will(returnValue(false));
    EXPECT_FALSE(GrpcHandler::GetInstance().InitDmiBusiness());
    MOCKER_CPP(&GrpcCommunicationMng::RegisterDecodeRequestHandler, bool (*)(DecodeRequestHandler))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(GrpcHandler::GetInstance().InitDmiBusiness());
    GlobalMockObject::verify();
    MockAllConfig();
    EXPECT_TRUE(GrpcHandler::GetInstance().InitDmiBusiness());
    prefillAndDecodeCommunication::DecodeParameters para;
    prefillAndDecodeCommunication::DecodeRequestResponse response;
    GrpcCommunicationMng::GetInstance().kvReleaseHandler_("test");
    void DecodeProcess(prefillAndDecodeCommunication::DecodeRequestResponse &response) noexcept;
    MOCKER_CPP(&SingleReqVllmOpenAiInferInterface::DecodeProcess,
        void (*)(prefillAndDecodeCommunication::DecodeRequestResponse&))
        .stubs();
    for (int i = 0; i <= 10; i++) {
        try {
            para.set_msgtype(i);
            GrpcCommunicationMng::GetInstance().decodeRequestHandler_(para, response);
        } catch (const std::exception& e) {
            std::cerr << "Exception for msgtype = " << i << ": " << e.what() << std::endl;
        }
    }
    EXPECT_TRUE(response.isvaliddecodeparameters());
}

TEST_F(GrpcHandlerTest, InitGrpcService)
{
    MOCKER_CPP(&GrpcCommunicationMng::Init, bool (*)(bool, const std::string&, const std::string&))
    .stubs()
    .will(returnValue(false));
    EXPECT_FALSE(GrpcHandler::GetInstance().InitGrpcService());
    GlobalMockObject::verify();
    MockAllConfig();
    MOCKER_CPP(&GrpcCommunicationMng::Init, bool (*)(bool, const std::string&, const std::string&))
    .stubs()
    .will(returnValue(true));
    EXPECT_TRUE(GrpcHandler::GetInstance().InitGrpcService());
    EXPECT_TRUE(GrpcHandler::GetInstance().InitDmiBusiness());
}

} // namespace llm