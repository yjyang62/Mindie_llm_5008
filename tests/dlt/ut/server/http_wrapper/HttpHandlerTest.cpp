/**
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
#include "base_config_manager.h"
#include <unistd.h>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "infer_instances.h"
#include "src/server/endpoint/http_wrapper/http_handler.h"
#include "https_server_helper.h"
#include "mock_util.h"
#include "src/server/endpoint/http_wrapper/http_handler.cpp"
#include "src/server/endpoint/utils/http_rest_resource.h"
#include "env_util.h"

using namespace mindie_llm;

MOCKER_CPP_OVERLOAD_EQ(mindie_llm::ModelDeployConfig)
MOCKER_CPP_OVERLOAD_EQ(mindie_llm::ScheduleConfig)
MOCKER_CPP_OVERLOAD_EQ(std::vector<ModelDeployConfig>)
MOCKER_CPP_OVERLOAD_EQ(std::vector<LoraConfig>)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(LoraConfig)
MOCKER_CPP_OVERLOAD_EQ(BackendConfig)
MOCKER_CPP_OVERLOAD_EQ(RanktableParam)
MOCKER_CPP_OVERLOAD_EQ(EngineMetric)

namespace mindie_llm {

class HttpHandlerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        MockAllConfig();
        requestContext = std::make_shared<RequestContext>(request, response);
        inferRequestId = "";
        server = new HttpsServerHelper(false, [](SSL_CTX &) { return false; }, 1000);
        StopServiceOption::stopServiceFlag.store(false);
    }

    void TearDown() override
    {
        if (server) {
            delete server;
            server = nullptr;
        }
        InferModelCheckFalse();
        requestContext.reset();
        StopServiceOption::stopServiceFlag.store(true);
        keepAlive = false;
        GlobalMockObject::verify();
    }

    httplib::Request request;
    httplib::Response response;
    ReqCtxPtr requestContext;
    std::string inferRequestId;
    ServerConfig serverConfig_;
    BackendConfig backendConfig_;
    ModelDeployConfig modelDeployConfig_;
    std::vector<ModelDeployConfig> modelConfig_;
    ScheduleConfig scheduleConfig_;
    LoraConfig loraConfig_;
    RanktableParam ranktableParam_;
    HttpsServerHelper *server;
    HttpHandler handler;

    void MockAllConfig()
    {
        MockServerConfig();
        MockBackendConfig();
        MockModelDeployConfig();
        MockScheduleConfig();
        MockLoraConfig();
        MockRanktableParam();
    }

    RanktableParam InitSimpleRanktable()
    {
        RanktableParam param;

        param.serverCount = 2;
        param.isMaster = false;
        param.worldSize = 16;
        param.globalWorldSize = param.worldSize;

        ServerEle server1;
        server1.serverId = "127.0.0.1";
        server1.containerIp = "127.0.0.1";

        for (int i = 0; i < 8; ++i) {
            DeviceEle device;
            device.deviceId = std::to_string(i);
            device.deviceIp = "10.20.2." + std::to_string(i + 2);
            device.rankId = std::to_string(i);
            server1.device.push_back(device);
        }

        ServerEle server2;
        server2.serverId = "61.47.2.184";
        server2.containerIp = "61.47.2.184";

        for (int i = 0; i < 8; ++i) {
            DeviceEle device;
            device.deviceId = std::to_string(i);
            device.deviceIp = "10.20.2." + std::to_string(i + 26);
            device.rankId = std::to_string(i + 8);
            server2.device.push_back(device);
        }

        param.serverList = {server1, server2};
        param.local = server1;
        param.master = server1;
        param.slaves = {server2};

        return param;
    }

    void MockRanktableParam()
    {
        ranktableParam_ = InitSimpleRanktable();
        MOCKER_CPP(GetRanktableParam, const RanktableParam& (*)())
            .stubs()
            .will(returnValue(ranktableParam_));
    }

    void MockLoraConfig()
    {
        loraConfig_.baseModel = "llama_65b";
        loraConfig_.loraName = "llama_65b";
        loraConfig_.loraPath = "../../config_manager/conf";
        std::vector<LoraConfig> loraConfigs = {loraConfig_};
        MOCKER_CPP(GetLoraConfig, const std::vector<LoraConfig>& (*)())
            .stubs()
            .will(returnValue(loraConfigs));
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

        modelConfig_ = {modelDeployConfig_};

        MOCKER_CPP(GetModelDeployConfig, const std::vector<ModelDeployConfig> & (*)())
            .stubs()
            .will(returnValue(modelConfig_));
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

    void InferModelCheckTrue()
    {
        InferInstance::GetInstance()->started_.store(true);
        std::string configPath = GetParentDirectory() + "/../../config_manager/conf/config_http.json";
        std::map<std::string, std::string> ipInfo = {{"infer_mode", "standard"}};
        std::shared_ptr<LlmManagerV2> llmManager =
            std::make_shared<LlmManagerV2>(configPath, nullptr, nullptr, nullptr, nullptr, nullptr, ipInfo);
        InferInstance::GetInstance()->llmManagers_ = {llmManager};
    }

    void InferModelCheckFalse()
    {
        InferInstance::GetInstance()->llmManagers_= {};
        InferInstance::GetInstance()->started_.store(false);
    }
};

TEST_F(HttpHandlerTest, BusinessInitialize)
{
    auto ret = handler.BusinessInitialize(*server);
    EXPECT_EQ(ret, 0);
}

TEST_F(HttpHandlerTest, ManagementInitialize)
{
    GlobalMockObject::verify();
    serverConfig_.inferMode = "dmi";
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig_));
    MockBackendConfig();
    MockModelDeployConfig();
    MockScheduleConfig();
    MockLoraConfig();
    MockRanktableParam();
    MOCKER_CPP(&DmiRole::RunThread, void (*)()).stubs();
    EXPECT_EQ(handler.ManagementInitialize(*server), 0);
}

using BatchMap = std::map<std::string, uint64_t>&;

TEST_F(HttpHandlerTest, JudgeRestProcess)
{
    InferModelCheckTrue();
    EngineMetric engineMetric;
    auto stubs = MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs();
    stubs.will(returnValue(engineMetric));
    EXPECT_FALSE(handler.JudgeRestProcess());
    engineMetric.schedulerInfo.reqsInfo.waitingRequestNum_ = 1;
    stubs.then(returnValue(engineMetric));
    EXPECT_TRUE(handler.JudgeRestProcess());
    InferModelCheckFalse();
    EXPECT_FALSE(handler.JudgeRestProcess());
}

TEST_F(HttpHandlerTest, GetAvalSlotNum_Success)
{
    uint64_t availableSlots = 0;
    Status status = GetAvalSlotNum(availableSlots);
    
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(availableSlots, 0);

    InferModelCheckTrue();
    uint64_t availableSlots2 = 0;
    EngineMetric engineMetric;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    Status status2 = GetAvalSlotNum(availableSlots2);
    EXPECT_TRUE(status2.IsOk());
    EXPECT_EQ(availableSlots2, 200);
}

TEST_F(HttpHandlerTest, TgiInferType)
{
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    requestContext->msgBody = R"({"inputs": 1})";
    int result = handler.HandlePostGenerate(requestContext);
    EXPECT_EQ(result, 1);
}

TEST_F(HttpHandlerTest, InvalidJson)
{
    MOCKER(JsonParse::GetInferTypeFromJsonStr)
            .stubs()
            .will(returnValue(1));
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    int result = handler.HandlePostGenerate(requestContext);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::UnprocessableContent_422);
}

TEST_F(HttpHandlerTest, HandlePostGenerate)
{
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    // case 1: parse failed
    MOCKER(JsonParse::GetInferTypeFromJsonStr)
        .stubs()
        .will(returnValue(1)); // return error
    EXPECT_EQ(handler.HandlePostGenerate(requestContext), false);
}

int MockGetInferTypeFromJsonStrTgi(const std::string &jsonStr, uint16_t &inferType)
{
    inferType = MSG_TYPE_TGI;
    return EP_OK;
}

int MockGetInferTypeFromJsonStrVllm(const std::string &jsonStr, uint16_t &inferType)
{
    inferType = MSG_TYPE_VLLM;
    return EP_OK;
}

TEST_F(HttpHandlerTest, HandlePostGenerateTgi)
{
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    // case 2: TGI type
    MOCKER_CPP(JsonParse::GetInferTypeFromJsonStr, uint32_t(*)(const std::string&, uint16_t&))
        .stubs().will(invoke(MockGetInferTypeFromJsonStrTgi));
    EXPECT_EQ(handler.HandlePostGenerate(requestContext), true);
}

TEST_F(HttpHandlerTest, HandlePostGenerateVllm)
{
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    // case 3: VLLM type
    MOCKER_CPP(JsonParse::GetInferTypeFromJsonStr, uint32_t(*)(const std::string&, uint16_t&))
        .stubs().will(invoke(MockGetInferTypeFromJsonStrVllm));
    EXPECT_EQ(handler.HandlePostGenerate(requestContext), true);
}

TEST_F(HttpHandlerTest, CanDmiRoleReqProcess)
{
    EXPECT_FALSE(CanDmiRoleReqProcess());
    GlobalMockObject::verify();
    serverConfig_.inferMode = "dmi";
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig_));
    InferInstance::GetInstance()->SetPDRoleStatus(PDRoleStatus::SWITCHING);
    EXPECT_FALSE(CanDmiRoleReqProcess());
    InferInstance::GetInstance()->SetPDRoleStatus(PDRoleStatus::READY);
    EXPECT_TRUE(CanDmiRoleReqProcess());
}

TEST_F(HttpHandlerTest, HandleGetSlotCount)
{
    std::string modelName = "llama_65b";
    MOCKER_CPP(GetUriParameters, std::string (*)(const httplib::Request &, uint32_t))
            .stubs()
            .will(returnValue(modelName));
    MOCKER(GetAvalSlotNum)
            .stubs()
            .will(returnValue(Status(Error::Code::ERROR, "Failed")))
            .then(returnValue(Status(Error::Code::OK, "Success")));
    MOCKER(GetRemainBlockNum)
            .stubs()
            .will(returnValue(Status(Error::Code::ERROR, "Failed")))
            .then(returnValue(Status(Error::Code::OK, "Success")));
    handler.HandleGetSlotCount(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::InternalServerError_500);
    handler.HandleGetSlotCount(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::InternalServerError_500);
    EXPECT_EQ(handler.HandleGetSlotCount(requestContext), 0);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(HttpHandlerTest, HandleTokenizer_NotInputs)
{
    MOCKER(JsonParse::GetContextJsonBody)
            .stubs()
            .will(returnValue(false))
            .then(returnValue(true));
    handler.HandleTokenizer(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::UnprocessableContent_422);
    handler.HandleTokenizer(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::UnprocessableContent_422);
    GlobalMockObject::verify();
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig_));
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::UnprocessableContent_422);
    requestContext->msgBody = R"({"inputs": "test"})";
    MOCKER_CPP(&TokenizerProcessPool::TikToken, Status (*)(const std::string &, int &, std::vector<std::string> &, bool))
            .stubs()
            .will(returnValue(Status(Error::Code::ERROR, "Failed")))
            .then(returnValue(Status(Error::Code::OK, "Success")));
    handler.HandleTokenizer(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::UnprocessableContent_422);
    handler.HandleTokenizer(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(HttpHandlerTest, HandleGetHealthStatus)
{
    GlobalMockObject::verify();
    serverConfig_.inferMode = "dmi";
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig_));
    handler.HandleGetHealthStatus(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::InternalServerError_500);
    InferModelCheckTrue();
    handler.HandleGetHealthStatus(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(HttpHandlerTest, HandleGetHealthStatus_LoraConfig)
{
    std::string modelName1 = "test";
    std::string modelName2 = "llama_65b";
    MOCKER_CPP(GetUriParameters, std::string (*)(const httplib::Request &, uint32_t))
            .stubs()
            .will(returnValue(modelName1))
            .then(returnValue(modelName2));
    LoraConfig config;
    EXPECT_FALSE(handler.GetRequestModelConfig(requestContext, config));
    EXPECT_TRUE(handler.GetRequestModelConfig(requestContext, config));
}

class RequestTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        MockServerConfig();
        MockScheduleConfig();
        StopServiceOption::stopServiceFlag.store(false);
        requestContext = std::make_shared<RequestContext>(request, response);
    }

    void TearDown() override
    {
        InferModelCheckFalse();
        StopServiceOption::stopServiceFlag.store(true);
        request.headers.clear();
        GlobalMockObject::verify();
    }

    void InferModelCheckTrue()
    {
        InferInstance::GetInstance()->started_.store(true);
        std::string configPath = GetParentDirectory() + "/../../config_manager/conf/config_http.json";
        std::map<std::string, std::string> ipInfo = {{"infer_mode", "standard"}};
        std::shared_ptr<LlmManagerV2> llmManager =
            std::make_shared<LlmManagerV2>(configPath, nullptr, nullptr, nullptr, nullptr, nullptr, ipInfo);
        InferInstance::GetInstance()->llmManagers_ = {llmManager};
    }

    void InferModelCheckFalse()
    {
        InferInstance::GetInstance()->llmManagers_= {};
        InferInstance::GetInstance()->started_.store(false);
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

ServerConfig serverConfig_;
ScheduleConfig scheduleConfig_;
httplib::Request request;
httplib::Response response;
ReqCtxPtr requestContext;
HttpHandler handler;
};

TEST_F(RequestTest, IsAllDMIHeadersExist)
{
    std::string reqError;
    EXPECT_FALSE(handler.IsAllDMIHeadersExist(request, reqError));
    EXPECT_EQ(reqError, "DMI request must have req-type, req-id, d-target headers");
    request.set_header("req-type", "test");
    request.set_header("req-id", "1");
    request.set_header("d-target", "target");
    EXPECT_TRUE(handler.IsAllDMIHeadersExist(request, reqError));
}

TEST_F(RequestTest, IsReqIdValid)
{
    request.set_header("req-id", "123abcABC");
    std::string reqError;
    EXPECT_TRUE(handler.IsReqIdValid(request, reqError));
    EXPECT_TRUE(reqError.empty());
}

TEST_F(RequestTest, IsReqTypeValid)
{
    request.set_header("req-id", "123g");
    std::string reqError;
    EXPECT_FALSE(handler.IsReqIdValid(request, reqError));
    EXPECT_FALSE(reqError.empty());
    EXPECT_NE(reqError.find("Invalid req-id"), std::string::npos);
    EXPECT_NE(reqError.find("123g"), std::string::npos);
}

TEST_F(RequestTest, IsDTargetValid)
{
    std::string exceedLengthReqId(1025, 'a');
    request.set_header("req-id", exceedLengthReqId);
    std::string reqError;
    EXPECT_FALSE(handler.IsReqIdValid(request, reqError));
    EXPECT_FALSE(reqError.empty());
    EXPECT_NE(reqError.find("length of req-id cannot exceed 1024"), std::string::npos);
}

TEST_F(RequestTest, IsRecomputeParamValid)
{
    request.set_header("is-recompute", "other");
    std::string reqError;
    EXPECT_FALSE(handler.IsRecomputeParamValid(request, reqError));
    request.headers.clear();
    request.set_header("is-recompute", "true");
    EXPECT_TRUE(handler.IsRecomputeParamValid(request, reqError));
}

TEST_F(RequestTest, CheckDMIReqValid)
{
    InferInstance::GetInstance()->SetPDRoleStatus(PDRoleStatus::UNKNOWN);
    EXPECT_FALSE(handler.CheckDMIReqValid(request, requestContext));
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::ServiceUnavailable_503);
    InferInstance::GetInstance()->SetPDRoleStatus(PDRoleStatus::READY);
    std::string model1 = "decode";
    std::string model2 = "prefill";
    InferInstance::GetInstance()->UpdatePDRole(model1);
    EXPECT_FALSE(handler.CheckDMIReqValid(request, requestContext));
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::BadRequest_400);
    InferInstance::GetInstance()->UpdatePDRole(model2);
    request.set_header("req-type", "prefill");
    request.set_header("req-id", "123");
    request.set_header("d-target", "127.0.0.1");
    request.set_header("is-recompute", "true");
    EXPECT_TRUE(handler.CheckDMIReqValid(request, requestContext));
}

TEST_F(RequestTest, SetJsonObj)
{
    InferModelCheckTrue();
    EngineMetric engineMetric;
    auto stubs = MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs();
    stubs.will(returnValue(engineMetric));
    OrderedJson jsonObj;
    handler.SetJsonObj(jsonObj);
    EXPECT_EQ(jsonObj["resource"]["waitingRequestNum"], 0);
}

TEST_F(RequestTest, RepeatedStop)
{
    MOCKER_CPP(&HttpHandler::JudgeRestProcess, bool (*)())
            .stubs()
            .will(returnValue(true))
            .then(returnValue(false));
    handler.StopService(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::OK_200);
    handler.StopService(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::InternalServerError_500);

}

TEST_F(RequestTest, HandleGeneralTGIPostGenerate_DecodeError)
{
    MOCKER(JsonParse::DecodeGeneralTGIStreamMode)
            .stubs()
            .will(returnValue(1));
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    handler.HandleGeneralTGIPostGenerate(request, requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::UnprocessableContent_422);
}

TEST_F(RequestTest, HandleGeneralTGIPostGenerate)
{
    MOCKER(JsonParse::DecodeGeneralTGIStreamMode)
            .stubs()
            .will(returnValue(0));
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    handler.HandleGeneralTGIPostGenerate(request, requestContext);
    EXPECT_EQ(requestContext->Res().status, -1);
}

TEST_F(RequestTest, HandleHttpMetrics_ERROR)
{
    handler.HandleHttpMetrics(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::InternalServerError_500);
    InferModelCheckTrue();
    EngineMetric engineMetric;
    auto stubs = MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs();
    stubs.will(returnValue(engineMetric));
    handler.HandleHttpMetrics(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::OK_200);
}

class StaticTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        requestContext = std::make_shared<RequestContext>(request, response);
    }

    void TearDown() override
    {
        InferModelCheckFalse();
        GlobalMockObject::verify();
    }

    void InferModelCheckTrue()
    {
        InferInstance::GetInstance()->started_.store(true);
        std::string configPath = GetParentDirectory() + "/../../config_manager/conf/config_http.json";
        std::map<std::string, std::string> ipInfo = {{"infer_mode", "standard"}};
        std::shared_ptr<LlmManagerV2> llmManager =
            std::make_shared<LlmManagerV2>(configPath, nullptr, nullptr, nullptr, nullptr, nullptr, ipInfo);
        InferInstance::GetInstance()->llmManagers_ = {llmManager};
    }

    void InferModelCheckFalse()
    {
        InferInstance::GetInstance()->llmManagers_= {};
        InferInstance::GetInstance()->started_.store(false);
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

httplib::Request request;
httplib::Response response;
ReqCtxPtr requestContext;
};

TEST_F(StaticTest, GetRemainBlockNum_Success)
{
    uint64_t actualRemainBlocks = 0;
    std::map<uint32_t, uint64_t> actualDpRemainBlocks;
    Status status = GetRemainBlockNum(actualRemainBlocks, &actualDpRemainBlocks);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(actualRemainBlocks, 0);
    EXPECT_EQ(actualDpRemainBlocks, (std::map<uint32_t, uint64_t>{}));
}

TEST_F(StaticTest, GetRemainBlockNum_WithoutDpRemainBlocksOut)
{
    InferModelCheckTrue();
    EngineMetric engineMetric;
    auto stubs = MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs();
    stubs.will(returnValue(engineMetric));
    uint64_t actualRemainBlocks = 0;
    Status status = GetRemainBlockNum(actualRemainBlocks);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(actualRemainBlocks, 0);
}

TEST_F(StaticTest, ReadyStatus)
{
    InferInstance::GetInstance()->SetPDRoleStatus(PDRoleStatus::READY);
    std::string statusString;
    ReqCtxPtr ctx = nullptr;
    bool result = ValidatePDRoleStatus(ctx, statusString);

    EXPECT_TRUE(result);
    EXPECT_EQ(statusString, "RoleReady");
}

TEST_F(StaticTest, CheckHealthAndStop_WhenStopped)
{
    HealthManager::UpdateHealth(false);
    EXPECT_TRUE(CheckHealthAndStop(requestContext));
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::ServiceUnavailable_503);
    StopServiceOption::stopServiceFlag.store(false);
    HealthManager::UpdateHealth(true);
    EXPECT_FALSE(CheckHealthAndStop(requestContext));
    StopServiceOption::stopServiceFlag.store(true);
    EXPECT_TRUE(CheckHealthAndStop(requestContext));
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::ServiceUnavailable_503);
    EXPECT_NE(response.body.find("The service has been stopped."), std::string::npos);
    HealthManager::UpdateHealth(false);
}

TEST_F(StaticTest, ReturnsPrefillWhenInputIsPrefill)
{
    InferReqType result = GetReqType("prefill", false);

    ASSERT_EQ(result, InferReqType::REQ_PREFILL);
}

TEST_F(StaticTest, ReturnsDefaultWhenInputIsEmpty)
{
    InferReqType result = GetReqType("", false);

    ASSERT_EQ(result, InferReqType::REQ_STAND_INFER);
}

TEST_F(StaticTest, ReturnsDefaultWhenInputIsPrefillUpperCase)
{
    InferReqType result = GetReqType("PREFILL", true);

    ASSERT_EQ(result, InferReqType::REQ_STAND_INFER);
}

TEST_F(StaticTest, EmptyWaitTime)
{
    uint32_t timeNum = 0;
    EXPECT_TRUE(CheckWaitTime("", timeNum, requestContext));
}

TEST_F(StaticTest, NumberTooLong)
{
    httplib::Request request;
    httplib::Response response;
    ReqCtxPtr requestContext = std::make_shared<RequestContext>(request, response);
    uint32_t timeNum = 0;
    EXPECT_FALSE(CheckWaitTime("-1234", timeNum, requestContext));
    EXPECT_EQ(timeNum, 0);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::NotFound_404);
    auto json = nlohmann::json::parse(requestContext->Res().body);
    EXPECT_EQ(json["error"], "Max wait time is " + std::to_string(CV_WAIT_TIME) +
              ", input is invalid or too long");
}

TEST_F(StaticTest, InvalidNumberFormat)
{
    httplib::Request request;
    httplib::Response response;
    ReqCtxPtr requestContext = std::make_shared<RequestContext>(request, response);
    uint32_t timeNum = 0;
    EXPECT_FALSE(CheckWaitTime("-abc", timeNum, requestContext));
    EXPECT_EQ(timeNum, 0);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::NotFound_404);
    auto json = nlohmann::json::parse(requestContext->Res().body);
    EXPECT_EQ(json["error"], "Max wait time is " + std::to_string(CV_WAIT_TIME) +
              ", input is invalid or too long");
}

TEST_F(StaticTest, ZeroValue)
{
    httplib::Request request;
    httplib::Response response;
    ReqCtxPtr requestContext = std::make_shared<RequestContext>(request, response);
    uint32_t timeNum = 0;
    EXPECT_FALSE(CheckWaitTime("-0", timeNum, requestContext));
    EXPECT_EQ(timeNum, 0);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::NotFound_404);

    auto json = nlohmann::json::parse(requestContext->Res().body);
    EXPECT_EQ(json["error"], "Wait time should be in range of [1, " +
              std::to_string(CV_WAIT_TIME) + "], input is not valid");
}

TEST_F(StaticTest, BoundaryValue1)
{
    httplib::Request request;
    httplib::Response response;
    ReqCtxPtr requestContext = std::make_shared<RequestContext>(request, response);
    uint32_t timeNum = 0;
    EXPECT_TRUE(CheckWaitTime("-1", timeNum, requestContext));
    EXPECT_EQ(timeNum, 1);
    EXPECT_EQ(requestContext->Res().status, -1);
}

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        MockAllConfig();
        requestContext = std::make_shared<RequestContext>(request, response);
    }

    void TearDown() override
    {
        requestContext.reset();
        GlobalMockObject::verify();
    }
 
    httplib::Request request;
    httplib::Response response;
    ReqCtxPtr requestContext;
    ServerConfig serverConfig_;
    ScheduleConfig scheduleConfig_;
    HttpHandler handler;

    void MockAllConfig()
    {
        MockServerConfig();
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

using StatusMap = const std::map<uint64_t, std::pair<std::string, bool>>&;

TEST_F(ConfigTest, HandleStatusV1_Dmi_200)
{
    std::map<uint64_t, std::pair<std::string, bool>> mixedMap = {
        {0, {"ok", true}},
        {1, {"error", false}},
        {2, {"ok", true}}
    };
    GlobalMockObject::verify();
    MOCKER_CPP(&DmiRole::GetRemoteNodeLinkStatus, StatusMap (*)())
        .stubs()
        .will(returnValue(mixedMap));
    MOCKER_CPP(&InferInstance::GetPDRoleStatus, PDRoleStatus (*)())
        .stubs()
        .will(returnValue(0));
    serverConfig_.inferMode = "dmi";
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig_));
    MockScheduleConfig();
    handler.HandleStatusV1(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(ConfigTest, HandleStatusV2_200)
{
    std::map<uint64_t, std::pair<std::string, bool>> mixedMap = {
        {0, {"ok", true}},
        {1, {"error", false}},
        {2, {"ok", true}}
    };
    MOCKER_CPP(&DmiRole::GetRemoteNodeLinkStatus, StatusMap (*)())
        .stubs()
        .will(returnValue(mixedMap));
    handler.HandleStatusV2(requestContext);
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(ConfigTest, HandleStatusV2_Dmi_Init)
{
    GlobalMockObject::verify();
    serverConfig_.inferMode = "dmi";
    MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig_));
    MOCKER_CPP(ValidatePDRoleStatus, bool (*)(const ReqCtxPtr&, std::string&))
            .stubs()
            .will(returnValue(false));
    handler.HandleStatusV2(requestContext);
    EXPECT_EQ(requestContext->Res().status, -1);
}

class DispatchInferTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        requestContext = std::make_shared<RequestContext>(request, response);
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }

httplib::Request request;
httplib::Response response;
ReqCtxPtr requestContext;
};

TEST_F(DispatchInferTest, StandInferRequest_REQ_STAND_INFER)
{
    std::shared_ptr<HttpReqHeadersOption> opt = std::make_shared<HttpReqHeadersOption>();
    opt->reqType = InferReqType::REQ_STAND_INFER;
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    std::string requestId = "test";
    MOCKER_CPP(&SingleReqInferInterfaceBase::GetRequestId, RequestIdNew (*)())
        .stubs()
        .will(returnValue(requestId));
    MOCKER_CPP(&InferInstance::ControlRequest, Status (*)(const RequestIdNew&, OperationV2))
        .stubs()
        .will(returnValue(Status(Error::Code::OK, "Success")));
    DispatchInfer(requestContext, opt, MSG_TYPE_TGI,
        [&](auto handler) {
            return nullptr;
        });
    EXPECT_EQ(requestContext->Res().status, -1);
}

TEST_F(DispatchInferTest, StandInferRequest_REQ_PREFILL)
{
    std::shared_ptr<HttpReqHeadersOption> opt = std::make_shared<HttpReqHeadersOption>();
    opt->reqType = InferReqType::REQ_PREFILL;
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    DispatchInfer(requestContext, opt, MSG_TYPE_TGI,
        [&](auto handler) {
            return nullptr;
        });
    EXPECT_EQ(requestContext->Res().status, -1);
}

TEST_F(DispatchInferTest, StandInferRequest_REQ_DECODE)
{
    std::shared_ptr<HttpReqHeadersOption> opt = std::make_shared<HttpReqHeadersOption>();
    opt->reqType = InferReqType::REQ_DECODE;
    MOCKER_CPP(&SingleReqInferInterfaceBase::Process, void (*)(RequestSPtr, const std::string&, const uint64_t&))
        .stubs();
    std::string requestId = "test";
    MOCKER_CPP(&SingleReqInferInterfaceBase::GetRequestId, RequestIdNew (*)())
        .stubs()
        .will(returnValue(requestId));
    DispatchInfer(requestContext, opt, MSG_TYPE_TGI,
        [&](auto handler) {
            return nullptr;
        });
    EXPECT_EQ(requestContext->Res().status, httplib::StatusCode::UnprocessableContent_422);
}

class DispatcherTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        requestContext = std::make_shared<RequestContext>(request, response);
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }

httplib::Request request;
httplib::Response response;
ReqCtxPtr requestContext;
};

TEST_F(DispatcherTest, DResultKeepAlive)
{
    EXPECT_EQ(dResultDispatcher, nullptr);
    HandleDResult(requestContext);
    EXPECT_NE(dResultDispatcher, nullptr);
    HandleDResult(requestContext);
    MOCKER_CPP(&DResultEventDispatcher::SendEvent, void (*)(const std::string &, bool, std::string))
        .stubs();
    MOCKER_CPP(memcpy_s, int (*)(PVOID dest, size_t destsz, CPVOID src, size_t count))
        .stubs()
        .will(returnValue(0))
        .then(returnValue(1))
        .then(returnValue(1))
        .then(returnValue(0));
    
    auto pastTime = boost::chrono::steady_clock::now() - boost::chrono::seconds(61);
    {
        std::unique_lock<std::mutex> lk(g_dResMutex);
        dResultDispatcher->lastTimestamp_ = pastTime;
    }
    
    EXPECT_FALSE(keepAlive.load());
    keepAlive.store(true);
    
    std::thread test_thread = std::thread([this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        keepAlive.store(false);
    });
    
    if (test_thread.joinable()) {
        test_thread.join();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::unique_lock<std::mutex> lk(g_dResMutex);
        dResultDispatcher.reset();
    }
    
    EXPECT_FALSE(keepAlive.load());
}

} // namespace mindie_llm