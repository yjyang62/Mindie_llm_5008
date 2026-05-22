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
#include <mockcpp/mockcpp.hpp>
#define private public
#include "infer_instances.h"
#include "infer_instances.cpp"
#include "config_manager.h"
#include "config_manager_impl.h"
#include "base_config_manager.h"
#include "mock_util.h"

using namespace mindie_llm;

MOCKER_CPP_OVERLOAD_EQ(mindie_llm::ModelDeployConfig)
MOCKER_CPP_OVERLOAD_EQ(mindie_llm::ScheduleConfig)
MOCKER_CPP_OVERLOAD_EQ(std::vector<ModelDeployConfig>)
MOCKER_CPP_OVERLOAD_EQ(std::vector<LoraConfig>)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(LoraConfig)
MOCKER_CPP_OVERLOAD_EQ(BackendConfig)
MOCKER_CPP_OVERLOAD_EQ(EngineMetric)
MOCKER_CPP_OVERLOAD_EQ(RanktableParam)

class InferInstanceTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        instance = InferInstance::GetInstance();
        configPath = GetParentDirectory() + "/../../config_manager/conf/config_http.json";
        MockAllConfig();
        MOCKER_CPP(&LlmManagerV2::Shutdown, void (*)()).stubs();
    }

    void TearDown() override
    {
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
    }

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

    ServerConfig serverConfig_;
    BackendConfig backendConfig_;
    ModelDeployConfig modelDeployConfig_;
    ScheduleConfig scheduleConfig_;
    LoraConfig loraConfig_;
    RanktableParam ranktableParam_;
    std::shared_ptr<InferInstance> instance;
    std::string configPath;

};

static inline Status Ok() { return Status(Error::Code::OK); }
static inline Status InvalidArg() { return Status(Error::Code::INVALID_ARG); }
static inline Status InternalErr() { return Status(Error::Code::ERROR); }

TEST_F(InferInstanceTest, InitFromEndpointCall_BackendConfigEmpty)
{
    GlobalMockObject::verify();
    std::vector<ModelDeployConfig> modelConfig = {};
    MOCKER_CPP(GetModelDeployConfig, const std::vector<ModelDeployConfig> & (*)())
        .stubs()
        .will(returnValue(modelConfig));
    MockServerConfig();
    MockBackendConfig();
    MockScheduleConfig();
    MockLoraConfig();
    MockRanktableParam();
    EXPECT_EQ(instance->InitFromEndpointCall(configPath).StatusCode(), Error::Code::ERROR);
}

TEST_F(InferInstanceTest, InitFromEndpointCall_InstanceNumber_ERROR)
{
    GlobalMockObject::verify();
    backendConfig_.modelInstanceNumber = 1;
    MOCKER_CPP(GetBackendConfig, const BackendConfig& (*)())
            .stubs()
            .will(returnValue(backendConfig_));
    MockServerConfig();
    MockModelDeployConfig();
    MockScheduleConfig();
    MockLoraConfig();
    MockRanktableParam();
    EXPECT_EQ(instance->InitFromEndpointCall(configPath).StatusCode(), Error::Code::ERROR);
}

using ModelMap = std::map<std::string, std::string>;

TEST_F(InferInstanceTest, InitFromEndpointCall_InitSingleInferInstance_ERROR)
{
    MOCKER_CPP(&InferInstance::InitSingleInferInstance, Status (*)(ModelMap, uint32_t))
            .stubs()
            .will(returnValue(InternalErr()));
    EXPECT_EQ(instance->InitFromEndpointCall(configPath).StatusCode(), Error::Code::ERROR);
}

TEST_F(InferInstanceTest, InitFromEndpointCall_InitSingleInferInstance_OK)
{
    MOCKER_CPP(&InferInstance::InitSingleInferInstance, Status (*)(ModelMap, uint32_t))
            .stubs()
            .will(returnValue(Ok()));
    EXPECT_EQ(instance->InitFromEndpointCall(configPath).StatusCode(), Error::Code::OK);
}

using LlmManagerInitFunc = Status (LlmManagerV2::*)(uint32_t, std::set<size_t>);

TEST_F(InferInstanceTest, InitSingleInferInstance_Init_ERROR)
{
    std::map<std::string, std::string> modelConfig =  {
        {"configPath", configPath},
        {"npuDeviceIds", SerializeSet(backendConfig_.npuDeviceIds[0])},
        {"inferMode", "standard"}
    };
    MOCKER_CPP(static_cast<LlmManagerInitFunc>(&LlmManagerV2::Init), Status (*)(uint32_t, std::set<size_t>))
        .stubs()
        .will(returnValue(InternalErr()));
    EXPECT_EQ(instance->InitSingleInferInstance(modelConfig, 1).StatusCode(), Error::Code::ERROR);
}

TEST_F(InferInstanceTest, InitSingleInferInstance_Init_OK)
{
    std::map<std::string, std::string> modelConfig =  {
        {"configPath", configPath},
        {"npuDeviceIds", SerializeSet(backendConfig_.npuDeviceIds[0])},
        {"inferMode", "standard"}
    };
    MOCKER_CPP(static_cast<LlmManagerInitFunc>(&LlmManagerV2::Init), Status (*)(uint32_t, std::set<size_t>))
        .stubs()
        .will(returnValue(Ok()));
    EXPECT_EQ(instance->InitSingleInferInstance(modelConfig, 1).StatusCode(), Error::Code::OK);
}

TEST_F(InferInstanceTest, InitSingleInferInstance_Dmi)
{
    GlobalMockObject::verify();
    std::map<std::string, std::string> modelConfig =  {
        {"configPath", configPath},
        {"npuDeviceIds", SerializeSet(backendConfig_.npuDeviceIds[0])},
        {"inferMode", "dmi"}
    };
    backendConfig_.multiNodesInferEnabled = false;
    MOCKER_CPP(GetBackendConfig, const BackendConfig& (*)())
            .stubs()
            .will(returnValue(backendConfig_));
    MockServerConfig();
    MockModelDeployConfig();
    MockScheduleConfig();
    MockLoraConfig();
    MockRanktableParam();
    EXPECT_EQ(instance->InitSingleInferInstance(modelConfig, 1).StatusCode(), Error::Code::OK);
}

TEST_F(InferInstanceTest, Process)
{
    RequestSPtr request = std::make_shared<Request>(RequestIdNew("test"));
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.remainBlocks_ = 1;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    MOCKER_CPP(&LlmManagerV2::AddRequest, Status (*)(RequestSPtr))
        .stubs()
        .will(returnValue(InternalErr()));
    EXPECT_EQ(instance->Process(request).StatusCode(), Error::Code::ERROR);
}

TEST_F(InferInstanceTest, ControlRequest)
{
    std::string reqId = "test";
    MOCKER_CPP(&LlmManagerV2::ControlRequest, Status (*)(const RequestIdNew&, OperationV2))
        .stubs()
        .will(returnValue(Ok()))
        .then(returnValue(InternalErr()));
    EXPECT_EQ(instance->ControlRequest(reqId, OperationV2::STOP).StatusCode(), Error::Code::OK);
    EXPECT_EQ(instance->ControlRequest(reqId, OperationV2::RELEASE_KV).StatusCode(), Error::Code::ERROR);
}

TEST_F(InferInstanceTest, GetProcessingRequest)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.waitingRequestNum_ = 1;
    engineMetric.schedulerInfo.reqsInfo.runningRequestNum_ = 2;
    engineMetric.schedulerInfo.reqsInfo.swappedRequestNum_ = 3;
    uint64_t total = 0;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetProcessingRequest(total).StatusCode(), Error::Code::OK);
    EXPECT_EQ(total, 6);
}

TEST_F(InferInstanceTest, GetWaitingRequest)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.waitingRequestNum_ = 2;
    uint64_t total = 0;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetWaitingRequest(total).StatusCode(), Error::Code::OK);
    EXPECT_EQ(total, 2);
}

TEST_F(InferInstanceTest, GetRunningRequest)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.runningRequestNum_ = 1;
    uint64_t total = 0;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetRunningRequest(total).StatusCode(), Error::Code::OK);
    EXPECT_EQ(total, 1);
}

TEST_F(InferInstanceTest, GetSwappedRequest)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.swappedRequestNum_ = 1;
    uint64_t total = 1;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetSwappedRequest(total).StatusCode(), Error::Code::OK);
    EXPECT_EQ(total, 1);
}

TEST_F(InferInstanceTest, GetCacheBlockNums)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.blockInfo.freeNpuBlockNum_ = 1;
    engineMetric.schedulerInfo.blockInfo.freeCpuBlockNum_ = 2;
    engineMetric.schedulerInfo.blockInfo.totalNpuBlockNum_ = 3;
    engineMetric.schedulerInfo.blockInfo.totalCpuBlockNum_ = 4;
    uint64_t freeNpuBlockNums = 0;
    uint64_t freeCpuBlockNums = 0;
    uint64_t totalNpuBlockNums = 0;
    uint64_t totalCpuBlockNums = 0;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetCacheBlockNums(freeNpuBlockNums, freeCpuBlockNums, totalNpuBlockNums, totalCpuBlockNums).StatusCode(), Error::Code::OK);
    EXPECT_EQ(freeNpuBlockNums, 1);
    EXPECT_EQ(freeCpuBlockNums, 2);
    EXPECT_EQ(totalNpuBlockNums, 3);
    EXPECT_EQ(totalCpuBlockNums, 4);
}

TEST_F(InferInstanceTest, GetRadixMatchNums)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.allRadixMatchNum_ = 1;
    engineMetric.schedulerInfo.reqsInfo.npuRadixMatchHitNum_ = 2;
    uint64_t allRadixMatchNum = 0;
    uint64_t npuRadixMatchHitNum = 0;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetRadixMatchNums(allRadixMatchNum, npuRadixMatchHitNum).StatusCode(), Error::Code::OK);
    EXPECT_EQ(allRadixMatchNum, 1);
    EXPECT_EQ(npuRadixMatchHitNum, 2);
}

TEST_F(InferInstanceTest, GetCumulativePreemptCount)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.cumulativePreemptCount_ = 1;
    uint64_t cumulativePreemptCount = 0;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetCumulativePreemptCount(cumulativePreemptCount).StatusCode(), Error::Code::OK);
    EXPECT_EQ(cumulativePreemptCount, 1);
}

TEST_F(InferInstanceTest, GetThroughput)
{
    EngineMetric engineMetric;
    engineMetric.prefillThroughput_ = 0.1f;
    engineMetric.decodeThroughput_ = 0.2f;
    float prefillThroughput = 0.0f;
    float decodeThroughput = 0.0f;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetThroughput(prefillThroughput, decodeThroughput).StatusCode(), Error::Code::OK);
    EXPECT_EQ(prefillThroughput, 0.1f);
    EXPECT_EQ(decodeThroughput, 0.2f);
}

TEST_F(InferInstanceTest, GetRequestBlockQuotas)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.remainBlocks_ = 1;
    engineMetric.schedulerInfo.reqsInfo.remainPrefillSlots_ = 2;
    engineMetric.schedulerInfo.reqsInfo.remainPrefillTokens_ = 3;
    std::map<uint32_t, uint64_t> dpRemainBlocks = {};
    uint64_t remainBlocks = 0;
    uint64_t remainPrefillSlots = 0;
    uint64_t remainPrefillTokens = 0;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    EXPECT_EQ(instance->GetRequestBlockQuotas(remainBlocks, remainPrefillSlots, remainPrefillTokens, dpRemainBlocks).StatusCode(), Error::Code::OK);
    EXPECT_EQ(remainBlocks, 1);
    EXPECT_EQ(remainPrefillSlots, 2);
    EXPECT_EQ(remainPrefillTokens, 3);
}

TEST_F(InferInstanceTest, GetNodeStatus)
{
    std::string key = "test";
    std::map<std::string, NodeHealthStatus> slaveStatus = {{key, NodeHealthStatus::READY}};
    EXPECT_EQ(instance->GetNodeStatus(slaveStatus).StatusCode(), Error::Code::OK);
    EXPECT_EQ(slaveStatus.size(), 0);
}

TEST_F(InferInstanceTest, ProcessFailLinkIp)
{
    RequestSPtr request = std::make_shared<Request>(RequestIdNew("test"));
    GlobalIpInfo globalIpInfo;
    EXPECT_TRUE(ProcessFailLinkIp(request, globalIpInfo));
    request->failedLinkInfos.push_back({1, 2});
    EXPECT_TRUE(ProcessFailLinkIp(request, globalIpInfo));
    EXPECT_EQ(globalIpInfo.failLinkInstanceIDAndReason[1], 2);
}

TEST_F(InferInstanceTest, AddAttributeToRequest)
{
    RequestSPtr request = std::make_shared<Request>(RequestIdNew("test"));
    GlobalIpInfo globalIpInfo;
    globalIpInfo.role = "prefill";
    AddAttributeToRequest(globalIpInfo, request);
    EXPECT_EQ(request->role, PDRole::PREFILL);
    globalIpInfo.role = "decode";
    AddAttributeToRequest(globalIpInfo, request);
    EXPECT_EQ(request->role, PDRole::DECODE);
    globalIpInfo.role = "UNKNOWN";
    AddAttributeToRequest(globalIpInfo, request);
    EXPECT_EQ(request->role, PDRole::UNKNOWN);
}

TEST_F(InferInstanceTest, AddDevicesToRequest)
{
    RequestSPtr request = std::make_shared<Request>(RequestIdNew("test"));
    GlobalIpInfo globalIpInfo;
    DeviceInfo deviceInfo;
    deviceInfo.deviceIp = "0.0.0.0";
    deviceInfo.devicePhysicalId = 1;
    deviceInfo.superDeviceId = 0;
    globalIpInfo.unlinkIpInfo.insert({2, {deviceInfo}});
    globalIpInfo.linkIpInfo.insert({1, {deviceInfo}});
    globalIpInfo.hostIpInfo.insert({1, {"test"}});
    globalIpInfo.superPodIdInfo.insert({1, "test"});
    EXPECT_FALSE(AddDevicesToRequest(globalIpInfo, request));
    globalIpInfo.superPodIdInfo[1] = "2";
    EXPECT_TRUE(AddDevicesToRequest(globalIpInfo, request));
    EXPECT_EQ(request->dpInstance2HostIps[1], std::vector<std::string>{"test"});
    EXPECT_EQ(request->dpInstance2HostIps[2], std::vector<std::string>{"127.0.0.1"});
    MOCKER(ProcessDevice).stubs().will(returnValue(true)).then(returnValue(false));
    EXPECT_FALSE(AddDevicesToRequest(globalIpInfo, request));
    EXPECT_FALSE(AddDevicesToRequest(globalIpInfo, request));
}

TEST_F(InferInstanceTest, AddPolicyToRequest)
{
    RequestSPtr request = std::make_shared<Request>(RequestIdNew("test"));
    GlobalIpInfo globalIpInfo;
    AddPolicyToRequest(globalIpInfo, request);
    EXPECT_EQ(request->spInfo[0], 1);
    EXPECT_EQ(request->cpInfo[0], 1);
}

TEST_F(InferInstanceTest, BasicFields)
{
    GlobalIpInfo info;
    info.role = "encoder";
    info.needSwitch = true;
    info.localInstanceId = 123;
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo["infer_mode"], "dmi");
    EXPECT_EQ(ipInfo["role"], "encoder");
    EXPECT_EQ(ipInfo["needSwitch"], "true");
    EXPECT_EQ(ipInfo["local_instance_id"], "123");
}

TEST_F(InferInstanceTest, RoleDecodeConversion)
{
    GlobalIpInfo info;
    info.role = "decode";
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo["role"], "decoder");
}

TEST_F(InferInstanceTest, HostIpList)
{
    GlobalIpInfo info;
    info.localHostIpList = {"192.168.1.1", "192.168.1.2"};
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo["local_host_ip"], "192.168.1.1,192.168.1.2");
}

TEST_F(InferInstanceTest, SuperPodIdPresent)
{
    GlobalIpInfo info;
    info.localSuperPodId = "superpod-123";
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo["local_super_pod_id"], "superpod-123");
}

TEST_F(InferInstanceTest, SuperPodIdAbsent)
{
    GlobalIpInfo info;
    info.localSuperPodId = "";
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo.find("local_super_pod_id"), ipInfo.end());
}

TEST_F(InferInstanceTest, DeviceInfoFields)
{
    GlobalIpInfo info;
    info.localDeviceIps = {"10.0.0.1", "10.0.0.2"};
    info.localDeviceLogicalIds = {"log1", "log2"};
    info.localDevicePhysicalIds = {"phy1", "phy2"};
    info.localDeviceRankIds = {"rank1", "rank2"};
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo["local_device_ip"], "10.0.0.1,10.0.0.2");
    EXPECT_EQ(ipInfo["local_logic_device_id"], "log1,log2");
    EXPECT_EQ(ipInfo["local_physical_device_id"], "phy1,phy2");
    EXPECT_EQ(ipInfo["local_rank_ids"], "rank1,rank2");
}

TEST_F(InferInstanceTest, SuperDeviceIdsPresent)
{
    GlobalIpInfo info;
    info.localSuperDeviceIds = {"1001", "1002"};
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo["local_super_device_id"], "1001,1002");
}

TEST_F(InferInstanceTest, SuperDeviceIdsAbsent)
{
    GlobalIpInfo info;
    info.localSuperDeviceIds = {};
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo.find("local_super_device_id"), ipInfo.end());
}

TEST_F(InferInstanceTest, SingleContainerInfo)
{
    GlobalIpInfo info;
    info.isSingleContainer = true;
    info.instanceIdxInPod = 3;
    info.numInstancesPerPod = 8;
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo["lccl_comm_shard_id"], "3");
    EXPECT_EQ(ipInfo["num_lccl_comm_shards"], "8");
}

TEST_F(InferInstanceTest, NotSingleContainer)
{
    GlobalIpInfo info;
    info.isSingleContainer = false;
    info.instanceIdxInPod = 3;
    info.numInstancesPerPod = 8;
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo.find("lccl_comm_shard_id"), ipInfo.end());
    EXPECT_EQ(ipInfo.find("num_lccl_comm_shards"), ipInfo.end());
}

TEST_F(InferInstanceTest, EmptyVectors)
{
    GlobalIpInfo info;
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(info, ipInfo);
    EXPECT_EQ(ipInfo["local_host_ip"], "");
    EXPECT_EQ(ipInfo["local_device_ip"], "");
    EXPECT_EQ(ipInfo["local_logic_device_id"], "");
    EXPECT_EQ(ipInfo["local_physical_device_id"], "");
    EXPECT_EQ(ipInfo["local_rank_ids"], "");
}

TEST_F(InferInstanceTest, InitPDNode)
{
    GlobalIpInfo info;
    info.localDeviceLogicalIds = {"1", "2"};
    GlobalMockObject::verify();
    backendConfig_.multiNodesInferEnabled = false;
    MOCKER_CPP(GetBackendConfig, const BackendConfig& (*)())
            .stubs()
            .will(returnValue(backendConfig_));
    MockServerConfig();
    MockModelDeployConfig();
    MockScheduleConfig();
    MockLoraConfig();
    MockRanktableParam();
    MOCKER_CPP(static_cast<LlmManagerInitFunc>(&LlmManagerV2::Init), Status (*)(uint32_t, std::set<size_t>))
        .stubs()
        .will(returnValue(InternalErr()))
        .then(returnValue(Ok()));
    EXPECT_EQ(instance->InitPDNode(info).StatusCode(), Error::Code::ERROR);
    EXPECT_EQ(instance->InitPDNode(info).StatusCode(), Error::Code::OK);
}

TEST_F(InferInstanceTest, ForcePRelease)
{
    MOCKER_CPP(&LlmManagerV2::UpdateEngineInfo, bool (*)(RequestSPtr&, bool))
        .stubs()
        .will(returnValue(false)).then(returnValue(true));
    EXPECT_EQ(instance->ForcePRelease().StatusCode(), Error::Code::ERROR);
    EXPECT_EQ(instance->ForcePRelease().StatusCode(), Error::Code::OK);
}

TEST_F(InferInstanceTest, GetBatchSchedulerMetrics)
{
    EngineMetric engineMetric;
    engineMetric.schedulerInfo.reqsInfo.waitingRequestNum_ = 1;
    engineMetric.schedulerInfo.reqsInfo.runningRequestNum_ = 2;
    engineMetric.schedulerInfo.reqsInfo.swappedRequestNum_ = 3;
    engineMetric.schedulerInfo.reqsInfo.remainBlocks_ = 1;
    engineMetric.schedulerInfo.reqsInfo.remainPrefillSlots_ = 2;
    engineMetric.schedulerInfo.reqsInfo.remainPrefillTokens_ = 3;
    uint64_t total = 0;
    MOCKER_CPP(&LlmManagerV2::CollectEngineMetric, EngineMetric (*)())
        .stubs()
        .will(returnValue(engineMetric));
    std::map<std::string, uint64_t> batchSchedulerMetrics = {};
    EXPECT_EQ(instance->GetBatchSchedulerMetrics(batchSchedulerMetrics).StatusCode(), Error::Code::OK);
    EXPECT_EQ(batchSchedulerMetrics["waitingInferRequestNum"], 5);
    EXPECT_EQ(batchSchedulerMetrics["processingInferRequestNum"], 30);
    EXPECT_EQ(batchSchedulerMetrics["runningInferRequestNum"], 10);
    EXPECT_EQ(batchSchedulerMetrics["swappedInferRequestNum"], 15);
    EXPECT_EQ(batchSchedulerMetrics["remainBlocks"], 5);
}

TEST_F(InferInstanceTest, GetPDRole)
{
    instance->UpdatePDRole("prefill");
    EXPECT_EQ(instance->GetPDRole(), "prefill");
    instance->UpdatePDRole("decode");
    EXPECT_EQ(instance->GetPDRole(), "decode");
    instance->UpdatePDRole("none");
    EXPECT_EQ(instance->GetPDRole(), "none");
}

TEST_F(InferInstanceTest, SetAndGetPDRoleStatus)
{
    EXPECT_EQ(instance->GetPDRoleStatus(), PDRoleStatus::UNKNOWN);
    instance->SetPDRoleStatus(PDRoleStatus::READY);
    EXPECT_EQ(instance->GetPDRoleStatus(), PDRoleStatus::READY);
}

TEST_F(InferInstanceTest, Finalize)
{
    EXPECT_EQ(instance->Finalize().StatusCode(), Error::Code::OK);
    EXPECT_EQ(instance->Finalize().StatusCode(), Error::Code::OK);
}