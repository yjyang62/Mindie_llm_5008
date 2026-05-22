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
#define private public
#include <nlohmann/json.hpp>
#include "mockcpp/mockcpp.hpp"
#include "src/server/endpoint/http_wrapper/dmi_role.h"
#include "src/server/endpoint/http_wrapper/dmi_role.cpp"
#include "httplib.h"
#include "parse_protocol.h"
#include <filesystem>
#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "mock_util.h"

using namespace mindie_llm;
using json = nlohmann::json;

MOCKER_CPP_OVERLOAD_EQ(ModelDeployConfig)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(BackendConfig)

namespace mindie_llm {

static std::string rankTableStringV1 = R"({
    "local": {
        "device": [
            {
                "device_id": "0",
                "device_ip": "1.1.1.1",
                "device_logical_id": "0"
            },
            {
                "device_id": "1",
                "device_ip": "1.1.1.2",
                "device_logical_id": "1"
            },
            {
                "device_id": "2",
                "device_ip": "1.1.1.3",
                "device_logical_id": "2"
            },
            {
                "device_id": "3",
                "device_ip": "1.1.1.4",
                "device_logical_id": "3"
            }
        ],
        "host_ip": "127.0.0.1",
        "id": 2003,
        "server_ip": "127.0.0.1",
		"instance_idx_in_pod": 0,
		"num_instances_per_pod": 1,
        "is_single_container": false
    },
    "peers": [
        {
            "device": [
                {
                    "device_id": "4",
                    "device_ip": "1.1.1.5",
                    "device_logical_id": "4"
                },
                {
                    "device_id": "5",
                    "device_ip": "1.1.1.6",
                    "device_logical_id": "5"
                },
                {
                    "device_id": "6",
                    "device_ip": "1.1.1.7",
                    "device_logical_id": "6"
                },
                {
                    "device_id": "7",
                    "device_ip": "1.1.1.8",
                    "device_logical_id": "7"
                }
            ],
            "host_ip": "127.0.0.1",
            "id": 2007,
            "server_ip": "127.0.0.1"
        }
    ]
})";

const std::string RESPONSE_OK_BODY = "{\"result\":\"ok\"}";
class DmiRoleTest : public testing::Test {
protected:
    void SetUp()
    {
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_http.json");
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        rankTableStringV2 = LoadJsonFile(GetParentDirectory() + "/../../config_manager/conf/v2_role_cross_node_2p_2d.json");
        if (rankTableStringV2.empty()) {
            return;
        }
        rankTableStringBefore = LoadJsonFile(GetParentDirectory() + "/../../config_manager/conf/role_1.json");
        if (rankTableStringBefore.empty()) {
            return;
        }

        rankTableStringAfter = LoadJsonFile(GetParentDirectory() + "/../../config_manager/conf/role_2.json");
        if (rankTableStringAfter.empty()) {
            return;
        }
        std::string validRequestBody;
        std::string validRequestBodyV2;
        std::string RESPONSE_OK_BODY;
        validRequestBody = R"({
            "rank_table": {
                "server_list": [
                    {
                        "server_id": "0.0.0.0",
                        "device": [
                            {"device_id": "0", "rank_id": "0"}
                        ]
                    }
                ]
            }
        })";
        validRequestBodyV2 = R"({
            "rank_table": {
                "server_list": [
                    {
                        "server_id": "1.1.1.1",
                        "device": [
                            {"device_id": "1", "rank_id": "1"}
                        ]
                    }
                ]
            }
        })";
    }

    void TearDown()
    {
        EnvUtil::GetInstance().ClearEnvVar("RANK_TABLE_FILE");
        EnvUtil::GetInstance().ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
        EnvUtil::GetInstance().ClearEnvVar("MIES_CONTAINER_IP");
        EnvUtil::GetInstance().ClearEnvVar("HOST_IP");
        GlobalMockObject::verify();
    }

    // Common function to load and parse a JSON file
    std::string LoadJsonFile(const std::string &filePath)
    {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Fail to open json file: " << filePath << std::endl;
            return ""; // Return empty string if file open fails
        }

        try {
            json j;
            file >> j; // Parse the JSON
            auto tabSize = 4;
            return j.dump(tabSize); // Return the JSON as a formatted string
        }
        catch (const json::parse_error &e) {
            std::cerr << "JSON Parse Error in file " << filePath << ": " << e.what() << std::endl;
            return ""; // Return empty string if parsing fails
        }
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

    void InitJson()
    {
        body = {
            {"local", {
                {
                    {"host_ip", "192.168.1.10"},
                    {"super_pod_id", "100"},
                    {"dp_inst_list", {
                        {
                            {"dp_inst_id", 1},
                            {"device", {
                                {
                                    {"device_ip", "10.0.0.1"},
                                    {"device_logical_id", "logical-1"},
                                    {"device_id", "physical-1"},
                                    {"rank_id", "0"},
                                    {"super_device_id", "super-1"}
                                },
                                {
                                    {"device_ip", "10.0.0.2"},
                                    {"device_logical_id", "logical-2"},
                                    {"device_id", "physical-2"},
                                    {"rank_id", "1"}
                                }
                            }}
                        }
                    }}
                },
                {
                    {"host_ip", "192.168.1.11"},
                    {"dp_inst_list", {
                        {
                            {"dp_inst_id", 2},
                            {"device", {
                                {
                                    {"device_ip", "10.0.0.3"},
                                    {"device_logical_id", "logical-3"},
                                    {"device_id", "physical-3"},
                                    {"rank_id", "2"}
                                }
                            }}
                        }
                    }}
                }
            }}
        };
    }

    void MockAllConfig()
    {
        MockServerConfig();
        MockBackendConfig();
        MockModelDeployConfig();
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
        serverConfig_.inferMode = "standard";
        serverConfig_.interCommTLSEnabled = true;
        serverConfig_.interCommPort = 1121;
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

    ServerConfig serverConfig_;
    BackendConfig backendConfig_;
    ModelDeployConfig modelDeployConfig_;
    DmiRole dmiRole;
    std::string rankTableStringV2;
    std::string rankTableStringBefore;
    std::string rankTableStringAfter;
    ordered_json body;
};

auto originalGetPDRole = &InferInstance::GetPDRole;

std::string MockGetPDRole()
{
    return "none";
}

TEST_F(DmiRoleTest, HandlePDRoleV1Init)
{
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req;
    req.body = rankTableStringV1;
    
    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    std::string roleName = "prefill";
    dmiRole.HandlePDRoleV1(ctx, roleName);

    EXPECT_EQ(ctx->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(DmiRoleTest, HandlePDRoleV1_PDParseRequestBodyToJsonFail)
{
    MOCKER_CPP(&DmiRole::PDParseRequestBodyToJson, bool (*)(const ReqCtxPtr&, ordered_json&))
        .stubs()
        .will(returnValue(false));
    httplib::Request req;
    req.body = rankTableStringV1;
    
    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    std::string roleName = "prefill";


    dmiRole.HandlePDRoleV1(ctx, roleName);

    EXPECT_EQ(ctx->Res().status, httplib::StatusCode::UnprocessableContent_422);
}

TEST_F(DmiRoleTest, HandlePDRoleV1NonSwitch)
{
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req1;
    req1.body = rankTableStringV1;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV1(ctx1, "prefill");
    EXPECT_EQ(ctx1->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx1->Res().body, "{\"result\":\"ok\"}");

    httplib::Request req2;
    req2.body = rankTableStringV1;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV1(ctx2, "decode");
    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(DmiRoleTest, HandlePDRoleV1Switch)
{
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req1;
    req1.body = rankTableStringV1;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV1(ctx1, "prefill");

    EXPECT_EQ(ctx1->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx1->Res().body, RESPONSE_OK_BODY);

    httplib::Request req2;
    req2.body = rankTableStringV1;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV1(ctx2, "decode");

    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx2->Res().body, RESPONSE_OK_BODY);
}

TEST_F(DmiRoleTest, HandlePDRoleV2Init_Success)
{
    const std::string validRequestBody = rankTableStringV2;

    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req;
    req.body = validRequestBody;
    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    std::string roleName = "prefill";

    dmiRole.HandlePDRoleV2(ctx, roleName);

    EXPECT_EQ(ctx->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx->Res().body, "{\"result\":\"ok\"}");
}

TEST_F(DmiRoleTest, HandlePDRoleV2Init_PDParseRequestBodyToJsonFail)
{
    MOCKER_CPP(&DmiRole::PDParseRequestBodyToJson, bool (*)(const ReqCtxPtr&, ordered_json&))
        .stubs()
        .will(returnValue(false));
    const std::string validRequestBody = rankTableStringV2;
    httplib::Request req;
    req.body = validRequestBody;
    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    std::string roleName = "prefill";

    dmiRole.HandlePDRoleV2(ctx, roleName);

    EXPECT_EQ(ctx->Res().status, httplib::StatusCode::UnprocessableContent_422);
    EXPECT_EQ(ctx->Res().body, "{\"error\":\"Req body converts to json fail. Reset to previous node status.\",\"error_type\":\"Input validation error\"}");
}

TEST_F(DmiRoleTest, HandlePDRoleV2Switch)
{
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req1;
    req1.body = rankTableStringV2;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV2(ctx1, "prefill");

    EXPECT_EQ(ctx1->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx1->Res().body, RESPONSE_OK_BODY);

    httplib::Request req2;
    req2.body = rankTableStringV2;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV2(ctx2, "decode");

    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx2->Res().body, RESPONSE_OK_BODY);
}

TEST_F(DmiRoleTest, HandlePDRoleV2NonSwitch)
{
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req1;
    req1.body = rankTableStringV2;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV2(ctx1, "prefill");

    EXPECT_EQ(ctx1->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx1->Res().body, RESPONSE_OK_BODY);

    httplib::Request req2;
    req2.body = rankTableStringV2;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV2(ctx2, "prefill");
    
    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx2->Res().body, RESPONSE_OK_BODY);
}

TEST_F(DmiRoleTest, HandlePDRoleV2RelinkFailure)
{
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    MOCKER_CPP(&DmiRole::UpdatePDInfoV2, bool(*)(const std::string&, const std::string&, const ordered_json&,
        GlobalIpInfo&))
            .stubs()
        .will(returnValue(false));

    httplib::Request req1;
    req1.body = rankTableStringBefore;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV2(ctx1, "prefill");

    httplib::Request req2;
    req2.body = rankTableStringAfter;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV2(ctx2, "prefill");

    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::ServiceUnavailable_503);
}

void MyRetryFunction(GlobalIpInfo& globalIpInfo)
{
    globalIpInfo.role = "test";
    return;
}

TEST_F(DmiRoleTest, RunThread)
{
    GlobalIpInfo globalIpInfo;
    dmiRole.ioContext_ = std::make_shared<boost::asio::io_context>();
    auto retryTask = std::make_unique<RetryTask>(MyRetryFunction, dmiRole.ioContext_, globalIpInfo);
    dmiRole.taskQueue_.Push(std::move(retryTask));
    dmiRole.retryTerminate_.store(true);
    dmiRole.RunThread();
    EXPECT_TRUE(dmiRole.retryTerminate_.load());
}

TEST_F(DmiRoleTest, ProcessInitInfoV2_NormalCase)
{
    InitJson();
    GlobalIpInfo globalIpInfo;
    dmiRole.ProcessInitInfoV2(body, globalIpInfo);

    EXPECT_TRUE(globalIpInfo.needInit);
    EXPECT_EQ(globalIpInfo.numInstancesPerPod, 64);
    EXPECT_EQ(globalIpInfo.localInstanceId, 0);
    EXPECT_EQ(globalIpInfo.localSuperPodId, "100");

    std::vector<std::string> expectedHostIps = {"192.168.1.10", "192.168.1.11"};
    std::vector<uint64_t> expectedDpInstIds = {1, 2};
    std::vector<std::string> expectedDeviceIps = {"10.0.0.1", "10.0.0.2", "10.0.0.3"};
    std::vector<std::string> expectedLogicalIds = {"logical-1", "logical-2", "logical-3"};
    std::vector<std::string> expectedPhysicalIds = {"physical-1", "physical-2", "physical-3"};
    std::vector<std::string> expectedRankIds = {"0", "1", "2"};
    std::vector<std::string> expectedSuperDeviceIds = {"super-1"};

    EXPECT_EQ(globalIpInfo.localHostIpList, expectedHostIps);
    EXPECT_EQ(globalIpInfo.localDpInstanceIds, expectedDpInstIds);
    EXPECT_EQ(globalIpInfo.localDeviceIps, expectedDeviceIps);
    EXPECT_EQ(globalIpInfo.localDeviceLogicalIds, expectedLogicalIds);
    EXPECT_EQ(globalIpInfo.localDevicePhysicalIds, expectedPhysicalIds);
    EXPECT_EQ(globalIpInfo.localDeviceRankIds, expectedRankIds);
    EXPECT_EQ(globalIpInfo.localSuperDeviceIds, expectedSuperDeviceIds);
}

TEST_F(DmiRoleTest, ProcessInitInfoV2_MissingField)
{
    GlobalIpInfo globalIpInfo;
    ordered_json body1 = {
        {"local", {
            {
                {"host_ip", "192.168.1.10"},
            }
        }}
    };
    EXPECT_THROW(
        dmiRole.ProcessInitInfoV2(body1, globalIpInfo),
        std::runtime_error
    );
}

TEST_F(DmiRoleTest, GetInstanceIdToServerIp)
{
    const std::map<uint32_t, std::string> expected = {};
    EXPECT_EQ(dmiRole.GetInstanceIdToServerIp(), expected);
}

TEST_F(DmiRoleTest, GetRemoteNodeLinkStatusV2)
{
    const std::map<uint64_t, std::pair<std::string, bool>> expected = {};
    EXPECT_EQ(dmiRole.GetRemoteNodeLinkStatusV2(), expected);
}

TEST_F(DmiRoleTest, RetryLinkCallback)
{
    MockAllConfig();
    MOCKER_CPP(&Status::IsOk, bool(*)()).stubs()
        .will(returnValue(false))
        .then(returnValue(true));
    GlobalIpInfo globalIpInfo;
    MOCKER_CPP(&InferInstance::AssignDmiRole, Status(*)(GlobalIpInfo&)).stubs()
        .will(returnValue(Status(Error::Code::OK, "Success")));
    dmiRole.RetryLinkCallback(globalIpInfo);
    dmiRole.RetryLinkCallback(globalIpInfo);
    EXPECT_EQ(dmiRole.successLinkIP_.size(), 0);
}

TEST_F(DmiRoleTest, SingleInstanceSingleDpInstance_Ok)
{
    std::map<uint64_t, std::pair<std::string, bool>> input = {
        {10001, {"status1", true}}
    };
    
    auto result = GetInstanceStatus(input);
    
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[1].first, "ok");
    EXPECT_TRUE(result[1].second);
}

TEST_F(DmiRoleTest, SingleInstanceSingleDpInstance_Error)
{
    std::map<uint64_t, std::pair<std::string, bool>> input = {
        {10001, {"error1", false}}
    };
    
    auto result = GetInstanceStatus(input);
    
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[1].first, "dp instance id : 10001error1");
    EXPECT_FALSE(result[1].second);
}

class UpdateSuccessLinkIpTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        globalIpInfo.unlinkIpInfo = {{1001, {}}, {1002, {}}};
        globalIpInfo.linkIpInfo = {{1003, {}}, {1004, {}}};
        globalIpInfo.retryLinkIpInfo = {{1005, {}}, {1006, {}}};
        globalIpInfo.failLinkInstanceIDAndReason = {{1005, 203005}, {1006, 203004}};
    }
    
    mindie_llm::GlobalIpInfo globalIpInfo;
    mindie_llm::DmiRole dmiRole;
};

TEST_F(UpdateSuccessLinkIpTest, BasicTest)
{
    dmiRole.UpdateSuccessLinkIp(globalIpInfo);

    const auto& successLinkIp = dmiRole.GetSuccessLinkIp();
    const auto& remoteNodeLinkStatus = dmiRole.GetRemoteNodeLinkStatus();

    EXPECT_EQ(successLinkIp.size(), 2);
    EXPECT_EQ(successLinkIp.count(1003), 1);
    EXPECT_EQ(successLinkIp.count(1004), 1);

    EXPECT_EQ(remoteNodeLinkStatus.size(), 4);
    EXPECT_EQ(remoteNodeLinkStatus.at(1003).first, "ok");
    EXPECT_TRUE(remoteNodeLinkStatus.at(1003).second);
    EXPECT_EQ(remoteNodeLinkStatus.at(1004).first, "ok");
    EXPECT_TRUE(remoteNodeLinkStatus.at(1004).second);
    EXPECT_EQ(remoteNodeLinkStatus.at(1005).first, "failed : Timeout");
    EXPECT_TRUE(remoteNodeLinkStatus.at(1005).second);
    EXPECT_EQ(remoteNodeLinkStatus.at(1006).first, "failed : Engine error");
    EXPECT_TRUE(remoteNodeLinkStatus.at(1006).second);
}

class ResetContextTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        dmiRole = new mindie_llm::DmiRole();
    }
    
    void TearDown() override
    {
        delete dmiRole;
        dmiRole = nullptr;
    }
    
    mindie_llm::DmiRole *dmiRole;
};

TEST_F(ResetContextTest, NoError)
{
    boost::system::error_code ec;
    dmiRole->ResetContext(ec);
    SUCCEED();
}

TEST_F(ResetContextTest, WithError)
{
    boost::system::error_code ec(1, boost::system::system_category());
    dmiRole->ResetContext(ec);
    SUCCEED();
}

class RetryTaskExecuteTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ioContext = std::make_shared<boost::asio::io_context>();
        globalIpInfo = new mindie_llm::GlobalIpInfo();
    }
    
    void TearDown() override
    {
        delete globalIpInfo;
        globalIpInfo = nullptr;
    }
    
    std::shared_ptr<boost::asio::io_context> ioContext;
    mindie_llm::GlobalIpInfo *globalIpInfo;
    std::atomic<int> retryFuncCalled{0};
};

TEST_F(RetryTaskExecuteTest, BasicTest)
{
    mindie_llm::RetryTask::RetryFunc retryFunc = [this](mindie_llm::GlobalIpInfo &globalIpInfo) {
        retryFuncCalled++;
    };
    mindie_llm::RetryTask task(retryFunc, ioContext, *globalIpInfo, 1);
    task.Execute();
    EXPECT_GT(retryFuncCalled, 0);
}

class ModifyPullKVFailIdTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        dmiRole = new mindie_llm::DmiRole();
    }
    
    void TearDown() override
    {
        delete dmiRole;
        dmiRole = nullptr;
    }
    mindie_llm::DmiRole *dmiRole;
};

TEST_F(ModifyPullKVFailIdTest, BasicTest)
{
    dmiRole->ModifyPullKVFailId(1001);
    const auto& successLinkIP = dmiRole->GetSuccessLinkIp();
    const auto& remoteNodeLinkStatus = dmiRole->GetRemoteNodeLinkStatus();
    bool isHealthy = dmiRole->IsHealthy();

    EXPECT_TRUE(successLinkIP.empty());
    EXPECT_EQ(remoteNodeLinkStatus.size(), 1);
    EXPECT_EQ(remoteNodeLinkStatus.at(1001).first, "failed : pull kv failed.");
    EXPECT_FALSE(isHealthy);
}

class GetLocalInstanceIdTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        dmiRole = new mindie_llm::DmiRole();
    }
    
    void TearDown() override
    {
        delete dmiRole;
        dmiRole = nullptr;
    }
    
    mindie_llm::DmiRole *dmiRole;
};

TEST_F(GetLocalInstanceIdTest, DefaultValue)
{
    const uint32_t &instanceId = dmiRole->GetLocalInstanceId();
    EXPECT_EQ(instanceId, 0);
}

class IsHealthyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        dmiRole = new mindie_llm::DmiRole();
    }
    
    void TearDown() override
    {
        delete dmiRole;
        dmiRole = nullptr;
    }
    mindie_llm::DmiRole *dmiRole;
};

TEST_F(IsHealthyTest, InitialStateHealthy)
{
    bool isHealthy = dmiRole->IsHealthy();
    EXPECT_TRUE(isHealthy);
}

TEST_F(IsHealthyTest, UnhealthyAfterModifyPullKVFailId)
{
    dmiRole->ModifyPullKVFailId(1001);
    bool isHealthy = dmiRole->IsHealthy();
    EXPECT_FALSE(isHealthy);
}
} // namespace llm