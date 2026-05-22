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
 
#include <stdio.h>
#include <string.h>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "executor.h"
#include "communicator.h"
#include "secodeFuzz.h"
namespace mindie_llm {
#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))
class ExecutorFuzzTest : public ::testing::Test {
protected:
    std::unordered_map<std::string, std::string> baseConfig;
    int seed;
    int repeat;
    std::shared_ptr<Executor> executor;
    void SetUp() override
    {
        seed = 0;
        repeat = 100;
        baseConfig = {{"backend_bin_path", "/tmp/bin"},         {"deploy_type", "INTER_PROCESS"},
                      {"executor_type", "LLM_EXECUTOR_PYTHON"}, {"world_size", "8"},
                      {"npu_device_ids", "0,1,2,3,4,5,6,7"},    {"globalWorldSize", "8"},
                      {"globalRankIds", "0,1,2,3,4,5,6,7"},     {"model_instance_type", "Standard"},
                      {"backend_log_file", "/tmp/log"},         {"asyncBatchscheduler", "21"}};

        executor = std::make_shared<Executor>();
    }

    char *GetFuzzName()
    {
        auto testCaseName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        return const_cast<char *>(testCaseName);
    }
};
TEST_F(ExecutorFuzzTest, FuzzExecuteCommand_SimplePaths)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        int pathType = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        std::string command;

        if (pathType == 0) {
            command = "echo 'valid command'";
        } else {
            command = "/this/path/should/never/exist_";
        }
        bool result = executor->ExecuteCommand(command);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzInitWorkerProcesses)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        ModelLaunchConfig modelConfig;
        modelConfig.npuNumPerDP = *(uint32_t *)DT_SetGetNumberRange(&g_Element[0], 1, 1, 16);
        modelConfig.npuNumPerNode = *(uint32_t *)DT_SetGetNumberRange(&g_Element[1], 1, 1, 16);
        modelConfig.npuDeviceIds = {};
        modelConfig.globalRankIds = {};
        for (int i = 0; i < 20; i++) {
            modelConfig.npuDeviceIds.push_back(
                std::to_string(*(uint32_t *)DT_SetGetNumberRange(&g_Element[2], 1, 1, 16)));
            modelConfig.globalRankIds.push_back(
                std::to_string(*(uint32_t *)DT_SetGetNumberRange(&g_Element[3], 1, 1, 16)));
        }
        modelConfig.isMultiNodesInfer = *(int *)DT_SetGetNumberRange(&g_Element[4], 0, 0, 1);
        modelConfig.isMasterNode = *(int *)DT_SetGetNumberRange(&g_Element[5], 0, 0, 1);

        std::string sharedMemPrefix =
            "/test_prefix_" + std::to_string(*(int *)DT_SetGetNumberRange(&g_Element[6], 0, 0, 100));
        bool retExecute = *(int *)DT_SetGetNumberRange(&g_Element[7], 0, 0, 1);
        MOCKER_CPP(&Executor::ExecuteCommand, bool (*)(const std::string &)).stubs().will(returnValue(retExecute));

        executor->InitWorkerProcesses(modelConfig, sharedMemPrefix);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzGetRemoteDPRankIdx)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        ModelLaunchConfig modelConfig;
        modelConfig.isMultiNodesInfer = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        modelConfig.npuNumPerDP = *(uint32_t *)DT_SetGetNumberRange(&g_Element[1], 1, 1, 16);
        modelConfig.npuNumPerNode = *(uint32_t *)DT_SetGetNumberRange(&g_Element[2], 1, 1, 16);
        modelConfig.ipcCommunicatorNum = *(uint32_t *)DT_SetGetNumberRange(&g_Element[3], 1, 1, 8);
        modelConfig.isMasterNode = *(int *)DT_SetGetNumberRange(&g_Element[4], 0, 0, 1);

        int rankIdx = *(int *)DT_SetGetNumberRange(&g_Element[5], 0, -100, 100);
        bool intraNodeTP = *(int *)DT_SetGetNumberRange(&g_Element[6], 0, 0, 1);

        executor->GetRemoteDPRankIdx(modelConfig, rankIdx, intraNodeTP);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzGetCpuBlockNum)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        int isInitialized = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);

        if (isInitialized == 0) {
            IExecutor::kvCacheOverview_.cpuBlockNum = 0xFFFFFFFF;
        } else {
            IExecutor::kvCacheOverview_.cpuBlockNum = *(uint32_t *)DT_SetGetNumberRange(&g_Element[1], 1, 1, 1000);
        }

        executor->GetCpuBlockNum();
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzGetNpuBlockNum)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        int isInitialized = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);

        if (isInitialized == 0) {
            IExecutor::kvCacheOverview_.npuBlockNum = 0xFFFFFFFF;
        } else {
            IExecutor::kvCacheOverview_.npuBlockNum = *(uint32_t *)DT_SetGetNumberRange(&g_Element[1], 1, 1, 1000);
        }

        executor->GetNpuBlockNum();
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzGetMaxPositionEmbeddings)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        int isInitialized = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);

        if (isInitialized == 0) {
            IExecutor::kvCacheOverview_.maxPositionEmbeddings = 0xFFFFFFFF;
        } else {
            IExecutor::kvCacheOverview_.maxPositionEmbeddings =
                *(uint32_t *)DT_SetGetNumberRange(&g_Element[1], 1, 1, 10000);
        }

        executor->GetMaxPositionEmbeddings();
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(ExecutorFuzzTest, FuzzAsyncResponseHandler)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        int msgType = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 6);
        ExecuteResponse response;
        response.set_msg_type(msgType);

        executor->AsyncResponseHandler(response);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzHandlePDLinkResponse)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        int hasPdLinkResp = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        ExecuteResponse executeResponse;
        if (hasPdLinkResp == 0) {
            executeResponse.mutable_pd_link_response();
        }
        executor->HandlePDLinkResponse(executeResponse);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzHandleExecuteModelResponse)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();

        int isHandlerNull = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        int hasModelResp = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);

        bool isHandlerCalled = false;
        if (isHandlerNull == 0) {
            executor->executeModelResponseHandler_ = [&](ModelBatchResultSPtr result) { isHandlerCalled = true; };
        } else {
            executor->executeModelResponseHandler_ = nullptr;
        }

        ExecuteResponse modelExecuteResponse;
        if (hasModelResp == 0) {
            modelExecuteResponse.mutable_execute_model_response();
        }

        executor->HandleExecuteModelResponse(modelExecuteResponse);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzHandleInitResult)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        std::vector<ExecuteResponse> responses(2);
        char *keyEnums[] = {"cpuBlockNum", "npuBlockNum", "maxPositionEmbeddings", "randKeyA", "randKeyB"};
        char *valEnums[] = {"1", "invalid", "12345"};

        auto *resultMap0 = responses[0].mutable_init_results()->mutable_init_result_map();
        char *key0_0 = DT_SetGetStringEnum(&g_Element[0], 12, 50, "cpuBlockNum", keyEnums, 5);
        char *key0_1 = DT_SetGetStringEnum(&g_Element[1], 12, 50, "npuBlockNum", keyEnums, 5);
        char *key0_2 = DT_SetGetStringEnum(&g_Element[2], 22, 50, "maxPositionEmbeddings", keyEnums, 5);
        char *val0_0 = DT_SetGetStringEnum(&g_Element[3], 2, 10, "1", valEnums, 3);
        char *val0_1 = DT_SetGetStringEnum(&g_Element[4], 2, 10, "1", valEnums, 3);
        char *val0_2 = DT_SetGetStringEnum(&g_Element[5], 2, 10, "1", valEnums, 3);
        resultMap0->insert({key0_0, val0_0});
        resultMap0->insert({key0_1, val0_1});
        resultMap0->insert({key0_2, val0_2});

        auto *resultMap1 = responses[1].mutable_init_results()->mutable_init_result_map();
        char *key1_0 = DT_SetGetStringEnum(&g_Element[6], 12, 50, "cpuBlockNum", keyEnums, 5);
        char *key1_1 = DT_SetGetStringEnum(&g_Element[7], 12, 50, "npuBlockNum", keyEnums, 5);
        char *key1_2 = DT_SetGetStringEnum(&g_Element[8], 22, 50, "maxPositionEmbeddings", keyEnums, 5);
        char *val1_0 = DT_SetGetStringEnum(&g_Element[9], 2, 10, "1", valEnums, 3);
        char *val1_1 = DT_SetGetStringEnum(&g_Element[10], 2, 10, "1", valEnums, 3);
        char *val1_2 = DT_SetGetStringEnum(&g_Element[11], 2, 10, "1", valEnums, 3);
        resultMap1->insert({key1_0, val1_0});
        resultMap1->insert({key1_1, val1_1});
        resultMap1->insert({key1_2, val1_2});

        bool ret = executor->HandleInitResult(responses);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(ExecutorFuzzTest, FuzzExecutorInstanceFinalize)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(retSend));
        MOCKER_CPP(&Communicator::CleanUp, void (*)()).stubs();
        bool ret = executor->ExecutorInstanceFinalize();
    }
    DT_FUZZ_END() SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzExecuteKVTransfer)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(retSend));
        bool isNull = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);
        PullKVRequestPtr pullKVRequest;
        if (!isNull) {
            pullKVRequest = std::make_unique<model_execute_data::PullKVRequest>();
        } else {
            pullKVRequest = nullptr;
        }
        auto responseHandler = [](PullKVResponseSPtr) {};
        bool ret = executor->ExecuteKVTransfer(pullKVRequest, responseHandler);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzAsyncExecuteModel)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(retSend));
        bool isNull = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);
        ExecuteModelRequestPtr modelRequest;
        if (!isNull) {
            modelRequest = std::make_unique<model_execute_data::ExecuteModelRequest>();
        } else {
            modelRequest = nullptr;
        }
        auto responseHandler = [](ModelBatchResultSPtr) {};

        bool ret = executor->AsyncExecuteModel(modelRequest, responseHandler);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(ExecutorFuzzTest, FuzzAsyncTGCleanup)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(retSend));
        TGCleanupRequestPtr cleanupRequest = std::make_unique<model_execute_data::TGCleanupRequest>();
        bool ret = executor->AsyncTGCleanup(cleanupRequest);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(ExecutorFuzzTest, FuzzSetupPDLink)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        bool retHandle = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);

        MOCKER_CPP(&Communicator::SendSharedSyncLinkRequestAndReceive, bool (*)(ExecuteRequest &, ExecuteResponse &))
            .stubs()
            .will(returnValue(retSend));

        MOCKER_CPP(&Executor::HandlePDLinkResponse, bool (*)(ExecuteResponse &)).stubs().will(returnValue(retHandle));

        PDLinkRequest pdLinkRequest;

        bool ret = executor->SetupPDLink(pdLinkRequest);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(ExecutorFuzzTest, FuzzInitModelExecution)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        bool retHandle = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);
        bool retLaunch = *(int *)DT_SetGetNumberRange(&g_Element[2], 0, 0, 1);

        MOCKER_CPP(&Communicator::SendModelInitRequestAndReceive,
                   bool (*)(ExecuteRequest &, std::vector<ExecuteResponse> &))
            .stubs()
            .will(returnValue(retSend));
        MOCKER_CPP(&Executor::HandleInitResult, bool (*)(std::vector<ExecuteResponse> &))
            .stubs()
            .will(returnValue(retHandle));
        MOCKER_CPP(&Communicator::LaunchIPCHandleResponseThreads, bool (*)(ResponseHandler))
            .stubs()
            .will(returnValue(retLaunch));

        bool ret = executor->InitModelExecution(baseConfig);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, ParseFromModelConfig)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        // 构造随机 model config
        char *deployTypeEnums[] = {"INTER_PROCESS", "HHH_PROCESS"};
        char *configNameEnums[] = {"cp", "xp"};
        baseConfig["globalWorldSize"] = std::to_string(*(int *)DT_SetGetNumberRange(&g_Element[0], 1, 1, 16));
        baseConfig["world_size"] = std::to_string(*(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 8));
        baseConfig["deploy_type"] = DT_SetGetStringEnum(&g_Element[2], 14, 15, "INTER_PROCESS", deployTypeEnums, 2);
        baseConfig["isMaster"] = std::to_string(*(int *)DT_SetGetNumberRange(&g_Element[3], 0, 0, 1));
        baseConfig["tp"] = std::to_string(*(int *)DT_SetGetNumberRange(&g_Element[4], 0, 0, 8));
        char *configName = DT_SetGetStringEnum(&g_Element[5], 3, 3, "cp", configNameEnums, 2);
        baseConfig[std::string(configName)] = "2";
        baseConfig["slaveIPs"] = "127.0.0.1";
        baseConfig["localIP"] = "127.0.0.1";
        ModelLaunchConfig modelLaunchConfig;
        bool isMultiNodesInfer = *(int *)DT_SetGetNumberRange(&g_Element[6], 0, 0, 1);
        bool fullFieldConfig = *(int *)DT_SetGetNumberRange(&g_Element[7], 0, 0, 1);
        if (!fullFieldConfig) {
            baseConfig.erase("world_size");
            baseConfig.erase("tp");
        }
        bool ret = executor->ParseFromModelConfig(baseConfig, modelLaunchConfig, isMultiNodesInfer);

        int rankIdx = *(int *)DT_SetGetNumberRange(&g_Element[8], -1, -4, 4);
        bool intraNodeTP = *(int *)DT_SetGetNumberRange(&g_Element[9], 0, 0, 1);
        int remoteIdx = executor->GetRemoteDPRankIdx(modelLaunchConfig, rankIdx, intraNodeTP);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzExecutorInstanceInit)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retParse = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        bool retIpc = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);
        bool retWorker = *(int *)DT_SetGetNumberRange(&g_Element[2], 0, 0, 1);
        bool retModelExec = *(int *)DT_SetGetNumberRange(&g_Element[3], 0, 0, 1);
        bool retGrpc = *(int *)DT_SetGetNumberRange(&g_Element[4], 0, 0, 1);
        baseConfig["tp"] = std::to_string(*(int *)DT_SetGetNumberRange(&g_Element[5], 0, 0, 20));
        baseConfig["dp"] = std::to_string(*(int *)DT_SetGetNumberRange(&g_Element[6], 0, 0, 20));
        baseConfig["isMaster"] = std::to_string(*(int *)DT_SetGetNumberRange(&g_Element[7], 0, 0, 1));
        // mock内部函数
        MOCKER_CPP(&Executor::ParseFromModelConfig,
                   bool (*)(std::unordered_map<std::string, std::string> &, ModelLaunchConfig &, bool))
            .stubs()
            .will(returnValue(retParse));
        MOCKER_CPP(&Communicator::InitIPCCommunicators, bool (*)(const std::string &, uint32_t))
            .stubs()
            .will(returnValue(retIpc));
        MOCKER_CPP(&Executor::InitWorkerProcesses, bool (*)(const ModelLaunchConfig &, const std::string &))
            .stubs()
            .will(returnValue(retWorker));
        MOCKER_CPP(&Executor::InitModelExecution, bool (*)(std::unordered_map<std::string, std::string> &))
            .stubs()
            .will(returnValue(retModelExec));
        MOCKER_CPP(&Communicator::InitGRPCCommunicator,
                   bool (*)(std::unordered_map<std::string, std::string> &, ResponseHandler, uint32_t))
            .stubs()
            .will(returnValue(retGrpc));

        bool isMultiNodesInfer = *(int *)DT_SetGetNumberRange(&g_Element[7], 0, 0, 1);
        size_t rankIdx = *(int *)DT_SetGetNumberRange(&g_Element[8], 0, 0, 20);
        std::map<std::string, std::string> config(baseConfig.begin(), baseConfig.end());
        bool ret = executor->ExecutorInstanceInit(config, isMultiNodesInfer, rankIdx);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(ExecutorFuzzTest, FuzzRegisterPullKVResponseHandler)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        std::shared_ptr<Executor> executor = std::make_shared<Executor>();
        bool pullKVResponseHandled = false;
        // 构造不同 handler
        int branchChoice = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        PullKVResponseHandler handler;
        if (branchChoice == 0) {
            handler = [&](PullKVResponseSPtr) { pullKVResponseHandled = true; };
        } else if (branchChoice == 1) {
            handler = nullptr;
        }
        executor->RegisterPullKVResponseHandler(handler);

        // 构造 response
        ExecuteResponse response;
        if (*(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1)) {
            response.mutable_pull_kv_response(); // 有 pull_kv_response
        }
        executor->HandleKVTransferResponse(response);
    }
    DT_FUZZ_END()
    SUCCEED();
}
} // namespace mindie_llm