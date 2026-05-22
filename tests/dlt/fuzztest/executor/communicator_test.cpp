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
#include <mockcpp/mockcpp.hpp>
#define private public
#include "communicator.h"
#include "ipc_communicator.h"
#include "secodeFuzz.h"
namespace mindie_llm {
class CommunicatorFuzzTest : public ::testing::Test {
protected:
    int seed;
    int repeat;
    std::unordered_map<std::string, std::string> config;
    std::shared_ptr<Communicator> comm;
    void SetUp() override
    {
        seed = 0;
        repeat = 100;
        config["isMaster"] = "1";
        config["slaveIPs"] = "127.0.0.2,127.0.0.3,127.0.0.4";
        config["masterIP"] = "127.0.0.1";
        config["multiNodesInferPort"] = "12345";
        config["localIP"] = "127.0.0.1";
        config["rank"] = "0";
        config["world_size"] = "1";
        config["dp"] = "4";
        comm = std::make_shared<Communicator>(config, false, 0, 0, false);
    }

    char *GetFuzzName()
    {
        auto testCaseName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        return const_cast<char *>(testCaseName);
    }
};
TEST_F(CommunicatorFuzzTest, FuzzSendAsyncRequestFull)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool isMultiNodes = *(int*)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        int msRole = *(int*)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);
        bool hasIPC = *(int*)DT_SetGetNumberRange(&g_Element[2], 0, 0, 1);
        bool hasGRPC = *(int*)DT_SetGetNumberRange(&g_Element[3], 0, 0, 1);
        bool sendLocalSuccess = *(int*)DT_SetGetNumberRange(&g_Element[4], 0, 0, 1);
        bool sendRemoteSuccess = *(int*)DT_SetGetNumberRange(&g_Element[5], 0, 0, 1);

        config["isMaster"] = msRole == 0 ? "1" : "0";

        comm = std::make_shared<Communicator>(config, isMultiNodes, 0, 0, false);

        comm->isMultiNodesInfer_ = isMultiNodes;
        comm->msRole_ = msRole == 0 ? MasterSlaveRole::MASTER : MasterSlaveRole::SLAVE;

        if (hasIPC) {
            comm->ipcCommunicatorExecute_ = std::make_unique<IPCCommunicator>("exec", 1);
            comm->ipcCommunicatorKVTransfer_ = std::make_unique<IPCCommunicator>("kv", 1);
        } else {
            comm->ipcCommunicatorExecute_.reset();
            comm->ipcCommunicatorKVTransfer_.reset();
        }

        if (hasGRPC) {
            MOCKER_CPP(&GRPCCommunicator::GetInstance,
                      std::shared_ptr<GRPCCommunicator>(*)(const std::unordered_map<std::string, std::string>&))
                .stubs()
                .will(returnValue(std::make_shared<GRPCCommunicator>(config)));

            comm->grpcCommunicator_ = GRPCCommunicator::GetInstance(config);
        } else {
            comm->grpcCommunicator_.reset();
        }

        MOCKER_CPP(&Communicator::SendAsyncRequestToLocal, bool(*)(ExecuteRequest&))
            .stubs()
            .will(returnValue(sendLocalSuccess));
        MOCKER_CPP(&Communicator::SendAsyncRequestToRemote, bool(*)(ExecuteRequest&))
            .stubs()
            .will(returnValue(sendRemoteSuccess));

        ExecuteRequest request;
        int type = *(int*)DT_SetGetNumberRange(&g_Element[6], 0, 0, 5);
        request.set_execute_type(static_cast<ExecuteType>(type));

        bool ret = comm->SendAsyncRequest(request);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(CommunicatorFuzzTest, FuzzCommunicatorConstructor)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        int isMaster = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        config["isMaster"] = std::to_string(isMaster);
        bool isMultiNodesInfer = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);
        int dpRankIdx = *(int *)DT_SetGetNumberRange(&g_Element[2], 0, 0, 3);
        int remoteDPRankIdx = *(int *)DT_SetGetNumberRange(&g_Element[3], 0, 0, 8);
        bool intraNodeTP = *(int *)DT_SetGetNumberRange(&g_Element[4], 0, 0, 1);

        Communicator communicator(config, isMultiNodesInfer, dpRankIdx, remoteDPRankIdx, intraNodeTP);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(CommunicatorFuzzTest, FuzzInitGRPCCommunicator)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        int msRole = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 2);
        if (msRole == 0) {
            comm->msRole_ = MasterSlaveRole::MASTER;
        } else if (msRole == 1) {
            comm->msRole_ = MasterSlaveRole::SLAVE;
        } else {
            comm->msRole_ = static_cast<MasterSlaveRole>(2);
        }
        bool retRegister = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);
        bool retInit = *(int *)DT_SetGetNumberRange(&g_Element[2], 0, 0, 1);
        config["isMaster"] = "1";
        config["slaveIPs"] = "127.0.0.1";
        config["masterIP"] = "127.0.0.1";
        config["multiNodesInferPort"] = "12345";
        config["localIP"] = "127.0.0.1";

        MOCKER_CPP(&GRPCCommunicator::RegisterResponseHandler, bool (*)(ResponseHandler, int))
            .stubs()
            .will(returnValue(retRegister));
        MOCKER_CPP(&GRPCCommunicator::RegisterRequestHandler, bool (*)(RequestHandler, int))
            .stubs()
            .will(returnValue(retRegister));
        MOCKER_CPP(&GRPCCommunicator::Init, bool (*)(uint32_t)).stubs().will(returnValue(retInit));

        ResponseHandler handler = [](ExecuteResponse &) {};
        uint32_t grpcCommunicatorNum = *(uint32_t *)DT_SetGetNumberRange(&g_Element[3], 1, 1, 8);

        bool ret = comm->InitGRPCCommunicator(config, handler, grpcCommunicatorNum);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(CommunicatorFuzzTest, FuzzSendModelInitRequestAndReceive)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        bool retRecv = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);

        MOCKER_CPP(&IPCCommunicator::SendMessageViaSM, bool (*)(ExecuteRequest &)).stubs().will(returnValue(retSend));
        MOCKER_CPP(&IPCCommunicator::ReceiveInitResponses, bool (*)(std::vector<ExecuteResponse> &))
            .stubs()
            .will(returnValue(retRecv));

        ExecuteRequest request;
        std::vector<ExecuteResponse> responses;
        bool ret = comm->SendModelInitRequestAndReceive(request, responses);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(CommunicatorFuzzTest, FuzzSendPDLinkRequestAndReceive)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        bool retRecv = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);

        MOCKER_CPP(&IPCCommunicator::SendMessageViaSM, bool (*)(ExecuteRequest &)).stubs().will(returnValue(retSend));
        MOCKER_CPP(&IPCCommunicator::ReceiveResponse, bool (*)(ExecuteResponse &)).stubs().will(returnValue(retRecv));

        ExecuteRequest request;
        ExecuteResponse response;
        bool ret = comm->SendSharedSyncLinkRequestAndReceive(request, response);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(CommunicatorFuzzTest, FuzzLaunchIPCHandleResponseThreads)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        int msRole = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        if (msRole == 0) {
            comm->msRole_ = MasterSlaveRole::MASTER;
        } else if (msRole == 1) {
            comm->msRole_ = MasterSlaveRole::SLAVE;
        }
        bool isMultiNodesInfer = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);
        comm->isMultiNodesInfer_ = isMultiNodesInfer;
        bool retResponse = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        bool retStart = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);

        MOCKER_CPP(&IPCCommunicator::RegisterResponseHandler, bool (*)(ResponseHandler))
            .stubs()
            .will(returnValue(retResponse));
        MOCKER_CPP(&IPCCommunicator::StartHandleResponseThread, bool (*)()).stubs().will(returnValue(retStart));

        ResponseHandler handler = [](ExecuteResponse &) {};
        bool ret = comm->LaunchIPCHandleResponseThreads(handler);
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(CommunicatorFuzzTest, FuzzSlaveNodeGRPCRequestHandler)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);

        MOCKER_CPP(&Communicator::SendAsyncRequestToLocal, bool (*)(ExecuteRequest &))
            .stubs()
            .will(returnValue(retSend));

        ExecuteRequest request;
        bool ret = comm->SlaveNodeGRPCRequestHandler(request);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST_F(CommunicatorFuzzTest, FuzzSlaveNodeIPCResponseHandler)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        MOCKCPP_NS::GlobalMockObject::reset();
        comm->intraNodeTP_ = *(int *)DT_SetGetNumberRange(&g_Element[0], 0, 0, 1);
        bool retSend = *(int *)DT_SetGetNumberRange(&g_Element[1], 0, 0, 1);

        MOCKER_CPP(&GRPCCommunicator::SendResponse, bool (*)(ExecuteResponse &, int, int))
            .stubs()
            .will(returnValue(retSend));

        ExecuteResponse response;
        bool ret = comm->SlaveNodeIPCResponseHandler(response);
    }
    DT_FUZZ_END()
    SUCCEED();
}
} // namespace mindie_llm