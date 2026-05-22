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
#include <gtest/gtest.h>
#define private public
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <pthread.h>
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/MockObject.h>
#include "grpc_communicator.h"
#include "secodeFuzz.h"

namespace mindie_llm {

using namespace mockcpp;

class GrpcCommunicatorTest : public ::testing::Test {
protected:
    int seed;
    int repeat;
    std::shared_ptr<GRPCCommunicator> grpcCommunicatorMaster_;
    std::shared_ptr<GRPCCommunicator> grpcCommunicatorSlave_;
    static char* ipEnums[];
    static const int ipCount;
    static char* portEnums[];
    static const int portCount;
    static char* ipListEnums[];
    static const int ipListCount;

    void SetUp() override
    {
        seed = 0;
        repeat = 10;
        GlobalMockObject::reset();
    }

    void TearDown() override
    {
        GlobalMockObject::reset();
    }

    char *GetFuzzName()
    {
        auto testCaseName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        return const_cast<char *>(testCaseName);
    }

    std::unordered_map<std::string, std::string> GetConfig(int &element)
    {
        char* slaveIPs = DT_SetGetStringEnum(
            &g_Element[element++], 100, 100, "127.0.0.2", ipListEnums, ipListCount
        );

        char* masterIP = DT_SetGetStringEnum(
            &g_Element[element++], 20, 20, "127.0.0.1", ipEnums, ipCount
        );

        char* port = DT_SetGetStringEnum(
            &g_Element[element++], 5, 6, "8080", portEnums, portCount
        );

        char* localIP = DT_SetGetStringEnum(
            &g_Element[element++], 20, 20, "127.0.0.2", ipEnums, ipCount
        );

        std::unordered_map<std::string, std::string> config = {
            {"isMaster", "1"},
            {"slaveIPs", slaveIPs},
            {"masterIP", masterIP},
            {"multiNodesInferPort", port},
            {"localIP", localIP}
        };
        std::cout << slaveIPs << "-" << masterIP << "-" << port << "-" << localIP << std::endl;
        return config;
    }
};

char* GrpcCommunicatorTest::ipEnums[] = {
    "localhost",
    "127.0.0.1",
    "127.0.0.2",
    "0.0.0.0"
};
const int GrpcCommunicatorTest::ipCount = sizeof(GrpcCommunicatorTest::ipEnums) / sizeof(GrpcCommunicatorTest::ipEnums[0]);

char* GrpcCommunicatorTest::portEnums[] = {
    "50051",
    "8080",
    "8000",
    "9000",
    "10000",
    "65535",
    "12345",
    "54321",
};
const int GrpcCommunicatorTest::portCount = sizeof(GrpcCommunicatorTest::portEnums) / sizeof(GrpcCommunicatorTest::portEnums[0]);

char* GrpcCommunicatorTest::ipListEnums[] = {
    "192.168.1.101",
    "10.0.0.2",
    "172.16.0.2",
    "127.0.0.2",
    "192.168.1.101"
};
const int GrpcCommunicatorTest::ipListCount = sizeof(GrpcCommunicatorTest::ipListEnums) / sizeof(GrpcCommunicatorTest::ipListEnums[0]);


TEST_F(GrpcCommunicatorTest, FuzzExecuteCommand_Init)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        int element = 0;
        std::unordered_map<std::string, std::string> config = GetConfig(element);
        config["isMaster"] = "1";
        grpcCommunicatorMaster_ = std::make_shared<GRPCCommunicator>(config);
        config["isMaster"] = "0";
        grpcCommunicatorSlave_ = std::make_shared<GRPCCommunicator>(config);
        std::thread server_thread([&] {
            std::cout << "start master!" << std::endl;
            bool masterInitRes = grpcCommunicatorMaster_->Init(1);
            std::cout << "start master success!" << std::endl;
            grpcCommunicatorMaster_->StopServer();
        });
        std::cout << "wait start master!" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "start slave!" << std::endl;
        bool slaveInitRes = grpcCommunicatorSlave_->Init(1);
        std::cout << "start slave success!" << std::endl;
        grpcCommunicatorSlave_->StopClient();
        server_thread.join();
    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(GrpcCommunicatorTest, SendRequest_Success)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, GetFuzzName(), 0)
    {
        int element = 0;
        std::unordered_map<std::string, std::string> config = GetConfig(element);
        config["isMaster"] = "1";
        grpcCommunicatorMaster_ = std::make_shared<GRPCCommunicator>(config);
        config["isMaster"] = "0";
        grpcCommunicatorSlave_ = std::make_shared<GRPCCommunicator>(config);

        std::thread server_thread([&] {
            bool masterInitRes = grpcCommunicatorMaster_->Init(1);
            ExecuteRequest request;
            request.set_execute_type(model_execute_data::ExecuteType::MODEL_INFER);
            for (int i = 0; i < 10000; i++) {
                bool sendRequest = grpcCommunicatorMaster_->SendRequest(request, 0, 0);
            }
            grpcCommunicatorMaster_->StopServer();
        });

        std::this_thread::sleep_for(std::chrono::seconds(1));
        bool slaveInitRes = grpcCommunicatorSlave_->Init(1);
        std::this_thread::sleep_for(std::chrono::seconds(5));
        grpcCommunicatorSlave_->StopClient();
        server_thread.join();
    }
    DT_FUZZ_END()
    SUCCEED();
}
}