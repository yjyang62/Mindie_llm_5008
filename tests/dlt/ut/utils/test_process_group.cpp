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
#include <memory>
#include <unistd.h>
#include <sys/wait.h>
#include <string>
#include "process_group.h"

using namespace mindie_llm;
using namespace std;

class ProcessGroupCCTest : public ::testing::Test {
protected:
    void SetUp() override {}
    // pgPtr_ = make_unique<ProcessGroup>("127.0.0.1", "1027", )
};
const int process_num = 4;

void CreatePGAndAllGather(int rank)
{
    std::cout << "child " << rank << " started" << std::endl;
    const string masterAddr = "127.0.0.1";
    const uint16_t port = 8888;

    ProcessGroup::GetInstance(masterAddr, port, masterAddr, rank, process_num, rank == 0);
    std::vector<torch::Tensor> inputs;
    torch::Tensor t = torch::tensor({rank});
    inputs.push_back(t);
    std::vector<std::vector<torch::Tensor>> output = ProcessGroup::GetInstance().AllGather(inputs);
    vector<vector<torch::Tensor>> expected = {
        {torch::tensor({0}), torch::tensor({1}), torch::tensor({2}), torch::tensor({3})}};
    for (int i = 0; i < process_num; i++)
        ASSERT_TRUE(torch::equal(expected[0][i], output[0][i]));
}

TEST_F(ProcessGroupCCTest, AllGather)
{
    for (int i = 0; i < process_num; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            throw runtime_error("fork failed");
        } else if (pid == 0) {
            // create own process group and communicate
            CreatePGAndAllGather(i);
            return;
        }
    }

    wait(NULL);
    std::cout << "Child processes finished. parent ended!" << std::endl;
}