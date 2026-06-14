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
#include "thread_group_cc.h"

using namespace mindie_llm;

class ThreadGroupCCTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ThreadGroupCCTest, AllGather)
{
    constexpr size_t numThreads = 3;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back([i]() {
            // Step 1: AllGather
            int times = 0;
            while (times < 10) {
                if (i == 0) {
                    std::vector<int64_t> sendData = {times, times + 1, times + 2, times + 3};
                    std::vector<std::vector<int64_t>> recvData;

                    std::cout << "[Master] AllGather ing...\n";
                    ThreadGroupCC::GetInstance(numThreads).AllGather(sendData, recvData, i);
                    std::cout << "[Master] AllGather done.\n";

                    std::cout << "[Receiver " << i << "] Received: ";
                    for (const std::vector<int64_t> &item : recvData) {
                        for (int64_t x : item) {
                            std::cout << x << " ";
                        }
                        std::cout << "\n";
                    }
                } else {
                    int64_t value = times + i;
                    std::vector<int64_t> sendData = {value, value + 1, value + 2, value + 3};
                    std::vector<std::vector<int64_t>> recvData;

                    ThreadGroupCC::GetInstance(numThreads).AllGather(sendData, recvData, i);

                    std::cout << "[Receiver " << i << "] Received: ";
                    for (const std::vector<int64_t> &item : recvData) {
                        for (int64_t x : item) {
                            std::cout << x << " ";
                        }
                        std::cout << "\n";
                    }
                }
                times++;
            }
        });
    }

    for (auto &t : threads) {
        t.join();
    }
}
