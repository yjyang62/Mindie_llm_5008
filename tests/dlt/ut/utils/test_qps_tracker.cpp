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
#include <thread>
#include <chrono>
#include "qps_tracker.h"

using namespace mindie_llm;
using namespace std;

class QPSTrackerTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(QPSTrackerTest, verifyQPS)
{
    float presetQPS = 100.0;
    int64_t sendIntervalMilli = (int64_t)(1000 / presetQPS);

    QPSTracker tracker(1000, 10); // 1000ms time window, and 10ms for a bucket
    thread sender([&tracker, sendIntervalMilli]() {
        int times = 10;
        while (times >= 0) {
            tracker.Record();
            times--;
            std::this_thread::sleep_for(chrono::milliseconds(sendIntervalMilli));
        }
    });

    int readQpsTimes = 1;
    while (readQpsTimes) {
        float qps = tracker.GetQPS(); // correct qps needs time to be accurate
        std::cout << "calcuated qps is " << qps << "; correct qps is " << presetQPS << std::endl;
        std::this_thread::sleep_for(chrono::milliseconds(sendIntervalMilli));
        readQpsTimes--;
    }
    sender.join();
}