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
 
#include <string>
#include <stdexcept>
#include <random>
#include <unistd.h>
#include <mutex>
#include <vector>
#include "gtest/gtest.h"
#define private public
#include "dummy_quota_manager.h"

using namespace mindie_llm;
using namespace std;

class DummyQuotaManagerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

void WaitOrKillCldProcess(pid_t pid)
{
    cout << "WaitOrKillCldProcess " << pid << endl;
    int status;
    int attempts = 0;
    int result = 0;
    do {
        result = waitpid(pid, &status, WNOHANG);
        if (result == 0) {
            usleep(100000); // 100ms delay
            attempts++;
        }
    } while (result == 0 && attempts < 20); // Timeout after 2 seconds
    if (result == 0) {
        // Timeout reached, kill the child
        kill(pid, SIGTERM);
        cout << "timeout , parent process killed child " << pid << endl;
        waitpid(pid, &status, 0); // Clean up
    }
    cout << "WaitOrKillCldProcess " << pid << "exit" << endl;
}

void SpendAllQuota(DummyQuotaManager &dummyQuotaMgr, bool isMaster, int rank, int process_num)
{
    for (int i = 0; i < dummyQuotaMgr.initQuota_; i++) {
        bool succ = dummyQuotaMgr.AcquireQuota(true); // dummy batch needs a quota
        ASSERT_TRUE(succ);
    }
    bool succ = dummyQuotaMgr.AcquireQuota(true); // all quota are out.
    ASSERT_FALSE(succ);
    succ = dummyQuotaMgr.AcquireQuota(true); // all quota are out.
    ASSERT_FALSE(succ);
    std::cout << "before wakeup left quota=" << dummyQuotaMgr.quotaLeft_.load() << std::endl;
    dummyQuotaMgr.Wakeup(); // all quota are out.
    this_thread::sleep_for(chrono::milliseconds(100));
    std::cout << "after wake up left quota=" << dummyQuotaMgr.quotaLeft_.load() << std::endl;
    succ = dummyQuotaMgr.AcquireQuota(true); // quota is restored
    ASSERT_TRUE(succ);
    succ = dummyQuotaMgr.AcquireQuota(false); // alway return true for real request
    ASSERT_TRUE(succ);
}

void RealReqWithDummy(DummyQuotaManager &dummyQuotaMgr, bool sendDummy)
{
    std::cout << "sendDummy=" << sendDummy << std::endl;
    bool succ = dummyQuotaMgr.AcquireQuota(sendDummy);
    ASSERT_TRUE(succ);
    this_thread::sleep_for(chrono::milliseconds(100));
    std::cout << "left quota=" << dummyQuotaMgr.quotaLeft_.load() << std::endl;
    succ = dummyQuotaMgr.AcquireQuota(true);
    ASSERT_TRUE(succ);
}

void RandomPressure(DummyQuotaManager &dummyQuotaMgr, int rounds)
{
    while (rounds) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 10);
        int randNum = dis(gen);
        bool sendDummy = (randNum <= 8); // more likely to send Dummy
        if (!sendDummy) {
            dummyQuotaMgr.Wakeup();
        }
        std::cout << "--p" << getpid() << " quota: " << dummyQuotaMgr.quotaLeft_.load() << std::endl;
        dummyQuotaMgr.AcquireQuota(sendDummy);
        std::cout << "++p" << getpid() << " quota: " << dummyQuotaMgr.quotaLeft_.load() << std::endl;
        this_thread::sleep_for(chrono::milliseconds(10 * static_cast<int>(randNum)));
        rounds--;
    }

    // drain quota, so don't block in QuotaAllGather_
    for (int i = 0; i < dummyQuotaMgr.initQuota_; i++) {
        dummyQuotaMgr.AcquireQuota(true); // dummy batch needs a quota
    }
    dummyQuotaMgr.threadNeedStop_.store(true);
    this_thread::sleep_for(chrono::milliseconds(100)); // wait thread to close to avoid crash
}

TEST_F(DummyQuotaManagerTest, DISABLED_TestSpendAllQuota)
{
    int process_num = 2;
    for (int i = 1; i < process_num; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            throw runtime_error("fork failed");
        } else if (pid == 0) {
            // create own process group and communicate
            ProcessGroup::GetInstance("127.0.0.1", 2222, "127.0.0.1", i, process_num, false);
            DummyQuotaManager dummyQuotaMgr;
            SpendAllQuota(dummyQuotaMgr, false, i, process_num);
            return;
        }
    }
    ProcessGroup::GetInstance("127.0.0.1", 2222, "127.0.0.1", 0, process_num, true);
    DummyQuotaManager dummyQuotaMgr;
    SpendAllQuota(dummyQuotaMgr, true, 0, process_num);
}

TEST_F(DummyQuotaManagerTest, DISABLED_TestRealRequestAndDummy)
{
    int process_num = 2;
    int quota = 1;
    int timeoutInSeconds = 1;
    for (int i = 1; i < process_num; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            throw runtime_error("fork failed");
        } else if (pid == 0) {
            // create own process group and communicate
            ProcessGroup::GetInstance("127.0.0.1", 2222, "127.0.0.1", i, process_num, false, timeoutInSeconds);
            DummyQuotaManager dummyQuotaMgr(quota);
            RealReqWithDummy(dummyQuotaMgr, false);
            return;
        }
    }
    ProcessGroup::GetInstance("127.0.0.1", 2222, "127.0.0.1", 0, process_num, true, timeoutInSeconds);
    DummyQuotaManager dummyQuotaMgr(quota);
    RealReqWithDummy(dummyQuotaMgr, true);
}

TEST_F(DummyQuotaManagerTest, RandomPressureTest)
{
    int process_num = 2;
    int rounds = 1;
    int quota = 2;
    int timeoutInSeconds = 1;
    vector<pid_t> pids;
    for (int i = 1; i < process_num; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            throw runtime_error("fork failed");
        } else if (pid == 0) {
            // create own process group and communicate
            ProcessGroup::GetInstance("127.0.0.1", 2222, "127.0.0.1", i, process_num, false, timeoutInSeconds);
            DummyQuotaManager dummyQuotaMgr(quota, i);
            RandomPressure(dummyQuotaMgr, rounds);
            cout << "child process exit" << endl;
            exit(0);
        } else {
            pids.push_back(pid);
        }
    }
    ProcessGroup::GetInstance("127.0.0.1", 2222, "127.0.0.1", 0, process_num, true, timeoutInSeconds);
    DummyQuotaManager dummyQuotaMgr(quota, 0);
    RandomPressure(dummyQuotaMgr, rounds);

    for (int i = 0; i < process_num - 1; i++) {
        WaitOrKillCldProcess(pids[i]);
    }
    cout << "parent process exit" << endl;
}