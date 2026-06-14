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
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <thread>

#include "http_rest_resource.h"
#include "threadpool_monitor.h"

namespace mindie_llm {
class ThreadPoolMonitorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto maxLinkNum = 4;
        auto maxQueueRequestsNum = 10;
        pool_ = std::make_unique<ThreadPoolMonitor>(maxLinkNum, maxQueueRequestsNum);
    }

    void TearDown() override
    {
        if (pool_) {
            pool_->shutdown();
        }
    }

    std::unique_ptr<ThreadPoolMonitor> pool_;
};

TEST_F(ThreadPoolMonitorTest, TestEnqueueJob)
{
    // Test enqueue functionality: add a job and ensure it's processed
    std::promise<void> promise;
    std::future<void> future = promise.get_future();

    pool_->enqueue([&]() {
        promise.set_value(); // Set promise when the task is done
    });

    EXPECT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
}

TEST_F(ThreadPoolMonitorTest, TestAddAndRemoveRequestToMonitor)
{
    httplib::Request request;
    httplib::Response response;
    auto waitDuration = 100;
    std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
    pool_->AddRequestToMonitor(context);
    std::this_thread::sleep_for(std::chrono::milliseconds(waitDuration));

    auto reqId = response.get_header_value("RequestUUID");
    pool_->RemoveMonitorRequest(reqId);
}

TEST_F(ThreadPoolMonitorTest, TestCheckAndRemoveClosedConnections)
{
    httplib::Request request;
    httplib::Response response;
    auto waitDuration = 100;
    std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
    std::map<std::string, std::shared_ptr<RequestContext>> monitorRequests;
    monitorRequests.insert({"test", context});
    pool_->CheckAndRemoveClosedConnections();
}

}