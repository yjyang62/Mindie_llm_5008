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
#include <mockcpp/mockcpp.hpp>
#define private public
#include "event_dispatcher.h"

using namespace mindie_llm;

class EventDispatcherTest : public testing::Test {
protected:
    void SetUp() override
    {
        dispatcher = std::make_shared<EventDispatcher>(123, "test-request", 3);
        sink.write = [&](const char* data, size_t size) -> bool {
            std::cout << "write called with size: " << size << std::endl;
            received_data.append(data, size);
            return true;
        };
        sink.done = [&]() {
            done_called = true;
            std::cout << "done called" << std::endl;
        };
    }

    void TearDown() override
    {
        received_data = "";
        done_called = false;
        GlobalMockObject::verify();
    }

    std::shared_ptr<EventDispatcher> dispatcher;
    httplib::DataSink sink;
    std::string received_data;
    bool done_called = false;

    boost::cv_status (boost::condition_variable::*wait_until_ptr)(
            boost::unique_lock<boost::mutex>&,
            const boost::chrono::steady_clock::time_point&
        ) = &boost::condition_variable::wait_until;
    
};

TEST_F(EventDispatcherTest, NormalEventFlow)
{
    MOCKER_CPP(wait_until_ptr, boost::cv_status (*)(
            boost::unique_lock<boost::mutex>&,
            const boost::chrono::steady_clock::time_point&
        )).stubs().will(MOCKCPP_NS::returnValue(boost::cv_status::no_timeout));
    boost::thread wait_thread([&] {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        dispatcher->WaitEvent(&sink);
    });
    
    dispatcher->SendEvent("Hello ", false);
    dispatcher->SendEvent("World", true);
    
    wait_thread.join();
    EXPECT_EQ(received_data, "Hello World");
    EXPECT_TRUE(dispatcher->isFinish_.load());
}

TEST_F(EventDispatcherTest, TimeoutScenario)
{
    MOCKER_CPP(wait_until_ptr, boost::cv_status (*)(
            boost::unique_lock<boost::mutex>&,
            const boost::chrono::steady_clock::time_point&
        )).stubs().will(MOCKCPP_NS::returnValue(boost::cv_status::timeout));
    boost::thread wait_thread([&] {
        dispatcher->WaitEvent(&sink);
    });
    
    wait_thread.join();
    
    EXPECT_EQ(received_data, "Engine callback timeout.");
    EXPECT_TRUE(done_called);
    EXPECT_TRUE(dispatcher->isFinish_.load());
}

TEST_F(EventDispatcherTest, ClearFunctionality)
{
    boost::thread wait_thread([&] {
        dispatcher->WaitEvent(&sink);
    });
    
    dispatcher->SendEvent("Test data", false);
    
    while (dispatcher->sendCount_.load() == 0) {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    }
    
    dispatcher->Clear();
    
    
    dispatcher->SendEvent("", true);
    wait_thread.join();
    EXPECT_EQ(dispatcher->clearCount_.load(), 0);
}

TEST_F(EventDispatcherTest, LastMessageHandling)
{
    MOCKER_CPP(wait_until_ptr, boost::cv_status (*)(
            boost::unique_lock<boost::mutex>&,
            const boost::chrono::steady_clock::time_point&
        )).stubs().will(MOCKCPP_NS::returnValue(boost::cv_status::no_timeout));
    boost::thread wait_thread([&] {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(500));
        dispatcher->WaitEvent(&sink);
    });
    
    dispatcher->SendEvent("Main message", true, " Last part");
    
    wait_thread.join();
    
    EXPECT_EQ(received_data, "Main message Last part");
    EXPECT_TRUE(done_called);
}

TEST_F(EventDispatcherTest, LogAndRequestIds)
{
    EXPECT_EQ(dispatcher->logId, 123);
    EXPECT_EQ(dispatcher->requestId_, "test-request");
}
