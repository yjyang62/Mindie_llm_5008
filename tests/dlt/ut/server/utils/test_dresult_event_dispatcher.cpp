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
#include "dresult_event_dispatcher.h"

using namespace mindie_llm;

class DResultEventDispatcherTest : public testing::Test {
protected:
    void SetUp() override
    {
        dispatcher = std::make_shared<DResultEventDispatcher>();
        
        sink.write = [this](const char* data, size_t size) -> bool {
            received_data.append(data, size);
            return true;
        };
        
        sink.done = [this]() {
            done_called = true;
            return;
        };
    }

    void TearDown() override
    {
        if (test_thread.joinable()) {
            dispatcher->isDestroyed_ = true;
            dispatcher->cv_.notify_all();
            test_thread.join();
        }
        GlobalMockObject::verify();
    }

    std::shared_ptr<DResultEventDispatcher> dispatcher;
    httplib::DataSink sink;
    std::string received_data;
    bool done_called = false;
    std::thread test_thread;
};

TEST_F(DResultEventDispatcherTest, WriteStreamMessage_Normal)
{
    auto msg1 = new std::string("Message1");
    auto msg2 = new std::string("Message2");
    dispatcher->queue_.push(msg1);
    dispatcher->queue_.push(msg2);
    
    dispatcher->WriteStreamMessage(&sink);
    
    EXPECT_EQ(received_data, "Message1Message2");
    EXPECT_FALSE(done_called);
}

TEST_F(DResultEventDispatcherTest, WriteStreamMessage_CallsDoneWhenFinished)
{
    dispatcher->isFinish_ = true;
    dispatcher->WriteStreamMessage(&sink);
    EXPECT_TRUE(done_called);
}

TEST_F(DResultEventDispatcherTest, WriteStreamMessage_ExceptionHandling)
{
    httplib::DataSink exceptionSink;
    exceptionSink.write = [](const char*, size_t) -> bool {
        throw std::runtime_error("Write error");
    };

    auto msg = new std::string("Test message");
    dispatcher->queue_.push(msg);
    
    EXPECT_THROW(dispatcher->WriteStreamMessage(&exceptionSink), std::runtime_error);
    
    std::string* dummy;
    EXPECT_FALSE(dispatcher->queue_.pop(dummy));
}

TEST_F(DResultEventDispatcherTest, WaitEvent_NormalOperation)
{

    auto msg = new std::string("WaitEvent message");
    dispatcher->queue_.push(msg);

    test_thread = std::thread([this]() {
        dispatcher->WaitEvent(&sink);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (test_thread.joinable()) {
        dispatcher->isDestroyed_ = true;
        dispatcher->cv_.notify_all();
        test_thread.join();
    }
    EXPECT_EQ(received_data, "WaitEvent message");
}

static void ThrowWriteError(httplib::DataSink* sink)
{
    throw std::runtime_error("Write error");
}

TEST_F(DResultEventDispatcherTest, InitialIntervalIsZero)
{
    auto interval = dispatcher->GetIntervalFromPrevSend();
    auto ns = interval.count();
    EXPECT_LT(ns, 100000000);
}