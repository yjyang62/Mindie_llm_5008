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
#include "https_server_helper.h"
#include "threadpool_monitor.h"

using namespace mindie_llm;

class HttpsServerHelperTest : public testing::Test {
protected:
    void SetUp() override
    {
        setupSslCtxCallback = [](SSL_CTX& sslCtx) -> bool {
            return true;
        };
        helper = std::make_shared<HttpsServerHelper>(false, setupSslCtxCallback, 1000);
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }

    std::shared_ptr<HttpsServerHelper> helper;
    std::function<bool(SSL_CTX& sslCtx)> setupSslCtxCallback;
};

TEST_F(HttpsServerHelperTest, IsValid_WhenCtxIsNull_ReturnsFalse)
{
    helper->ctx_ = nullptr;
    EXPECT_FALSE(helper->is_valid());
}

TEST_F(HttpsServerHelperTest, IsValid_WhenCtxIsNotNull_ReturnsTrue)
{
    EXPECT_TRUE(helper->is_valid());
}

TEST_F(HttpsServerHelperTest, ProcessAndCloseSocket)
{
    helper->openSSL = false;
    EXPECT_FALSE(helper->process_and_close_socket(123));
    helper->openSSL = true;
    EXPECT_FALSE(helper->process_and_close_socket(123));
}

TEST_F(HttpsServerHelperTest, AddRequestToMonitor_CallsThreadPool)
{
    httplib::Request req = httplib::Request();
    httplib::Response resp = httplib::Response();
    auto reqContext = std::make_shared<RequestContext>(req, resp);
    helper->threadpool_ = new (std::nothrow) ThreadPoolMonitor(10, 10);
    helper->AddRequestToMonitor(reqContext);
    std::cout << "1" << std::endl;
    helper->RemoveMonitorRequest("");
}

