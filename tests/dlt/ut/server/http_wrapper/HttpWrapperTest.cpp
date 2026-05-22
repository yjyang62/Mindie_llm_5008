/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <vector>
#include <thread>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "http_wrapper.h"
#include "http_server.h"

using namespace mindie_llm;

namespace mindie_llm {
class HttpWrapperTest : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(HttpWrapperTest, StartSuccess)
{
    MOCKER(HttpServer::HttpServerInit).stubs().will(returnValue(0));
    bool ret = HttpWrapper::Instance().Start();
    EXPECT_EQ(ret, true);
    HttpWrapper::Instance().Stop();
}

TEST_F(HttpWrapperTest, StartFail)
{
    MOCKER(HttpServer::HttpServerInit).stubs().will(returnValue(1U));
    bool ret = HttpWrapper::Instance().Start();
    EXPECT_EQ(ret, false);
    HttpWrapper::Instance().Stop();
}

} // namespace llm