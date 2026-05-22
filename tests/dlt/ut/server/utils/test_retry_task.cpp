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
#include "retry_task.h"

using namespace mindie_llm;

class RetryTaskTest : public testing::Test {
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(RetryTaskTest, Execute)
{
    auto ioContext = std::make_shared<boost::asio::io_context>();

    int count = 1;
    RetryTask::RetryFunc func = [&](GlobalIpInfo& info) {
        count++;
        info.needInit = true;
    };
    GlobalIpInfo globalIpInfo;
    RetryTask task(func, ioContext, globalIpInfo);
    task.Execute();
    EXPECT_EQ(count, 2);
}