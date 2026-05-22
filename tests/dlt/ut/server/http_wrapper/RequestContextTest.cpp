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
#include "http_rest_resource.h"
#include "httplib.h"

using namespace mindie_llm;

class RequestContextTest : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(RequestContextTest, testCallbackId)
{
    httplib::Request request;
    httplib::Response response;
    RequestContext reqCtx(request, response);
    uint32_t id = 10;
    reqCtx.SetCallbackId(id);
    uint32_t callbackId = reqCtx.CallbackId();
    EXPECT_EQ(callbackId, id);
}