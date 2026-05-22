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
#include <httplib.h>
#include <atomic>
#include <memory>
#define private public
#include "http_rest_resource.h"

using namespace mindie_llm;

class HttpRestResourceTest : public testing::Test {
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST(HttpRestResourceTest, ConstructorAndGetters)
{
    httplib::Request req;
    req.method = "GET";
    req.path = "/test";
    req.version = "HTTP/1.1";
    req.body = "test body";
    
    httplib::Response res;
    res.set_header("RequestUUID", "test-uuid");
    
    RequestContext ctx(req, res);
    ctx.Req();
    ctx.Res();
    ctx.SetResponseFinished(true);
    bool flagOut = false;
    ctx.LockAndCheckResponseFinished(flagOut);
    EXPECT_TRUE(flagOut);
    EXPECT_EQ(ctx.Method(), "GET");
    EXPECT_EQ(ctx.Path(), "/test");
    EXPECT_EQ(ctx.Version(), "HTTP/1.1");
    EXPECT_EQ(ctx.MsgBody(), "test body");
    EXPECT_EQ(ctx.GetHTTPRequestUUID(), "test-uuid");
    ctx.SetHTTPRequestUUID("test2");
    EXPECT_EQ(ctx.GetHTTPRequestUUID(), "test2");
    EXPECT_EQ(ctx.InferRequestId(), 1);
}

TEST(HttpRestResourceTest, CallbackIdManagement)
{
    httplib::Request req;
    httplib::Response res;
    RequestContext ctx(req, res);
    
    uint32_t initialId = ctx.CallbackId();
    ctx.SetCallbackId(100);
    EXPECT_EQ(ctx.CallbackId(), 100);
    EXPECT_NE(initialId, 100);
}

TEST(HttpRestResourceTest, StopInferRequest)
{
    httplib::Request req;
    httplib::Response res;
    RequestContext ctx(req, res);
    
    bool handlerCalled = false;
    ctx.SetStopInferFunction([&]() { handlerCalled = true; });
    
    ctx.StopInferRequest();
    EXPECT_TRUE(handlerCalled);
}

TEST(HttpRestResourceTest, ResponseNobody)
{
    httplib::Request req;
    httplib::Response res;
    auto ctx = std::make_shared<RequestContext>(req, res);
    
    int result = HttpRestResource::ResponseNobody(ctx, 204);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(res.status, 204);
    EXPECT_TRUE(res.body.empty());
}

TEST(HttpRestResourceTest, ResponseJsonBody)
{
    httplib::Request req;
    httplib::Response res;
    auto ctx = std::make_shared<RequestContext>(req, res);
    
    int result = HttpRestResource::ResponseJsonBody(ctx, 200, R"({"status":"ok"})");
    EXPECT_EQ(result, 0);
    EXPECT_EQ(res.status, 200);
    EXPECT_EQ(res.get_header_value("Content-Type"), "application/json");
    EXPECT_EQ(res.body, R"({"status":"ok"})");
}

TEST(HttpRestResourceTest, ResponseWithBodyAndHeaders)
{
    httplib::Request req;
    httplib::Response res;
    auto ctx = std::make_shared<RequestContext>(req, res);
    
    auto headers = std::make_shared<std::map<std::string, std::string>>();
    headers->emplace("X-Custom-Header", "value");
    
    int result = HttpRestResource::ResponseWithBody(ctx, 201, "text/plain", "Created", headers);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(res.status, 201);
    EXPECT_EQ(res.get_header_value("Content-Type"), "text/plain");
    EXPECT_EQ(res.get_header_value("X-Custom-Header"), "value");
    EXPECT_EQ(res.body, "Created");
}

TEST(HttpRestResourceTest, WrapperJson)
{
    std::string result = HttpRestResource::WrapperJson("Invalid input", "validation");
    EXPECT_EQ(result, R"({"error":"Invalid input","error_type":"validation"})");
}

TEST(HttpRestResourceTest, WrapperStatusJson)
{
    std::string result = HttpRestResource::WrapperStatusJson("running");
    EXPECT_EQ(result, R"({"status":"running"})");
}

TEST(HttpRestResourceTest, ResponseChunkedBody)
{
    httplib::Request req;
    httplib::Response res;
    auto ctx = std::make_shared<RequestContext>(req, res);
    auto dispatcher = std::make_shared<EventDispatcher>();
    HttpRestResource::ResponseChunkedBody(ctx, dispatcher);
    EXPECT_EQ(res.get_header_value("Content-Type"), "text/event-stream");
}