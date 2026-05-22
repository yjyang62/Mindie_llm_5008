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
#include <gtest/gtest.h>
#include <string>
#include "dmi_msg_receiver.h"

using namespace prefillAndDecodeCommunication;

namespace mindie_llm {

class DmiMsgReceiverTest : public testing::Test {
protected:
    void SetUp() override
    {
        decodeRequestReceiver_ = std::make_unique<DecodeRequestReceiver>("127.0.0.1:50051");
        kvReleaseReceiver_ = std::make_unique<KvReleaseReceiver>("127.0.0.1:50051");
    }

    void TearDown() override
    {
    }

    std::unique_ptr<DecodeRequestReceiver> decodeRequestReceiver_;
    std::unique_ptr<KvReleaseReceiver> kvReleaseReceiver_;
};

TEST_F(DmiMsgReceiverTest, DecodeRequestReceiver_InvalidRequestReturnsCancelled)
{
    grpc::ServerContext context_;
    
    {
        DecodeRequestResponse response;
        grpc::Status status = decodeRequestReceiver_->DecodeRequestChannel(
            &context_, nullptr, &response
        );

        EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
        EXPECT_FALSE(response.isvaliddecodeparameters());
        EXPECT_EQ(response.errormessage(), "Request is nullptr");
    }

    {
        DecodeParameters invalid_request;
        grpc::Status status = decodeRequestReceiver_->DecodeRequestChannel(
            &context_, &invalid_request, nullptr
        );
        EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    }

    {
        DecodeParameters invalid_request;
        DecodeRequestResponse response;
        invalid_request.set_maxnewtoken(-1);
        
        grpc::Status status = decodeRequestReceiver_->DecodeRequestChannel(
            &context_, &invalid_request, &response
        );

        EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
        EXPECT_FALSE(response.isvaliddecodeparameters());
        EXPECT_EQ(response.errormessage(), "MaxOutPutLen is invalid");
    }
}

TEST_F(DmiMsgReceiverTest, DecodeRequestReceiver_DecodeRequestHandler_IsNull)
{
    grpc::ServerContext context_;
    DecodeParameters invalid_request;
    DecodeRequestResponse response;
    grpc::Status status = decodeRequestReceiver_->DecodeRequestChannel(
            &context_, &invalid_request, &response
        );

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_TRUE(response.errormessage().empty());
}

TEST_F(DmiMsgReceiverTest, DecodeRequestReceiver_ValidRequestCallsRegisteredHandler)
{
    grpc::ServerContext context_;
    DecodeParameters valid_request;
    DecodeRequestResponse response;
    
    bool handlerCalled = false;
    DecodeRequestHandler mockHandler = [&](const DecodeParameters& req,
                                           DecodeRequestResponse& res) {
        handlerCalled = true;

        res.set_isvaliddecodeparameters(true);
        res.set_errormessage("Request processed");
    };

    EXPECT_FALSE(decodeRequestReceiver_->RegisterMsgHandler(nullptr));
    EXPECT_TRUE(decodeRequestReceiver_->RegisterMsgHandler(mockHandler));
    
    grpc::Status status = decodeRequestReceiver_->DecodeRequestChannel(
        &context_, &valid_request, &response
    );
    
    EXPECT_EQ(status.error_code(), grpc::StatusCode::OK);
    EXPECT_TRUE(handlerCalled);
    EXPECT_TRUE(response.isvaliddecodeparameters());
    EXPECT_EQ(response.errormessage(), "Request processed");
}

TEST_F(DmiMsgReceiverTest, KvReleaseReceiver_InvalidRequestReturnsCancelled)
{
    grpc::ServerContext context_;

    google::protobuf::Empty response;
    grpc::Status status = kvReleaseReceiver_->ReleaseKVCacheChannel(
        &context_, nullptr, &response
    );

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
}

TEST_F(DmiMsgReceiverTest, KvReleaseReceiver_DecodeRequestHandler_IsNull)
{
    grpc::ServerContext context_;
    RequestId invalid_request;
    google::protobuf::Empty response;
    grpc::Status status = kvReleaseReceiver_->ReleaseKVCacheChannel(
            &context_, &invalid_request, &response
        );

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
}

TEST_F(DmiMsgReceiverTest, KvReleaseReceiver_ValidRequestCallsRegisteredHandler)
{
    grpc::ServerContext context_;
    RequestId valid_request;
    google::protobuf::Empty response;
    valid_request.set_reqid("test");
    bool handlerCalled = false;
    std::mutex mtx;
    std::condition_variable cv;

    KVReleaseHandler mockHandler = [&](const std::string& requestID) {
        std::lock_guard<std::mutex> lock(mtx);
        handlerCalled = true;
        EXPECT_EQ(requestID, "test");
        cv.notify_one();
    };

    EXPECT_FALSE(kvReleaseReceiver_->RegisterMsgHandler(nullptr));
    EXPECT_TRUE(kvReleaseReceiver_->RegisterMsgHandler(mockHandler));
    
    grpc::Status status = kvReleaseReceiver_->ReleaseKVCacheChannel(
        &context_, &valid_request, &response
    );
    
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait_for(lock, std::chrono::seconds(1), [&] { return handlerCalled; });
    }

    EXPECT_EQ(status.error_code(), grpc::StatusCode::OK);
    EXPECT_TRUE(handlerCalled);
}

} // namespace endpoint