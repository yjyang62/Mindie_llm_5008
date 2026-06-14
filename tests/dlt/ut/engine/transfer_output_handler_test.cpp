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

#include <gtest/gtest.h>
#include "transfer_output_handler.h"
#include "live_infer_context.h"

using namespace mindie_llm;
using namespace model_execute_data;

class TransferOutputHandlerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        transferOutputHandler_ = std::make_shared<TransferOutputHandler>(
            [this](ResponseSPtr response) { responses_.push_back(response); });
    }

    static PullKVResponseSPtr CreatePullKVResponse()
    {
        PullKVResponse response;

        auto *success_result = response.add_pull_kv_results();
        success_result->set_request_id("1");
        success_result->set_pd_error_code(PDErrorCode::SUCCESS);

        auto *failed_result = response.add_pull_kv_results();
        failed_result->set_request_id("2");
        failed_result->set_pd_error_code(PDErrorCode::PD_PULL_KV_ERROR);

        for (int i = 0; i < 2; ++i) {
            SequenceSPtr seqSPtr = std::make_shared<Sequence>(i + 1, 0);
            std::vector<SequenceSPtr> seqs = {seqSPtr};
            RequestId reqId = std::to_string(i + 1);
            SequenceGroupSPtr seqGrpSPtr = std::make_shared<SequenceGroup>(reqId, seqs);
            seqGrpSPtr->metrics_.inferReqId_ = reqId;
            LiveInferContext::GetInstance(0)->Add(seqGrpSPtr);
        }

        return std::make_shared<PullKVResponse>(response);
    }

    std::shared_ptr<TransferOutputHandler> transferOutputHandler_;
    std::vector<ResponseSPtr> responses_;
};

TEST_F(TransferOutputHandlerTest, TestEntry4Executor)
{
    PullKVResponseSPtr pullKVResponse = CreatePullKVResponse();
    transferOutputHandler_->Entry4Executor(pullKVResponse);
    EXPECT_EQ(transferOutputHandler_->GetPulledReqIds().Size(), 1);
    EXPECT_EQ(transferOutputHandler_->GetPulledReqIds().Front(), "1");
    EXPECT_EQ(responses_.size(), 2);

    ResponseSPtr response = responses_.at(0); // success response
    EXPECT_EQ(response->transferStatusFlag, TransferStatusType::PULL_KV_COMPLETE);
    EXPECT_EQ(response->inferStatusFlag, InferStatusType::ITERATION_CONTINUE);
    EXPECT_FALSE(response->isEos);
    EXPECT_EQ(response->responseContents[0].pdErrorCode, static_cast<uint16_t>(PDErrorCode::SUCCESS));

    response = responses_.at(1); // failed response
    EXPECT_EQ(response->transferStatusFlag, TransferStatusType::PULL_KV_COMPLETE);
    EXPECT_EQ(response->inferStatusFlag, InferStatusType::PULL_KV_ERROR);
    EXPECT_TRUE(response->isEos);
    EXPECT_EQ(response->responseContents[0].pdErrorCode, static_cast<uint16_t>(PDErrorCode::PD_PULL_KV_ERROR));
}