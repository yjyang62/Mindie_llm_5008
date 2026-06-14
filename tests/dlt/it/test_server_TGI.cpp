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
#include <mockcpp/mockcpp.hpp>
#include "http_rest_resource.h"
#define private public
#include "single_req_tgi_text_infer_interface.h"
#include "single_llm_pnd_req_handler.h"
#include "infer_instances.h"
#include "response.h"
#include "llm_manager_impl.h"
#include "llm_manager_v2.h"
#include "llm_engine.h"
#include "infer_tokenizer.h"
#include "endpoint_def.h"
#include "executor.h"
#include "base64_util.h"

namespace mindie_llm {
extern uint32_t g_vocabSizeConfig;
extern uint32_t g_maxPositionEmbeddings;
extern uint32_t g_maxSeqLen;
extern uint32_t g_maxInputTokenLen;
extern uint32_t g_maxTopKConfig;


#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))
static Status MockEncodeSuccess(TokenizerProcessPool *pool, const std::string &prompt, std::vector<int64_t> &tokenIds,
                                HeadFlag flag, uint64_t &timestamp)
{
    tokenIds = {1, 2, 3};
    return Status(Error::Code::OK, "Success");
}
static void EmptyMock(const std::string &) {}
class TGIInferTestF : public ::testing::Test {
protected:
    std::map<std::string, std::string> modelConfig;
    std::map<std::string, std::string> ipInfo;
    SchedulerConfig config;
    Role pdRole;
    std::shared_ptr<InferInstance> inferInstance;
    void SetUp() override
    {
        MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool *, const std::string &,
                                                            std::vector<int64_t> &, HeadFlag, uint64_t &, const bool))
            .stubs()
            .will(invoke(&MockEncodeSuccess));

        MOCKER_CPP(&SingleLLMPnDReqHandler::ProcessNonStreamModeRequest, void (*)(const std::string &))
            .stubs()
            .will(invoke(&EmptyMock));

        modelConfig["configPath"] = "";
        modelConfig["npuDeviceIds"] = "0";
        modelConfig["inferMode"] = "standard";
        ipInfo = {{"infer_mode", "standard"}};

        config.cacheBlockSize = 128;
        config.npuBlockNum = 1024;
        config.cpuBlockNum = 0;
        config.policyType = 0;
        config.maxSeqLen = 1024;
        config.maxIterTimes = 512;
        config.dpSize = 1;
        config.enablePrefixCache = false;
        config.enableSplit = false;
        config.prefillPolicyType = 0;
        config.decodePolicyType = 0;

        pdRole = Role::PnD;

        g_vocabSizeConfig = 1024;
        g_maxPositionEmbeddings = 1024;
        g_maxSeqLen = 1024;
        g_maxInputTokenLen = 1024;
        g_maxTopKConfig = 1024;
        inferInstance = GetInferInstance();
        inferInstance->started_ = false; // make sure not go through InferInstance::Forward
    }
};

TEST_F(TGIInferTestF, ShouldSeqGroupMatchParametersInRequest)
{
    httplib::Request request;
    httplib::Response response;
    request.method = "mockMethod";
    request.path = "mockPath";
    request.version = "mockVersion";
    request.body = R"({"inputs": "Please introduce yourself.", "parameters": {"max_new_tokens": 30,
    "stop": ["stop1", "stop2"], "temperature": 1.5, "top_k": 10, "top_p": 0.5, "typical_p": 0.5,
    "do_sample": true, "seed": 1, "watermark": true, "adapter_id": "lora-123"}})";
    std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
    std::shared_ptr<SingleLLMPnDReqHandler> singleLLMPnDReqHandler = std::make_shared<SingleLLMPnDReqHandler>(context);
    bool isReCompute = false;
    auto infer = std::make_shared<SingleReqTgiTextInferInterface>(singleLLMPnDReqHandler, isReCompute);
    // create request and process
    infer->Process();

    // prepare for InferInstance::Forward
    std::shared_ptr<LlmManagerV2> llmManager =
        std::make_shared<LlmManagerV2>("", nullptr, nullptr, nullptr, nullptr, nullptr, ipInfo);
    inferInstance->llmManagers_.push_back(llmManager);

    inferInstance->started_ = true;
    // imitate InferInstance::Forward
    // prepare for LlmManagerImpl::ProcessRequests
    std::vector<IExecutorSPtr> executors;
    executors.push_back(std::make_shared<Executor>());

    inferInstance->llmManagers_[0]->impl_->llmEnginePtr_ = MakeLlmEngine(config, executors, nullptr, pdRole);
    // add request to llmEngine
    auto llmInferRequest = infer->request_;
    inferInstance->llmManagers_[0]->AddRequest(llmInferRequest);

    // compare maxOutputLen_ value in seqGroup with max_new_tokens
    //  maxOutputLen = min(maxIterTimes, maxSeqLen - promptToken, max_new_tokens)
    uint64_t maxOutputLen =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->maxOutputLen;

    EXPECT_EQ(maxOutputLen, 30);
    std::string stop = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->stopStrings;
    std::string trueStop = Base64Util::Encode("[\"stop1\",\"stop2\"]");
    EXPECT_EQ(stop, trueStop);

    float trueTemperature = 1.5f;
    float temperature =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->temperature.value();
    EXPECT_EQ(temperature, trueTemperature);

    int32_t trueTopK = 10;
    int32_t topK = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->topK.value();
    EXPECT_EQ(topK, trueTopK);

    float trueTopP = 0.5f;
    float topP = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->topP.value();
    EXPECT_EQ(topP, trueTopP);

    float typicalP = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->typicalP.value();
    EXPECT_EQ(typicalP, 0.5f);

    bool doSample = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->doSample.value();
    EXPECT_EQ(doSample, true);

    int64_t seed = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->seed.value();
    EXPECT_EQ(seed, 1);

    bool watermark = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->watermark.value();
    EXPECT_EQ(watermark, true);

    std::string adapterId = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->loraId_.value();
    EXPECT_EQ(adapterId, "lora-123");
}

TEST_F(TGIInferTestF, ShouldSeqGroupReturnDefaultValuesWhenParametersNotSetInRequest)
{
    httplib::Request request;
    httplib::Response response;
    request.method = "mockMethod";
    request.path = "mockPath";
    request.version = "mockVersion";
    request.body = R"({"inputs":"Please introduce yourself."})";

    std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
    std::shared_ptr<SingleLLMPnDReqHandler> singleLLMPnDReqHandler = std::make_shared<SingleLLMPnDReqHandler>(context);
    bool isReCompute = false;
    auto infer = std::make_shared<SingleReqTgiTextInferInterface>(singleLLMPnDReqHandler, isReCompute);
    // create request and process
    infer->Process();

    // prepare for InferInstance::Forward
    std::shared_ptr<LlmManagerV2> llmManager =
        std::make_shared<LlmManagerV2>("", nullptr, nullptr, nullptr, nullptr, nullptr, ipInfo);
    inferInstance->llmManagers_.push_back(llmManager);

    inferInstance->started_ = true;
    // imitate InferInstance::Forward
    // prepare for LlmManagerImpl::ProcessRequests
    std::vector<IExecutorSPtr> executors;
    executors.push_back(std::make_shared<Executor>());

    inferInstance->llmManagers_[0]->impl_->llmEnginePtr_ = MakeLlmEngine(config, executors, nullptr, pdRole);
    // add request to llmEngine
    auto llmInferRequest = infer->request_;
    inferInstance->llmManagers_[0]->AddRequest(llmInferRequest);
 
    // compare maxOutputLen_ value in seqGroup with max_new_tokens
    uint64_t maxOutputLen =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->maxOutputLen;

    EXPECT_EQ(maxOutputLen, 20); // 20 == MAX_NEW_TOKENS_DFT

    int64_t seed = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->seed.value();
    // default random seed value
    EXPECT_GT(seed, 0);
    EXPECT_LE(seed, 184467440737095615);
}

} // namespace mindie_llm