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

#include "test_llm_manager_adapter.h"
#include "memory_utils.h"

namespace mindie_llm {

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubInputIds()
{
    std::vector<std::shared_ptr<InferRequest>> requests;

    InferRequestId requestId("req_1");
    auto req = std::make_shared<InferRequest>(requestId);

    std::vector<int64_t> inferTokens = {1, 2, 3};
    std::vector<int64_t> shape = {1, static_cast<int64_t>(inferTokens.size())};
    auto tensor = std::make_shared<InferTensor>("INPUT_IDS", InferDataType::TYPE_INT64, shape);
    tensor->Allocate(inferTokens.size() * sizeof(int64_t));
    std::copy(inferTokens.begin(), inferTokens.end(), static_cast<int64_t *>(tensor->data));
    req->AddTensor("INPUT_IDS", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubLoraId()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    auto req = std::make_shared<InferRequest>(requestId);
    std::string loraId = "test_lora_123";
    std::vector<int64_t> shape = {1, static_cast<int64_t>(loraId.size())};
    auto tensor = std::make_shared<InferTensor>("LORA_ID", InferDataType::TYPE_STRING, shape);
    tensor->Allocate((loraId.size() + 1) * sizeof(char));
    auto *buffer = static_cast<char *>(tensor->data);
    auto ret = strcpy_s(buffer, loraId.size() + 1, loraId.c_str());
    if (ret != 0) {
        std::cout << "Failed to strcpy_s data for lora ids" << std::endl;
        return requests;
    }
    req->AddTensor("LORA_ID", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubIgnoreEos()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    bool ignoreEos = true;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("IGNORE_EOS", InferDataType::TYPE_BOOL, shape);
    tensor->Allocate(paramNum * sizeof(bool));
    auto *buffer = static_cast<bool *>(tensor->data);
    buffer[0u] = ignoreEos;
    req->AddTensor("IGNORE_EOS", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubStopStrings()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    auto req = std::make_shared<InferRequest>(requestId);
    std::string stopStrings = "test_stopStrings_123";
    std::vector<int64_t> shape = {1, static_cast<int64_t>(stopStrings.size())};
    auto tensor = std::make_shared<InferTensor>("STOP_STRINGS", InferDataType::TYPE_STRING, shape);
    tensor->Allocate((stopStrings.size() + 1) * sizeof(char));
    auto *buffer = static_cast<char *>(tensor->data);
    auto ret = strcpy_s(buffer, stopStrings.size() + 1, stopStrings.c_str());
    if (ret != 0) {
        std::cout << "Failed to strcpy_s data for STOP_STRINGS" << std::endl;
        return requests;
    }
    req->AddTensor("STOP_STRINGS", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubLogProbs()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    bool logProbs = true;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("LOGPROBS", InferDataType::TYPE_BOOL, shape);
    tensor->Allocate(paramNum * sizeof(bool));
    auto *buffer = static_cast<bool *>(tensor->data);
    buffer[0u] = logProbs;
    req->AddTensor("LOGPROBS", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTopLogProbs()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    uint32_t topLogProbs = 123;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("TOP_LOGPROBS", InferDataType::TYPE_UINT32, shape);
    tensor->Allocate(paramNum * sizeof(uint32_t));
    auto *buffer = static_cast<uint32_t *>(tensor->data);
    buffer[0u] = topLogProbs;
    req->AddTensor("TOP_LOGPROBS", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTemperature()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    float temperature = 123.123f;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("TEMPERATURE", InferDataType::TYPE_FP32, shape);
    tensor->Allocate(paramNum * sizeof(float));
    auto *buffer = static_cast<float *>(tensor->data);
    buffer[0u] = temperature;
    req->AddTensor("TEMPERATURE", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTopK()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    int32_t topK = 12;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("TOP_K", InferDataType::TYPE_INT32, shape);
    tensor->Allocate(paramNum * sizeof(int32_t));
    auto *buffer = static_cast<int32_t *>(tensor->data);
    buffer[0u] = topK;
    req->AddTensor("TOP_K", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTopP()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    float topP = 12.12f;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("TOP_P", InferDataType::TYPE_FP32, shape);
    tensor->Allocate(paramNum * sizeof(float));
    auto *buffer = static_cast<float *>(tensor->data);
    buffer[0u] = topP;
    req->AddTensor("TOP_P", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTypicalP()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    float typicalP = 1.1f;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("TYPICAL_P", InferDataType::TYPE_FP32, shape);
    tensor->Allocate(paramNum * sizeof(float));
    auto *buffer = static_cast<float *>(tensor->data);
    buffer[0u] = typicalP;
    req->AddTensor("TYPICAL_P", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubDoSample()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    bool doSample = true;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("DO_SAMPLE", InferDataType::TYPE_BOOL, shape);
    tensor->Allocate(paramNum * sizeof(bool));
    auto *buffer = static_cast<bool *>(tensor->data);
    buffer[0u] = doSample;
    req->AddTensor("DO_SAMPLE", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubSeed()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    uint64_t seed = 123;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("SEED", InferDataType::TYPE_UINT64, shape);
    tensor->Allocate(paramNum * sizeof(uint64_t));
    auto *buffer = static_cast<uint64_t *>(tensor->data);
    buffer[0u] = seed;
    req->AddTensor("SEED", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubRepetitionPenalty()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    float repetitionPenalty = 1.2f;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("REPETITION_PENALTY", InferDataType::TYPE_FP32, shape);
    tensor->Allocate(paramNum * sizeof(float));
    auto *buffer = static_cast<float *>(tensor->data);
    buffer[0u] = repetitionPenalty;
    req->AddTensor("REPETITION_PENALTY", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubFrequencyPenalty()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    float frequencyPenalty = 1.3f;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("FREQUENCY_PENALTY", InferDataType::TYPE_FP32, shape);
    tensor->Allocate(paramNum * sizeof(float));
    auto *buffer = static_cast<float *>(tensor->data);
    buffer[0u] = frequencyPenalty;
    req->AddTensor("FREQUENCY_PENALTY", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubPresencyPenalty()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    float presencyPenalty = 1.4f;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("PRESENCE_PENALTY", InferDataType::TYPE_FP32, shape);
    tensor->Allocate(paramNum * sizeof(float));
    auto *buffer = static_cast<float *>(tensor->data);
    buffer[0u] = presencyPenalty;
    req->AddTensor("PRESENCE_PENALTY", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubIncludeStopStrInOutput()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    bool includeStopStrInOutput = true;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("INCLUDE_STOP_STR_IN_OUTPUT", InferDataType::TYPE_BOOL, shape);
    tensor->Allocate(paramNum * sizeof(bool));
    auto *buffer = static_cast<bool *>(tensor->data);
    buffer[0u] = includeStopStrInOutput;
    req->AddTensor("INCLUDE_STOP_STR_IN_OUTPUT", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubWatermark()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    bool watermark = true;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("WATERMARK", InferDataType::TYPE_BOOL, shape);
    tensor->Allocate(paramNum * sizeof(bool));
    auto *buffer = static_cast<bool *>(tensor->data);
    buffer[0u] = watermark;
    req->AddTensor("WATERMARK", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubN()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    uint32_t n = 1234;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("N", InferDataType::TYPE_UINT32, shape);
    tensor->Allocate(paramNum * sizeof(uint32_t));
    auto *buffer = static_cast<uint32_t *>(tensor->data);
    buffer[0u] = n;
    req->AddTensor("N", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubBestOf()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    uint32_t bestOf = 1234;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("BEST_OF", InferDataType::TYPE_UINT32, shape);
    tensor->Allocate(paramNum * sizeof(uint32_t));
    auto *buffer = static_cast<uint32_t *>(tensor->data);
    buffer[0u] = bestOf;
    req->AddTensor("BEST_OF", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubUseBeamSearch()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    bool useBeamSearch = true;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("USE_BEAM_SEARCH", InferDataType::TYPE_BOOL, shape);
    tensor->Allocate(paramNum * sizeof(bool));
    auto *buffer = static_cast<bool *>(tensor->data);
    buffer[0u] = useBeamSearch;
    req->AddTensor("USE_BEAM_SEARCH", tensor);

    requests.emplace_back(req);
    return requests;
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStubLengthPenalty()
{
    std::vector<std::shared_ptr<InferRequest>> requests;
    InferRequestId requestId("req_1");
    const int64_t paramNum = 1;
    auto req = std::make_shared<InferRequest>(requestId);
    float lengthPenalty = 1.5f;
    std::vector<int64_t> shape = {1, paramNum};
    auto tensor = std::make_shared<InferTensor>("LENGTH_PENALTY", InferDataType::TYPE_FP32, shape);
    tensor->Allocate(paramNum * sizeof(float));
    auto *buffer = static_cast<float *>(tensor->data);
    buffer[0u] = lengthPenalty;
    req->AddTensor("LENGTH_PENALTY", tensor);

    requests.emplace_back(req);
    return requests;
}

}