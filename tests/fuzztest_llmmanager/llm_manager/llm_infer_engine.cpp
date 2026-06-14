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
#include "llm_infer_engine.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <mutex>
#include <shared_mutex>

using Json = nlohmann::json;
namespace mindie_llm {
LlmInferEngine::~LlmInferEngine() = default;

bool LlmInferEngine::Create(const std::string &configPath, std::map<std::string, std::string> ipInfo)
{
    getRequestsCallback_ = [this]() {
        std::vector<std::shared_ptr<mindie_llm::InferRequest>> requests{};
        std::unique_lock<std::mutex> lock(forwardMutex_);
        while (!this->requestQueue_.empty()) {
            requests.push_back(this->requestQueue_.front());
            this->requestQueue_.pop();
        }
        return requests;
    };

    sendResponsesCallback_ = [this](mindie_llm::InferRequestId reqId, const mindie_llm::TensorMap &output, bool isFinal,
                                    const std::string &errMsg) {};
    //  Control signal callback, Get stop list from the queue
    controlSignalCallback_ = [this]() {
        std::vector<std::pair<mindie_llm::InferRequestId, mindie_llm::OperationV2>> stopLists;
        while (!this->stopQueue_.empty()) {
            stopLists.push_back(this->stopQueue_.front());
            this->stopQueue_.pop();
        }
        return stopLists;
    };

    sendStatusResponseCallback_ = [this](mindie_llm::InferRequestId reqId, mindie_llm::Status status,
                                         mindie_llm::StatusResponseType type) {};

    statusCallback_ = [this](const std::string &status) {};

    llmManager_ = std::make_shared<mindie_llm::LlmManager>(configPath, getRequestsCallback_, sendResponsesCallback_,
                                                           controlSignalCallback_, statusCallback_,
                                                           sendStatusResponseCallback_, ipInfo);

    return true;
}

bool LlmInferEngine::Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds)
{
    auto status = llmManager_->Init(modelInstanceId, npuDeviceIds);
    if (!status.IsOk()) {
        std::cout << "InitLlmManager failed: " << status.StatusMsg() << std::endl;
        return false;
    }
    return true;
}

bool LlmInferEngine::Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds,
                          std::map<std::string, std::string> extendInfo)
{
    auto status = llmManager_->Init(modelInstanceId, npuDeviceIds, extendInfo);
    if (!status.IsOk()) {
        std::cout << "InitLlmManager failed: " << status.StatusMsg() << std::endl;
        return false;
    }
    return true;
}

uint32_t LlmInferEngine::GetMaxPositionEmbeddings() const
{
    return llmManager_->GetMaxPositionEmbeddings();
}

void LlmInferEngine::Forward(std::shared_ptr<mindie_llm::InferRequest> request) { this->requestQueue_.push(request); }

void LlmInferEngine::ForwardControlRequest(std::pair<mindie_llm::InferRequestId, mindie_llm::OperationV2> contrlRequest)
{
    this->stopQueue_.push(contrlRequest);
}
void LlmInferEngine::Finalize() { llmManager_->Shutdown(); }
} // namespace mindie_llm