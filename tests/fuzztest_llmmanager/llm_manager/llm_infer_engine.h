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

#pragma once

#include "infer_request.h"
#include "llm_manager.h"
#include <condition_variable>
#include <queue>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace mindie_llm {
class LlmInferEngine {
public:
    LlmInferEngine() = default;

    ~LlmInferEngine();

    bool Create(const std::string &configPath = "",
                std::map<std::string, std::string> ipInfo = std::map<std::string, std::string>());

    bool Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds);

    bool Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds, std::map<std::string, std::string> extendInfo);

    uint32_t GetMaxPositionEmbeddings() const;

    void Forward(std::shared_ptr<mindie_llm::InferRequest> request);

    void Finalize();

    void ForwardControlRequest(std::pair<mindie_llm::InferRequestId, mindie_llm::OperationV2> contrlRequest);

    std::shared_ptr<mindie_llm::LlmManager> llmManager_;

private:
    mindie_llm::GetRequestsCallback getRequestsCallback_;
    mindie_llm::SendResponsesCallback sendResponsesCallback_;
    mindie_llm::ControlSignalCallback controlSignalCallback_;
    mindie_llm::LlmManagerStatsCallback statusCallback_;
    mindie_llm::SendStatusResponseCallback sendStatusResponseCallback_;
    std::mutex forwardMutex_;
    std::queue<std::shared_ptr<mindie_llm::InferRequest>> requestQueue_;
    std::queue<std::pair<mindie_llm::InferRequestId, mindie_llm::OperationV2>> stopQueue_{};
};
} // namespace mindie_llm