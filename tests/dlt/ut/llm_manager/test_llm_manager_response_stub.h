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
 
#pragma once

#include "test_llm_manager_adapter.h"

namespace mindie_llm {
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubInputIds();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubLoraId();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubIgnoreEos();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubStopStrings();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubLogProbs();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTopLogProbs();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTemperature();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTopK();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTopP();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubTypicalP();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubDoSample();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubSeed();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubRepetitionPenalty();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubFrequencyPenalty();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubPresencyPenalty();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubIncludeStopStrInOutput();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubWatermark();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubN();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubBestOf();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubUseBeamSearch();
std::vector<std::shared_ptr<InferRequest>> GetRequestsStubLengthPenalty();
}