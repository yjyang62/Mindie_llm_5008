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

namespace mindie_llm {

std::map<std::string, SpyResponseInfo> LlmManagerTest::responseSpyMap;

void SetResponseSpy(InferRequestId requestId, const TensorMap& outputs, bool isFinal, const std::string& errorMsg)
{
    SpyResponseInfo responseInfo;
    responseInfo.outputs = outputs;
    responseInfo.isFinal = isFinal;
    LlmManagerTest::responseSpyMap[requestId.GetRequestIdString()] = responseInfo;
}

std::optional<SpyResponseInfo> GetResponseSpy(const std::string& requestId)
{
    auto it = LlmManagerTest::responseSpyMap.find(requestId);
    if (it != LlmManagerTest::responseSpyMap.end()) {
        return it->second;
    }
    return std::nullopt;
}

void ClearResponseSpy()
{
    LlmManagerTest::responseSpyMap.clear();
}

std::vector<std::shared_ptr<InferRequest>> GetRequestsStub()
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
}