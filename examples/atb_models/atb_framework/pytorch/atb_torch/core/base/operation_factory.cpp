/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "operation_factory.h"

namespace atb_torch {
OperationFactory &OperationFactory::Instance()
{
    static OperationFactory instance;
    return instance;
}

void OperationFactory::RegisterOperation(const std::string &opType, OperationCreateFunc func)
{
    operationCreateFuncMap_[opType] = func;
    ATB_SPEED_LOG_DEBUG("RegisterOperation success " << opType);
}

atb::Operation *OperationFactory::CreateOperation(const std::string &opType, const std::string &opParam)
{
    auto it = operationCreateFuncMap_.find(opType);
    if (it == operationCreateFuncMap_.end()) {
        ATB_SPEED_LOG_ERROR("Create atb operation fail, not find opType:" << opType);
        return nullptr;
    }

    size_t maxParamLength = 200000;
    if (opParam.size() > maxParamLength) {
        ATB_SPEED_LOG_ERROR("Create atb operation fail, op_param's length (" << opParam.size()
            << ") should be smaller than " << maxParamLength);
        return nullptr;
    }

    nlohmann::json opParamJson;
    try {
        opParamJson = nlohmann::json::parse(opParam);
    } catch (const std::exception &e) {
        ATB_SPEED_LOG_ERROR("Create atb operation fail, error:" << e.what());
        return nullptr;
    }

    return it->second(opParamJson);
}

std::vector<std::string> OperationFactory::SupportOperations() const
{
    std::vector<std::string> operationTypes;
    for (auto &it : operationCreateFuncMap_) {
        operationTypes.push_back(it.first);
    }
    return operationTypes;
}
} // namespace atb_torch