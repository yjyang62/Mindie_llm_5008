/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "atb_speed/utils/operation_factory.h"
#include "atb_speed/log.h"

namespace atb_speed {
bool OperationFactory::Register(const std::string &operationName, CreateOperationFuncPtr createOperation)
{
    auto it = OperationFactory::GetRegistryMap().find(operationName);
    if (it != OperationFactory::GetRegistryMap().end()) {
        ATB_SPEED_LOG_WARN(operationName << " operation already exists, but the duplication doesn't matter.");
        return false;
    }
    OperationFactory::GetRegistryMap()[operationName] = createOperation;
    return true;
}

atb::Operation *OperationFactory::CreateOperation(const std::string &operationName, const nlohmann::json &param)
{
    auto it = OperationFactory::GetRegistryMap().find(operationName);
    if (it != OperationFactory::GetRegistryMap().end()) {
        if (it->second == nullptr) {
            ATB_SPEED_LOG_ERROR("Find operation error: " << operationName);
            return nullptr;
        }
        ATB_SPEED_LOG_DEBUG("Find operation: " << operationName);
        return it->second(param);
    }
    ATB_SPEED_LOG_WARN("OperationName: " << operationName << " not find in operation factory map");
    return nullptr;
}

std::unordered_map<std::string, CreateOperationFuncPtr> &OperationFactory::GetRegistryMap()
{
    static std::unordered_map<std::string, CreateOperationFuncPtr> operationRegistryMap;
    return operationRegistryMap;
}
} // namespace atb_speed
