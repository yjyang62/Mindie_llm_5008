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

#ifndef ATB_TORCH_OPERATION_FACTORY_H
#define ATB_TORCH_OPERATION_FACTORY_H
#include <map>
#include <string>
#include <functional>
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_torch {
using OperationCreateFunc = std::function<atb::Operation *(const nlohmann::json &opParamJson)>;

class OperationFactory {
public:
    static OperationFactory &Instance();
    void RegisterOperation(const std::string &opType, OperationCreateFunc func);
    atb::Operation *CreateOperation(const std::string &opType, const std::string &opParam);
    std::vector<std::string> SupportOperations() const;

private:
    std::map<std::string, OperationCreateFunc> operationCreateFuncMap_;
};

#define REGISTER_OPERATION(opType, operationCreateFunc)     \
    struct Register##_##opType##_##operationCreateFunc {    \
        inline Register##_##opType##_##operationCreateFunc() noexcept \
        {  \
            atb_torch::OperationFactory::Instance().RegisterOperation(#opType, operationCreateFunc); \
        }  \
    } static g_register_##opType##operationCreateFunc
} // namespace atb_torch
#endif