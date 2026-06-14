/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_UTILS_MODEL_FACTORY_H
#define ATB_SPEED_UTILS_MODEL_FACTORY_H

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "atb_speed/base/model.h"

namespace atb_speed {
using CreateModelFuncPtr = std::function<std::shared_ptr<atb_speed::Model>(const std::string &)>;

class ModelFactory {
public:
    static bool Register(const std::string &modelName, CreateModelFuncPtr createModel);
    static std::shared_ptr<atb_speed::Model> CreateInstance(const std::string &modelName, const std::string &param);
private:
    static std::unordered_map<std::string, CreateModelFuncPtr> &GetRegistryMap();
};

#define MODEL_NAMESPACE_STRINGIFY(modelNameSpace) #modelNameSpace
#define REGISTER_MODEL(nameSpace, modelName)                                                      \
        struct Register##_##nameSpace##_##modelName {                                             \
            inline Register##_##nameSpace##_##modelName() noexcept                                \
            {                                                                                     \
                ModelFactory::Register(MODEL_NAMESPACE_STRINGIFY(nameSpace##_##modelName),        \
                    [](const std::string &param) { return std::make_shared<modelName>(param); }); \
            }                                                                                     \
        } static instance_##nameSpace##modelName
} // namespace atb_speed
#endif