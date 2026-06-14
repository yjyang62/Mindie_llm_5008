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

#ifndef ATB_SPEED_DYNAMIC_PARAM_H
#define ATB_SPEED_DYNAMIC_PARAM_H
#include <nlohmann/json.hpp>
#include "atb_speed/utils/singleton.h"
#include "operations/fusion/utils.h"
#include "models/base/param/param_utils.h"

namespace atb_speed {
namespace base {

template<typename T>
class DynamicParam {
public:
    void Parse(std::string name, nlohmann::json &paramJson)
    {
        this->enableDap_ = false;  // reset

        this->name_ = name;
        if (!paramJson.contains(name)) { return; }
        this->data_ = FetchJsonParam<T>(paramJson, name);

        std::string suffix = GetSingleton<common::DapManager>().GetSuccessorSuffix();
        if (!paramJson.contains(name + suffix)) { return; }
        this->enableDap_ = true;
        GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        this->successorData_ = FetchJsonParam<T>(paramJson, name + suffix);
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }

    T& Get()
    {
        common::DapRole role = GetSingleton<common::DapManager>().GetRole();
        return role == common::DapRole::SUCCESSOR ? this->successorData_ : this->data_;
    }

    std::string name_  = "";

private:
    T data_;
    T successorData_;
    bool enableDap_ = false;
};
} // namespace base
} // namespace atb_speed
#endif