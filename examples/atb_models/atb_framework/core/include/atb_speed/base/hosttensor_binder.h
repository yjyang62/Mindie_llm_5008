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

#ifndef ATB_SPEED_HOSTTENSOR_BINDER_H
#define ATB_SPEED_HOSTTENSOR_BINDER_H
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>

namespace atb_speed {
class HostTensorBinder {
public:
    HostTensorBinder() = default;
    virtual ~HostTensorBinder() = default;
    virtual void ParseParam(const nlohmann::json &paramJson) = 0;
    virtual void BindTensor(atb::VariantPack &variantPack) = 0;
};
} // namespace atb_speed
#endif