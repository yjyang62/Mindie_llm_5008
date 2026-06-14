/**
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

#ifndef ATB_SPEED_MODELS_DEVICE_LIMITED_ROUTING_OPERATION_H
#define ATB_SPEED_MODELS_DEVICE_LIMITED_ROUTING_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace deviceLimitedRouting {
struct DeviceLimitedRoutingParam {
    int numOfExperts = 64;  /// number of experts in total
    int numOfGroups = 8;  /// number of groups/device in total
    atb::SVector<int32_t> topkGroups = {3};  /// number of groups/device selected
};

/// This function creates a sub-graph that completes the Device-Limited expert selection mechanism
/// that is first designed for DeepseekV2.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateDeviceLimitedRoutingOperation(const DeviceLimitedRoutingParam &param, atb::Operation **operation);

}
} // namespace atb_speed
#endif