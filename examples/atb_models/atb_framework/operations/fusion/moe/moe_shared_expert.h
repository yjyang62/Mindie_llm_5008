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

#ifndef ATB_SPEED_MODELS_MOE_SHARED_EXPERT_H
#define ATB_SPEED_MODELS_MOE_SHARED_EXPERT_H

#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {

constexpr uint64_t SHARED_MOE_GATE_LINEAR_INDEX = 0;
constexpr uint64_t SHARED_MOE_UP_LINEAR_INDEX = 1;
constexpr uint64_t SHARED_MOE_DOWN_LINEAR_INDEX = 2;
constexpr uint64_t SHARED_MOE_SHAREGATE_LINEAR_INDEX = 3;

struct SharedExpertParam {
    bool transposeGateup = true;  /// A flag indicating whether the B matrix of gateup operation should be transposed
    bool transposeDown = false;  /// A flag indicating whether the B matrix of down operation should be transposed
    bool hasSharedExpertGate = true;  /// A flag indicating whether there is routing mechanism for shared experts
    bool enableSwiGLUQuantForSharedExperts = false;
    bool supportSwiGLU = true;  /// A flag indicating whether the device supports SwiGlu operator
    bool isBF16 = false; /// A flag indicating whether the model runs on bfloat16
    bool enableCVOverlap = false; /// A flag indicating whether the model use cube and vector parallel
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;   /// The quantization type of the packed weights
    int quantGroupSize = 0; /// Group size of per-group quantization
    /// The quantization type used to facilitate the calculation of the quantization type of the linear operation
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    /// A list of quantization types of the linear operations in this sub-graph
    std::vector<int> mlpLinearQuantType = {};
    /// A list of flags indicating whether the B matrecies of the linear operations should be tranpsoed
    std::vector<int> mlpLinearTransposeType = {};
};

/// This funciton constructs the tensor map of this sub-graph.
/// \return A flag that indicates whether the opertaion is successfully created or not.
std::map<std::string, uint32_t> ConstructTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

/// This function creates a sub-graph that performance the shared-experts calculation on the input.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSharedExpertOperation(
    const SharedExpertParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif