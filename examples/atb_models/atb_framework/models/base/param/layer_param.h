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

#ifndef ATB_SPEED_BASE_LAYER_PARAM_H
#define ATB_SPEED_BASE_LAYER_PARAM_H
#include <vector>
#include <nlohmann/json.hpp>
#include "models/base/param/param.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace base {

/// Parameters for the base layer, inherited from the `Param` class.
///
/// In addition to the parameters defined in the Param class,
/// this class introduces additional parameters specific to the base `DecoderLayer` class.
class LayerParam : public Param {
public:
    LayerParam() {};
    ~LayerParam() override {};
    void PrintParam() override;
    void CheckParam() override;

    /// The layer index, starting from 0
    int layerId = 0;
    /// Number of hidden layers
    int numHiddenLayers = 0;
    /// Information for tensor parallelism
    atb_speed::common::TensorParallelInfo tensorParallelInfo;
    /// Indicates the pack type and the quantization type of the qkv linear and gate up linear.
    std::vector<int> packQuantType = {
        common::PackQuantType::PACK_QUANT_UNDEFINED, common::PackQuantType::PACK_QUANT_UNDEFINED
    };
    /// Specifies the quantization type for the following linear module:
    /// q linear, k linear, v linear, dense linear, gate linear, up linear, and down linear.
    std::vector<int> linearQuantType = {
        common::LinearType::INVALID, common::LinearType::INVALID, common::LinearType::INVALID,
        common::LinearType::INVALID, common::LinearType::INVALID, common::LinearType::INVALID,
        common::LinearType::INVALID
    };
    /// Defines the transpose type of the second matrix in the matmul operation for the following linear module:
    /// q linear, k linear, v linear, dense linear, gate linear, up linear, and down linear.
    std::vector<int> linearTransposeType = {};
    /// Specifies whether the following linear module has bias:
    /// qkv linear, dense linear, gateup linear and down linear.
    std::vector<bool> linearHasBias = {false, false, false, false};
    /// Specifies the weight description of the following linear module:
    /// qkv linear, dense linear, gateup linear and down linear.
    std::vector<int> linearDescs = {
        common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC,
        common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC, common::LinearDesc::INVALID_DESC,
        common::LinearDesc::INVALID_DESC
    };
    /// Specifies whether the input norm and post attention norm enable antioutlier
    std::vector<bool> isAntiOutlier = {false, false};
};
} // namespace base
} // namespace atb_speed
#endif