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

#ifndef ATB_SPEED_MODELS_MOE_MLP_OPERATION_H
#define ATB_SPEED_MODELS_MOE_MLP_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {
/// The non-deepseek models do not have the scaledTopk feature enabled by default
constexpr int TOPK_SCALE_DEFAULT_CLOSE = -1;
struct MoeMlpParam {
    bool transpose = true;  /// A flag indicating whether matrecies need to be transpose for matrix multiplications
    bool supportSwiGLU = true;  /// A flag indicating whether the device supports SwiGlu operator
    bool shiftedTopK = false;  /// A flag indicating whether or not to shift topk
    int32_t topk = 2;  /// The number of experts selected for each token
    int scaledTopk = -1;  /// The number of experts selected for each token after scale
    bool enableInitRoutingCutoff = false;  /// A flag indicating whether to use scaled topk option
    int gmmQuantType = 0;  /// Quantization type of the Gourped linear transformation
    uint32_t numOfExperts = 8;  /// The total number of experts utilized by the model
    uint32_t numOfDeviceExperts = 8;  /// The number of experts loaded to the device
    std::vector<int> moeLinearQuantType = {};  /// The list of quantization types of linear operations in MoE graph
    std::vector<int32_t> deviceExpert = {0, 1, 2, 3, 4, 5, 6, 7};  /// The list of experts loaded on the device
    int expertParallelDegree = 0;  /// The specific realization of expert parallelism strategy utilized by the model
    bool hasBias = false;  /// A flag indicating whether there are bias to the linear operation weights
    bool isBF16 = false;  /// A flag indicating whether the model runs on bfloat16
    bool gateUpTransposeB = false;  /// A flag indicating whether the B matrix of gateup operation should be transposed
    bool downTransposeB = false;  /// A flag indicating whether the B matrix of down operation should be transposed
    bool enableFusedRouting = false;  /// A flag indicating whether to use integrated routing operators
    /// A flag indicating whether or not to use integrated GMM+Swiglu+quant operators.
    bool enableGMMSwigluQuant = false;
    /// A flag indicating whether or not to use fused atb GMM+Swiglu+quant operators instead of aclnn.
    bool enableAtlasGMMFused = false;
    bool enableInitQuant = false; /// A flag indicating whether to use routing-quant integrated operator
    bool enableSwigluQuant = false; /// A flag indicating whether to use swiglu-quant integrated operator
    bool enableMoeParallel = false; /// A flag indicating whether the model use Moe parallel
    bool enableCVOverlap = false; /// A flag indicating whether the model use cube and vector parallel
    bool hasMoeEp = false; /// A flag indicating whether the model uses expert parallelism
    bool enableDispatchCombineV2 = false; /// A flag indicating whether to use dispatch_v2 and combine_v2
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;   /// The quantization type of the packed weights
    /// The quantization type used to facilitate the calculation of the quantization type of the linear operation
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    /// The group size used for dequantizing the weight tensor in the per-group quantization approach
    int quantGroupSize = 0;

    std::string backend = "hccl";
    int moeEpRank = 0;
    int moeEpSize = 1;
    int maxDecodeDpTokenSize = 0;
    std::string moeEpDomain = "";
    std::string moeEpRankTableFile = "";
    bool hasMlpTp = false;
    int mlpTpRank = 0;
    int mlpTpSize = 1;
    std::string mlpTpDomain = "";
    std::string mlpTpRankTableFile = "";

    bool enableMoeDistribute = false;
    bool enableExpertCumSumOutput = false;
    bool enableGatingDp = false;
    int64_t numDanglingSharedExperts = 0;
    uint32_t numOfRedundantExpert = 0;
};

/// This funciton creates a sub-graph that performs the FFN of a model with MoE structure
atb::Status CreateMoeMlpOperation(const MoeMlpParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif