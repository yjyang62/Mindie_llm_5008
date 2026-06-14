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

#ifndef ATB_SPEED_MODELS_SPARSE_MOE_OPERATION_H
#define ATB_SPEED_MODELS_SPARSE_MOE_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/comm.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {

enum SparseMoeIdx : int {
    ROUTER_IDX = 0,
    MOE_MLP_GATE_IDX,
    MOE_MLP_UP_IDX,
    MOE_MLP_DOWN_IDX
};

struct SparseMoeParam {
    atb::SVector<int64_t> axes = {1};  /// The axes on which softmax is applied
    atb::SVector<int32_t> num = {6}; /// The number of experts selected for each token
    atb::SVector<int32_t> topkGroups = {3};  /// The number of groups/device selected
    int scaledTopk = -1;  /// The non-deepseek models do not have the scaledTopk feature enabled by default
    bool enableInitRoutingCutoff = false;  /// A flag indicating whether to use scaled topk option
    std::vector<int> moeLinearQuantType = {};  /// The list of quantization types of linear operations in MoE graph
    std::vector<int32_t> deviceExpert = {};   /// The list of experts loaded on the device
    uint32_t numOfDeviceExperts = 64;  /// The number of experts loaded to the device
    uint32_t numOfExperts = 64;  /// The total number of experts utilized by the model
    int numOfGroups = 8;  /// number of groups in total
    int expertParallelDegree = 0;  /// The specific realization of expert parallelism strategy utilized by the model
    float routedScalingFactor = 1.0;  /// The optional scaling factor for expert scores
    bool transpose = true;  /// A flag indicating whether matrecies need to be transpose for matrix multiplications
    bool supportSwiGLU = true;  /// A flag indicating whether the device supports SwiGlu operator
    bool isBF16 = false;  /// A flag indicating whether the model runs on bfloat16
    bool isDynamicEp = false;  /// A flag indicating whether to use dynamic expert parallelism mechanism
    std::string routingMethod = "softMaxTopK";  /// The way in which the top k experts are selected
    std::string processLogits = "none";  /// The way in which expert scores are further processed
    bool gateUpTransposeB = false;  /// A flag indicating whether the B matrix of gateup operation should be transposed
    bool downTransposeB = false;  /// A flag indicating whether the B matrix of down operation should be transposed
    bool enableFusedRouting = false;  /// A flag indicating whether or not to use integrated routing operators
    bool enableInitQuant = false; /// A flag indicating whether to use routing-quant integrated operator
    bool enableSwigluQuant = false; /// A flag indicating whether to use swiglu-quant integrated operator
    bool enableMoeParallel = false; /// A flag indicating whether the model use Moe parallel
    bool enableCVOverlap = false; /// A flag indicating whether the model use cube and vector parallel
    bool enableFusedTopk = false; /// A flag indicating whether to use fused topk operator
    bool rounterHasBias = false;  /// A flag indicating whether is bias in the expert selection process
    /// A flag indicating whether or not to use integrated GMM+Swiglu+quant operators.
    bool enableGMMSwigluQuant = false;
    /// A flag indicating whether or not to use fused atb GMM+Swiglu+quant operators instead of aclnn.
    bool enableAtlasGMMFused = false;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;  // The quantization type of the packed weights
    int quantGroupSize = 0; /// Group size of per-group quantization
    /// The quantization type used to facilitate the calculation of the quantization type of the linear operation
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    bool useStdNorm = false;  /// A flag indicating whether the model utilizes std to normalize expert scores
    bool enableDispatchCombineV2 = false; /// A flag indicating whether to use dispatch_v2 and combine_v2

    bool enableMoeDistribute = false; /// A flag indicating whether to use moe distribute fusion operator
    bool enableExpertCumSumOutput = false; /// A flag indicating whether output ExpertCumSum
    bool enableTopkOutput = false; /// A flag indicating whether output Topk
    bool enableGatingDp = false;  /// A flag indicating whether gate dp
    bool enableGatingShift = false;  /// A flag indicating whether gate need shift
    bool enableGatingOverlap = false;  /// A flag indicating whether Gating overlap
    bool enableFp32GateInput = false;
    int64_t numDanglingSharedExperts = 0;

    bool enableATBGateMatmul = false;  /// A flag indicating whether enable ATB GateMatmul
    bool enableLoadBalance = false;
    bool enableEPWB = false;
    uint32_t numOfRedundantExpert = 0;
    bool hasBias = false;

    bool enableNodeBaseAll2All = false;
    int maxDecodeDpTokenSize = 0;
    HcclComm dispatchAndCombineHcclComm = nullptr;
    std::string dispatchAndCombinecommDomain = "";

    bool hasMoeEp = false;
    atb_speed::common::ParallelInfo moeEpParallelInfo;
    atb_speed::common::ParallelInfo mlpTpParallelInfo;
    atb_speed::common::ParallelInfo moeEpInterNodeParallelInfo;
    atb_speed::common::ParallelInfo moeEpIntraNodeParallelInfo;

    bool enableLcocAll2All = false;
    std::string lcclMoeEpDomain = "";
    HcclComm lcclMoeEpHcclComm = nullptr;

    bool mixSharedRouting = false;
};

/// This function creates the graph of the MoE of a model.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparseMoeOperation(const SparseMoeParam &param, atb::Operation **operation);
/// This function adds a linear transformation operator that calculates the row score of each expert on each token.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparseMoemoeGate(
    const SparseMoeParam &param, atb::Node &linearNode, atb::GraphParam opGraph);
/// This function adds a linear transformation operator that calculates the row score of each expert on each token
/// in float32 dtype.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparseMoemoeGateFp32(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph);
/// This function adds a softmax operator that process the score of each expert on each token to the graph.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparseMoesoftMax(
    const SparseMoeParam &param, atb::Node &softMaxNode, atb::GraphParam opGraph);
/// This function adds a sorting operator that selects top experts for each token to the graph.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparsMoetopK(
    const SparseMoeParam &param, atb::Node &topKNode, atb::GraphParam opGraph);
/// This function, working along with `CreateSparseMoedivide`, normalizes the scores of top experts.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparseMoereduce(atb::Node &reduceNode, atb::GraphParam opGraph);
/// This function, working along with `CreateSparseMoereduce`, normalizes the scores of top experts.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparseMoedivide(
    std::shared_ptr<int64_t> batchDimPtr, atb::Node &divideNode, atb::GraphParam opGraph);
/// This funciton adds an std operator that normalizes expert scores to the graph.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparseMoeStd(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph);
///  This function, working along with `CreateSparseMoeStd`, normalizes the scores of top experts.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateSparseMoeNorm(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph);
///  This function, working along with `CreateFusedAddTopkDiv`, concat extra expert for top experts.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateConcatExpertOperation(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param,
    atb::GraphParam &opGraph);
///  This function, working along with `CreateConcatExpertOperation`, concat extra weight for top experts.
/// \return A flag that indicates whether the opertaion is successfully created or not.
atb::Status CreateConcatWeightOperation(
    std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph);
///  This function is used for decrease duplicate code for eplb data collection.
/// \return void.
template <typename Container>
inline void SetOutTensorDescsForEPLB(Container &outTensorDescs, const uint32_t &outTensoridx,
    const uint32_t num, bool isGlist = true)
{
    outTensorDescs.at(outTensoridx) = atb::TensorDesc{};
    outTensorDescs.at(outTensoridx).format = ACL_FORMAT_ND;
    outTensorDescs.at(outTensoridx).shape.dimNum = 1;
    outTensorDescs.at(outTensoridx).dtype = isGlist ? ACL_INT64 : ACL_INT32;
    outTensorDescs.at(outTensoridx).shape.dims[0] = num;
}

}
} // namespace atb_speed
#endif