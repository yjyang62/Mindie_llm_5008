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

#include "sparse_moe.h"
#include <atb/atb_infer.h>
#include <list>
#include <memory>
#include "moe_mlp.h"
#include "device_limited_routing.h"
#include "operations/fusion/moe/ep/dynamic_ep_moe.h"
#include "operations/aclnn/ops/moe_topk_softmax_operation.h"
#include "operations/aclnn/ops/vector_norm_operation.h"
#include "operations/aclnn/ops/std_operation.h"
#include "operations/aclnn/ops/sigmoid_operation.h"
#include "operations/aclnn/ops/matmul_operation.h"
#include "operations/aclnn/ops/concat_operation.h"
#include "atb_speed/base/event_manager.h"
#include "atb_speed/utils/check_util.h"

namespace atb_speed {
namespace common {

const uint64_t NODE_SIZE_INCR_NORMALIZATION  = 2;
static const uint64_t STREAM1 = 1;
static const uint64_t NUM1 = 1;
static const uint64_t NUM2 = 2;
static const uint64_t NUM3 = 3;
static const uint64_t NUM4 = 4;
static const uint64_t NUM5 = 5;
constexpr uint32_t TOPK_IN_NUM = 4;
constexpr uint32_t TOPK_IN3_DIM = 3;

std::map<std::string, std::vector<std::string>> GetSparseMoeInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> moeMlpInTensorCandidates = {
        {"default", {
            "in_hiddenstates", "in_gate_weight", "in_gate_bias", "in_gate_descale", "in_gate_offset",
            "in_gate_scale", "in_gate_compress_idx", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
            "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array",
            "in_expert_group", "in_one_hot", "in_zero_hot"}
        },
        {"ep", {
            "in_start_expert_idx", "in_device_expert_count", "in_padding_idx"}
        },
        {"dynamic_ep", {
            "in_dynamic_ep_idx", "in_moe_idx"}
        },
        {"force_load_balance", {
            "in_fake_selected_experts"}
        },
        {"epwb", {
            "in_expert_routing_map"}
        },
        {"gating_dp", {
            "in_hiddenstates_slice", "in_attn_unpadding_idx"}
        },
        {"fp32_gate_input", {
            "in_hiddenstates_fp32"}
        }
    };
    return moeMlpInTensorCandidates;
}

atb::Status ConstructATBGateMatmulTensorMap(const SparseMoeParam &param, std::vector<std::string> &interTensorList)
{
    if (!param.enableFp32GateInput) {
        interTensorList.push_back("intermediate_hiddenstates_fp32");
    }
    interTensorList.push_back("intermediate_router_logits_fp32");
    return atb::NO_ERROR;
}

atb::Status ConstructRoutingTensorMap(const SparseMoeParam &param, std::vector<std::string> &interTensorList)
{
    if (param.enableATBGateMatmul && (param.routingMethod == "noAuxTc" || param.routingMethod == "topkFused")) {
        ConstructATBGateMatmulTensorMap(param, interTensorList);
    }
    if (!param.enableFusedTopk) {
        interTensorList.push_back("intermediate_router_weights");
        interTensorList.push_back("intermediate_router_weights_topk");
        if (param.routingMethod == "noAuxTc" || param.routingMethod == "topkFused") {
            interTensorList.push_back("intermediate_router_weights_for_choice");
            interTensorList.push_back("intermediate_router_weights_topk_temp");
        }
        if (param.useStdNorm) {
            interTensorList.push_back("intermediate_router_logits_std");
        }
        if (param.processLogits == "normalization" ||
            param.processLogits == "norm" ||
            param.processLogits == "normScaling") {
            interTensorList.push_back("intermediate_router_weights_topk_reduced");
            interTensorList.push_back("intermediate_router_weights_topk_sumed");
        } else if (param.processLogits == "scaling") {
            interTensorList.push_back("intermediate_router_weights_topk_reduced");
        }
    }
    if (param.processLogits != "none") {
        interTensorList.push_back("intermediate_router_weights_topk_reduced_fp32");
        if (!param.enableMoeDistribute) {
            interTensorList.push_back("intermediate_router_weights_topk_reduced_fp16");
        }
    } else if (param.enableMoeDistribute && param.enableDispatchCombineV2) {
        interTensorList.push_back("intermediate_router_weights_topk_fp32");
    }
    if (param.mixSharedRouting) {
        interTensorList.push_back("intermediate_router_weights_topk_reduced_mix_shared");
        interTensorList.push_back("intermediate_selected_experts_mix_shared");
    }
    return atb::NO_ERROR;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const SparseMoeParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto moeMlpInTensorCandidates = GetSparseMoeInTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {
        "intermediate_router_logits", "intermediate_selected_experts"};
    std::vector<std::string> outTensorList = {"out_moe_rout"};
    AddTensorToList(moeMlpInTensorCandidates, "default", inTensorList);
    if (param.enableLoadBalance) {
        AddTensorToList(moeMlpInTensorCandidates, "force_load_balance", inTensorList);
    }
    if (param.enableExpertCumSumOutput) {
        outTensorList.push_back("out_gmm_cumsum_list");
    }
    if (param.enableTopkOutput) {
        outTensorList.push_back("out_selected_experts");
    }
    ConstructRoutingTensorMap(param, interTensorList);
    if (param.hasMoeEp) {
        AddTensorToList(moeMlpInTensorCandidates, "ep", inTensorList);
        if (param.isDynamicEp) {
            AddTensorToList(moeMlpInTensorCandidates, "dynamic_ep", inTensorList);
        }
    }
    if (param.enableEPWB) {
        AddTensorToList(moeMlpInTensorCandidates, "epwb", inTensorList);
        interTensorList.push_back("intermediate_selected_experts_routed");
    }
    if (param.enableGatingDp) {
        AddTensorToList(moeMlpInTensorCandidates, "gating_dp", inTensorList);
        interTensorList.push_back("intermediate_selected_experts_with_padding_all");
        interTensorList.push_back("intermediate_selected_experts_all");
        interTensorList.push_back("intermediate_router_weights_topk_reduced_with_padding_all");
        interTensorList.push_back("intermediate_router_weights_topk_reduced_all");
    }
    if (param.enableGatingShift) {
        interTensorList.push_back("intermediate_router_logits_split_1");
        interTensorList.push_back("intermediate_router_logits_split_2");
        interTensorList.push_back("intermediate_router_logits_shifted");
    }
    if (param.enableFp32GateInput) {
        AddTensorToList(moeMlpInTensorCandidates, "fp32_gate_input", inTensorList);
    }
    if (param.mixSharedRouting) {
        inTensorList.push_back("mix_shared_routing_weight");
        inTensorList.push_back("mix_shared_routing_expert");
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();
    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}


atb::Status CreateSparseMoemoeGate(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node linearNode;
    FusionLinearParam moeGateParam;
    moeGateParam.transposeType = common::TRANSPOSE;
    moeGateParam.hasBias = param.rounterHasBias;
    moeGateParam.isBF16 = param.isBF16;
    moeGateParam.quantType = atb_speed::common::GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.moeLinearQuantType[SparseMoeIdx::ROUTER_IDX], false);
    moeGateParam.quantGroupSize = 0;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(moeGateParam, &linearNode.operation));
    linearNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                              GetTensorIdx(tensorMap, "in_gate_weight"),
                              GetTensorIdx(tensorMap, "in_gate_scale"),
                              GetTensorIdx(tensorMap, "in_gate_offset"),
                              GetTensorIdx(tensorMap, "in_gate_descale"),
                              GetTensorIdx(tensorMap, "in_gate_bias"),
                              GetTensorIdx(tensorMap, "in_gate_compress_idx")};
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    opGraph.nodes.push_back(linearNode);
    ATB_SPEED_LOG_DEBUG("Router logits calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoemoeGateFp32Atb(std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    if (!param.enableFp32GateInput) {
        atb::Node castUp;
        atb::infer::ElewiseParam castUpParam;
        castUpParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
        castUpParam.outTensorType = ACL_FLOAT;
        castUp.inTensorIds = {GetTensorIdx(tensorMap, (param.enableGatingDp) ?
                                "in_hiddenstates_slice" : "in_hiddenstates")};
        castUp.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_hiddenstates_fp32")};
        CHECK_OPERATION_STATUS_RETURN(CreateOperation(castUpParam, &castUp.operation));
        opGraph.nodes.push_back(castUp);
    }

    atb::Node linearNode;
    atb::infer::LinearParam moeGateParam;
    moeGateParam.hasBias = false;
    moeGateParam.transposeB = true;
    linearNode.inTensorIds = {GetTensorIdx(tensorMap, param.enableFp32GateInput ? "in_hiddenstates_fp32" : \
            "intermediate_hiddenstates_fp32"),
            GetTensorIdx(tensorMap, "in_gate_weight")};
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits_fp32")};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(moeGateParam, &linearNode.operation));
    if (param.enableGatingOverlap) {
        atb::SetExecuteStreamId(linearNode.operation, STREAM1);
    }
    opGraph.nodes.push_back(linearNode);

    atb::Node castDown;
    atb::infer::ElewiseParam castDownParam;
    castDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castDownParam.outTensorType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    castDown.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits_fp32")};
    castDown.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castDownParam, &castDown.operation));
    opGraph.nodes.push_back(castDown);

    ATB_SPEED_LOG_DEBUG("Router logits calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoemoeGateFp32Aclnn(std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node linearNode;
    // using aclnn matmul
    atb_speed::common::AclNNMatmulParam moeGateParam;
    moeGateParam.hasBias = false;
    moeGateParam.transposeB = true;
    linearNode.operation = new atb_speed::common::MatmulOperation("SparseMoeGateNode", moeGateParam);
    linearNode.inTensorIds = {GetTensorIdx(tensorMap, (param.enableGatingDp) ?
                              "in_hiddenstates_slice" : "in_hiddenstates"),
                              GetTensorIdx(tensorMap, "in_gate_weight")};
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    if (param.enableGatingOverlap) {
        atb::SetExecuteStreamId(linearNode.operation, STREAM1);
    }
    opGraph.nodes.push_back(linearNode);
    ATB_SPEED_LOG_DEBUG("Router logits calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoemoeGateFp32(std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    if (param.enableATBGateMatmul) {
        ATB_SPEED_LOG_DEBUG("Create ATB Fp32 Gate Matmul");
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoemoeGateFp32Atb(tensorMap, param, opGraph));
    } else {
        ATB_SPEED_LOG_DEBUG("Create ACLNN Fp32 Gate Matmul");
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoemoeGateFp32Aclnn(tensorMap, param, opGraph));
    }
    return atb::NO_ERROR;
}

atb::Status CreateSplit(std::map<std::string, uint32_t> &tensorMap, const SparseMoeParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node splitNode;
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = NUM1;
    splitParam.splitNum = NUM2;
    if (param.deviceExpert[0] == 0) {
        splitParam.splitSizes = {static_cast<int32_t>(NUM1), \
            static_cast<int32_t>(param.numOfExperts) - static_cast<int32_t>(NUM1)};
    } else {
        splitParam.splitSizes = {param.deviceExpert[0],
            static_cast<int32_t>(param.numOfExperts) - param.deviceExpert[0]};
    }
    CREATE_OPERATION(splitParam, &splitNode.operation);
    splitNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_router_logits"});
    splitNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_router_logits_split_1",
                                                                             "intermediate_router_logits_split_2"});
    opGraph.nodes.push_back(splitNode);
    return atb::NO_ERROR;
}

atb::Status CreateConcat(std::map<std::string, uint32_t> &tensorMap, const SparseMoeParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node concatNode;
    atb::infer::ConcatParam catParam;
    catParam.concatDim = -1;
    CREATE_OPERATION(catParam, &concatNode.operation);
    if (param.deviceExpert[0] == 0) {
        concatNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
            {"intermediate_router_logits_split_1", "intermediate_router_logits_split_2"});
    } else {
        concatNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
            {"intermediate_router_logits_split_2", "intermediate_router_logits_split_1"});
    }
    concatNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_router_logits_shifted"});
    opGraph.nodes.push_back(concatNode);
    return atb::NO_ERROR;
}

atb::Status CreateFusedAddTopkDiv(std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node fusedAddTopkDivNode;
    atb::infer::FusedAddTopkDivParam fusedAddTopkDivParam;
    fusedAddTopkDivParam.groupNum = param.numOfGroups;
    CheckPositive(param.topkGroups.size());
    fusedAddTopkDivParam.groupTopk = param.topkGroups[0];
    // 2: chosen number within each group
    constexpr uint32_t chosenNumEachGroup = 2;
    uint32_t numOfGroups = param.numOfGroups > 0 ? param.numOfGroups : 1;
    fusedAddTopkDivParam.n = std::min(param.numOfExperts / numOfGroups, chosenNumEachGroup);
    CheckPositive(param.num.size());
    fusedAddTopkDivParam.k = param.num[0];
    fusedAddTopkDivParam.scale = param.routedScalingFactor;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(fusedAddTopkDivParam, &fusedAddTopkDivNode.operation));
    fusedAddTopkDivNode.inTensorIds = {GetTensorIdx(tensorMap, (param.enableGatingShift) ? \
                                       "intermediate_router_logits_shifted" : "intermediate_router_logits"),
                                       GetTensorIdx(tensorMap, "in_gate_bias")};
    fusedAddTopkDivNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced_fp32"),
                                        GetTensorIdx(tensorMap, "intermediate_selected_experts")};
    if (param.enableGatingOverlap) {
        atb::SetExecuteStreamId(fusedAddTopkDivNode.operation, STREAM1);
    }
    opGraph.nodes.push_back(fusedAddTopkDivNode);
    ATB_SPEED_LOG_DEBUG("FusedAddTopkDivOperation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoeStd(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node stdNode;
    stdNode.operation = new atb_speed::common::StdOperation("SparseMoeStdNode");
    stdNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    stdNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits_std")};
    opGraph.nodes.push_back(stdNode);
    ATB_SPEED_LOG_DEBUG("Router logits std calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoeNorm(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node normNode;
    atb::infer::ElewiseParam normParam;
    normParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(normParam, &normNode.operation));
    normNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits"),
        GetTensorIdx(tensorMap, "intermediate_router_logits_std")};
    normNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    opGraph.nodes.push_back(normNode);
    ATB_SPEED_LOG_DEBUG("Router weights norm calculated success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoesoftMax(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node softMaxNode;
    atb::infer::SoftmaxParam softMaxParam;
    softMaxParam.axes = param.axes;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(softMaxParam, &softMaxNode.operation));
    softMaxNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    softMaxNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights")};
    opGraph.nodes.push_back(softMaxNode);
    ATB_SPEED_LOG_DEBUG("Router weights calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoeSigmoid(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node sigmoidNode;
    sigmoidNode.operation = new atb_speed::common::SigmoidOperation("SparseMoeSigmoidNode");
    sigmoidNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    sigmoidNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights")};
    opGraph.nodes.push_back(sigmoidNode);
    ATB_SPEED_LOG_DEBUG("Router logits sigmoid calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateScoreAdd(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node scoreAddNode;
    atb::infer::ElewiseParam scoreAddParam;
    scoreAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(scoreAddParam, &scoreAddNode.operation));
    scoreAddNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights"),
                                GetTensorIdx(tensorMap, "in_gate_bias")};
    scoreAddNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_for_choice")};
    opGraph.nodes.push_back(scoreAddNode);
    ATB_SPEED_LOG_DEBUG("Score add calculated success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoetopK(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node topKNode;
    atb::infer::SortParam topKParam;
    topKParam.num = param.num;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(topKParam, &topKNode.operation));
    std::string topKInTensorName;
    std::vector<std::string> topKOutTensorNames;
    if (param.routingMethod == "noAuxTc" || param.routingMethod == "topkFused") {
        topKInTensorName = "intermediate_router_weights_for_choice";
        topKOutTensorNames.push_back("intermediate_router_weights_topk_temp");
    } else {
        topKInTensorName = "intermediate_router_weights";
        topKOutTensorNames.push_back("intermediate_router_weights_topk");
    }
    topKOutTensorNames.push_back("intermediate_selected_experts");
    topKNode.inTensorIds = {GetTensorIdx(tensorMap, topKInTensorName)};
    topKNode.outTensorIds = GetTensorIdxList(tensorMap, topKOutTensorNames);
    opGraph.nodes.push_back(topKNode);
    ATB_SPEED_LOG_DEBUG("Expert selection success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoetopKGather(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node topKGaterNode;
    atb::infer::GatherParam topKGatherParam;
    topKGatherParam.axis = 1;
    topKGatherParam.batchDims = 1;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(topKGatherParam, &topKGaterNode.operation));
    topKGaterNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights"),
                                 GetTensorIdx(tensorMap, "intermediate_selected_experts")};
    topKGaterNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk")};
    opGraph.nodes.push_back(topKGaterNode);
    ATB_SPEED_LOG_DEBUG("Expert weight gather success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoereduce(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node reduceNode;
    atb::infer::ReduceParam reduceParam;
    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis = {1};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(reduceParam, &reduceNode.operation));
    reduceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk")};
    reduceNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_sumed")};
    opGraph.nodes.push_back(reduceNode);
    ATB_SPEED_LOG_DEBUG("Reduce sum calculated success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoedivide(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node divideNode;
    atb::infer::ElewiseParam divideParam;
    divideParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(divideParam, &divideNode.operation));
    divideNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk"),
                              GetTensorIdx(tensorMap, "intermediate_router_weights_topk_sumed")};
    divideNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced")};
    divideNode.inTensorReshapeFuncs.resize(divideNode.inTensorIds.size());
    divideNode.inTensorReshapeFuncs[1] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };
    opGraph.nodes.push_back(divideNode);
    ATB_SPEED_LOG_DEBUG("Router weights calculated success");
    return atb::NO_ERROR;
}

atb::Status CreateElewiseMuls(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node mulNode;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.mulsParam.varAttr = param.routedScalingFactor;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &mulNode.operation));
    std::string mulInTensorName = param.processLogits == "normScaling" ? \
                                  "intermediate_router_weights_topk_reduced" : "intermediate_router_weights_topk";
    mulNode.inTensorIds = {GetTensorIdx(tensorMap, mulInTensorName)};
    mulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced")};
    opGraph.nodes.push_back(mulNode);
    ATB_SPEED_LOG_DEBUG("ElewiseMuls calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateDeviceLimitedRouting(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node deviceLimitedNode;
    atb_speed::deviceLimitedRouting::DeviceLimitedRoutingParam deviceLimitedRoutingParam;
    deviceLimitedRoutingParam.numOfExperts = param.numOfExperts;
    deviceLimitedRoutingParam.numOfGroups = param.numOfGroups;
    deviceLimitedRoutingParam.topkGroups = param.topkGroups;
    atb_speed::deviceLimitedRouting::CreateDeviceLimitedRoutingOperation(deviceLimitedRoutingParam,
                                                                         &deviceLimitedNode.operation);
    deviceLimitedNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights"),
                                     GetTensorIdx(tensorMap, "in_expert_group"),
                                     GetTensorIdx(tensorMap, "in_one_hot"),
                                     GetTensorIdx(tensorMap, "in_zero_hot")};
    deviceLimitedNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights")};
    opGraph.nodes.push_back(deviceLimitedNode);
    ATB_SPEED_LOG_DEBUG("Router logits calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGroupOperation(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    // When n_group == 1, grouptopk degrades to the ordinary topk.
    if (param.topkGroups[0] == 1) {
        return atb::NO_ERROR;
    }
    atb::Node deviceLimitedNode;
    atb::infer::GroupTopkParam groupedParam;
    groupedParam.groupNum = param.numOfGroups;
    CheckPositive(param.topkGroups.size());
    groupedParam.k = param.topkGroups[0];
    if (param.routingMethod == "noAuxTc") {
        groupedParam.groupMultiFlag = atb::infer::GroupTopkParam::GroupMultiFlag::SUM_MULTI_MAX;
        groupedParam.n = 2; // 2: chosen number within each group
    }
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(groupedParam, &deviceLimitedNode.operation));
    std::vector<std::string> deviceLimitedInTensorNames;
    std::string deviceLimitedOutTensorName;
    if (param.routingMethod == "noAuxTc") {
        deviceLimitedInTensorNames.push_back("intermediate_router_weights_for_choice");
        deviceLimitedOutTensorName = "intermediate_router_weights_for_choice";
    } else {
        deviceLimitedInTensorNames.push_back("intermediate_router_weights");
        deviceLimitedOutTensorName = "intermediate_router_weights";
    }
    deviceLimitedInTensorNames.push_back("in_expert_group");
    deviceLimitedNode.inTensorIds = GetTensorIdxList(tensorMap, deviceLimitedInTensorNames);
    deviceLimitedNode.outTensorIds = {GetTensorIdx(tensorMap, deviceLimitedOutTensorName)};
    opGraph.nodes.push_back(deviceLimitedNode);
    ATB_SPEED_LOG_DEBUG("Fusion Router logits calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseTopkSoftMax(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node topKNode;
    atb_speed::common::MoeTopkSoftmaxParam moeTopkSoftmaxParam;
    moeTopkSoftmaxParam.topkNum = int64_t(param.num.at(0));
    topKNode.operation = new atb_speed::common::MoeTopkSoftmaxOperation("MoeTopkSoftmaxOperation", moeTopkSoftmaxParam);
    topKNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    topKNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk"),
                             GetTensorIdx(tensorMap, "intermediate_selected_experts"),
                             GetTensorIdx(tensorMap, "intermediate_router_weights")};
    opGraph.nodes.push_back(topKNode);
    ATB_SPEED_LOG_DEBUG("Expert selection success");
    return atb::NO_ERROR;
}

atb::Status CreateVectorNorm(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph)
{
    atb::Node vectorNormNode;
    atb_speed::common::AclNNVectorNormParam aclNNVectorNormParam;
    vectorNormNode.operation = new atb_speed::common::VectorNormOperation("vectorNormOperation", aclNNVectorNormParam);
    vectorNormNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk")};
    vectorNormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_sumed")};
    opGraph.nodes.push_back(vectorNormNode);
    ATB_SPEED_LOG_DEBUG("execute vector norm success");
    return atb::NO_ERROR;
}

atb::Status SetDynamicExpertParam(atb_speed::common::DynamicEpMoEParam &dynamicExpertParam, const SparseMoeParam &param)
{
    dynamicExpertParam.transpose = param.transpose;
    dynamicExpertParam.topk = param.num.at(0);
    if (param.mixSharedRouting) {
        dynamicExpertParam.topk = dynamicExpertParam.topk + 1;
    }
    dynamicExpertParam.scaledTopk = param.scaledTopk;
    dynamicExpertParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
    dynamicExpertParam.numOfExperts = param.numOfExperts;
    dynamicExpertParam.numOfDeviceExperts = param.numOfDeviceExperts;
    dynamicExpertParam.supportSwiGLU = param.supportSwiGLU;
    dynamicExpertParam.expertParallelDegree = param.expertParallelDegree;
    dynamicExpertParam.isDynamicEp = param.isDynamicEp;
    dynamicExpertParam.deviceExpert = param.deviceExpert;
    dynamicExpertParam.enableMoeDistribute = param.enableMoeDistribute;
    dynamicExpertParam.enableFusedTopk = param.enableFusedTopk;
    dynamicExpertParam.moeLinearQuantType = param.moeLinearQuantType;
    dynamicExpertParam.packQuantType = param.packQuantType;
    dynamicExpertParam.denseQuantType = param.denseQuantType;
    dynamicExpertParam.isBF16 = param.isBF16;
    dynamicExpertParam.gateUpTransposeB = param.gateUpTransposeB;
    dynamicExpertParam.downTransposeB = param.downTransposeB;
    dynamicExpertParam.enableFusedRouting = param.enableFusedRouting;
    dynamicExpertParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    dynamicExpertParam.enableInitQuant = param.enableInitQuant;
    dynamicExpertParam.enableSwigluQuant = param.enableSwigluQuant;
    dynamicExpertParam.enableAtlasGMMFused = param.enableAtlasGMMFused;
    dynamicExpertParam.quantGroupSize = param.quantGroupSize;
    dynamicExpertParam.enableCVOverlap = param.enableCVOverlap;
    dynamicExpertParam.routingMethod = param.routingMethod;
    dynamicExpertParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    if (param.enableEPWB) {
        dynamicExpertParam.numOfExperts = param.numOfExperts + param.numOfRedundantExpert;
    }
    dynamicExpertParam.numDanglingSharedExperts = param.numDanglingSharedExperts;
    dynamicExpertParam.numOfRedundantExpert = param.numOfRedundantExpert;
    dynamicExpertParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;
    dynamicExpertParam.enableGatingDp = param.enableGatingDp;
    dynamicExpertParam.enableNodeBaseAll2All = param.enableNodeBaseAll2All;

    dynamicExpertParam.hasMoeEp = param.hasMoeEp;
    dynamicExpertParam.moeEpParallelInfo = param.moeEpParallelInfo;
    dynamicExpertParam.mlpTpParallelInfo = param.mlpTpParallelInfo;
    dynamicExpertParam.moeEpInterNodeParallelInfo = param.moeEpInterNodeParallelInfo;
    dynamicExpertParam.moeEpIntraNodeParallelInfo = param.moeEpIntraNodeParallelInfo;

    dynamicExpertParam.dispatchAndCombinecommDomain = param.dispatchAndCombinecommDomain;
    dynamicExpertParam.dispatchAndCombineHcclComm = param.dispatchAndCombineHcclComm;

    dynamicExpertParam.enableDispatchCombineV2 = param.enableDispatchCombineV2;
    dynamicExpertParam.enableLcocAll2All = param.enableLcocAll2All;
    dynamicExpertParam.lcclMoeEpDomain = param.lcclMoeEpDomain;
    dynamicExpertParam.lcclMoeEpHcclComm = param.lcclMoeEpHcclComm;
    dynamicExpertParam.mixSharedRouting = param.mixSharedRouting;
    dynamicExpertParam.enableEPWB = param.enableEPWB;
    return atb::NO_ERROR;
}

atb::Status CreateFp32Cast(std::map<std::string, uint32_t> &tensorMap, const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    atb::Node castNode;
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_FLOAT;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, param.processLogits != "none" ?
                            "intermediate_router_weights_topk_reduced" : "intermediate_router_weights_topk")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, param.processLogits != "none" ?
                            "intermediate_router_weights_topk_reduced_fp32" : "intermediate_router_weights_topk_fp32")};
    opGraph.nodes.push_back(castNode);
    ATB_SPEED_LOG_DEBUG("Cast calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateFp16Cast(std::map<std::string, uint32_t> &tensorMap,
                           const SparseMoeParam &param,
                           atb::GraphParam &opGraph)
{
    atb::Node castNode;
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    if (param.mixSharedRouting) {
        castNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced_mix_shared")};
    } else {
        castNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced_fp32")};
    }
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced_fp16")};
    if (param.enableGatingOverlap) {
        atb::SetExecuteStreamId(castNode.operation, STREAM1);
    }
    opGraph.nodes.push_back(castNode);
    ATB_SPEED_LOG_DEBUG("Cast calculation success");
    return atb::NO_ERROR;
}

atb::Status SetExpertRoutingMap(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param,
    atb::GraphParam &opGraph)
{
    atb::Node padNode;
    atb::infer::GatherParam padParam;
    atb::CreateOperation(padParam, &padNode.operation);
    std::string intermediate_selected_experts_name;
    if (param.mixSharedRouting) {
        intermediate_selected_experts_name = "intermediate_selected_experts_mix_shared";
    } else if (param.enableGatingDp) {
        intermediate_selected_experts_name = "intermediate_selected_experts_all";
    } else {
        intermediate_selected_experts_name = "intermediate_selected_experts";
    }
    padNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, { "in_expert_routing_map", intermediate_selected_experts_name});
    padNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_selected_experts_routed"});

    padNode.inTensorReshapeFuncs.resize(padNode.inTensorIds.size());
    padNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // 2: dimNum
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    padNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // 2: dimNum
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    opGraph.nodes.push_back(padNode);
    ATB_SPEED_LOG_DEBUG("create padNode");
    return atb::NO_ERROR;
}

std::list<std::string> GetOutTensorName(const SparseMoeParam &param)
{
    std::list<std::string> nameList;
    nameList.push_back("in_hiddenstates");
    nameList.push_back("in_mlp_gateup_weight_expert");
    nameList.push_back("in_mlp_gateup_bias_expert");
    nameList.push_back("in_mlp_gateup_descale_expert");
    nameList.push_back("in_mlp_gateup_offset_expert");
    nameList.push_back("in_mlp_gateup_scale_expert");
    nameList.push_back("in_mlp_gateup_compress_idx_expert");
    nameList.push_back("in_mlp_down_weight_expert");
    nameList.push_back("in_mlp_down_bias_expert");
    nameList.push_back("in_mlp_down_descale_expert");
    nameList.push_back("in_mlp_down_offset_expert");
    nameList.push_back("in_mlp_down_scale_expert");
    nameList.push_back("in_mlp_down_compress_idx_expert");
    nameList.push_back("in_expert_array");
    if (param.enableLoadBalance) {
        nameList.push_back("in_fake_selected_experts");
    } else if (param.enableEPWB) {
        nameList.push_back("intermediate_selected_experts_routed");
    } else if (param.enableGatingDp) {
        nameList.push_back("intermediate_selected_experts_all");
    } else {
        nameList.push_back("intermediate_selected_experts");
    }
    if (!param.enableGatingDp) {
        if (param.processLogits != "none") {
            std::string routerWeightsTopkReducedName;
            if (param.enableMoeDistribute) {
                if (param.mixSharedRouting) {
                    routerWeightsTopkReducedName = "intermediate_router_weights_topk_reduced_mix_shared";
                } else {
                    routerWeightsTopkReducedName = "intermediate_router_weights_topk_reduced_fp32";
                }
            } else if (!param.enableMoeDistribute) {
                routerWeightsTopkReducedName = "intermediate_router_weights_topk_reduced_fp16";
            } else {
                if (param.mixSharedRouting) {
                    routerWeightsTopkReducedName = "intermediate_router_weights_topk_reduced_mix_shared";
                } else {
                    routerWeightsTopkReducedName = "intermediate_router_weights_topk_reduced";
                }
            }
            nameList.push_back(routerWeightsTopkReducedName);
        } else {
            if (param.enableMoeDistribute && param.enableDispatchCombineV2) {
                nameList.push_back("intermediate_router_weights_topk_fp32");
            } else {
                nameList.push_back("intermediate_router_weights_topk");
            }
        }
    } else {
        nameList.push_back("intermediate_router_weights_topk_reduced_all");
    }
    nameList.push_back("in_one_hot");
    nameList.push_back("in_zero_hot");
    if (param.hasMoeEp) {
        nameList.push_back("in_start_expert_idx");
        nameList.push_back("in_device_expert_count");
        nameList.push_back("in_padding_idx");
        if (param.isDynamicEp) {
            nameList.push_back("in_dynamic_ep_idx");
            nameList.push_back("in_moe_idx");
        }
    }
    return nameList;
}

atb::Status GetoutTensorIdx(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    if (param.enableEPWB) {
        if (param.mixSharedRouting) {
            CHECK_OPERATION_STATUS_RETURN(CreateConcatExpertOperation(tensorMap, param, opGraph));
        }
        SetExpertRoutingMap(tensorMap, param, opGraph);
    }
    atb::Node expertNode;
    atb_speed::common::DynamicEpMoEParam dynamicExpertParam;
    SetDynamicExpertParam(dynamicExpertParam, param);
    atb_speed::common::CreateDynamicEpMoEOperation(dynamicExpertParam, &expertNode.operation);
    expertNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_rout")};
    if (param.enableExpertCumSumOutput) {
        expertNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "out_gmm_cumsum_list"));
    }
    std::list<std::string> nameList = GetOutTensorName(param);
    for (auto iter = nameList.cbegin(); iter != nameList.cend(); ++iter) {
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, *iter));
    }
    if (param.enableEPWB) {
        int selected_expert_idx = 0;
        for (const auto& name : nameList) {
            if (name.find("selected_expert") != std::string::npos) {
                break;
            }
            selected_expert_idx++;
        }
        expertNode.inTensorReshapeFuncs.resize(expertNode.inTensorIds.size());
        if (!param.mixSharedRouting) {
            expertNode.inTensorReshapeFuncs[selected_expert_idx] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 2; // 2: dimNum
                newShape.dims[0] = oldShape.dims[0] / param.num.at(0);
                newShape.dims[1] = param.num.at(0);
            };
        } else {
            expertNode.inTensorReshapeFuncs[selected_expert_idx] = [=](const atb::Dims &oldShape, atb::Dims &newShape) { // 14: topk
                newShape.dimNum = 2; // 2: dimNum
                newShape.dims[0] = oldShape.dims[0] / (param.num.at(0) + 1);
                newShape.dims[1] = param.num.at(0) + 1;
            };
        }
    }
    opGraph.nodes.push_back(expertNode);
    ATB_SPEED_LOG_DEBUG("Expert Group calculation success5");
    return atb::NO_ERROR;
}

atb::Status RoutingBlock(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph)
{
    if (param.enableFusedTopk) {
        CHECK_OPERATION_STATUS_RETURN(CreateFusedAddTopkDiv(tensorMap, param, opGraph));
        if (param.mixSharedRouting) {
            CHECK_OPERATION_STATUS_RETURN(CreateConcatWeightOperation(tensorMap, opGraph));
        }
    } else {
        if (param.routingMethod == "deviceLimited") {
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoesoftMax(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateGroupOperation(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoetopK(tensorMap, param, opGraph));
        } else if (param.routingMethod == "integratedSoftmaxTopK") {
            CHECK_OPERATION_STATUS_RETURN(CreateSparseTopkSoftMax(tensorMap, param, opGraph));
        } else if (param.routingMethod == "noAuxTc") {
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoeSigmoid(tensorMap, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateScoreAdd(tensorMap, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateGroupOperation(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoetopK(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoetopKGather(tensorMap, opGraph));
        } else if (param.routingMethod == "topkFused") {
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoesoftMax(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateScoreAdd(tensorMap, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoetopK(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoetopKGather(tensorMap, opGraph));
        } else {
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoesoftMax(tensorMap, param, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoetopK(tensorMap, param, opGraph));
        }
    }
    ATB_SPEED_LOG_DEBUG("Routing Block success");
    return atb::NO_ERROR;
}

atb::Status CreateConcatWeightOperation(
    std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph)
{
    ATB_SPEED_LOG_DEBUG("AclNNConcat weight start");
    atb::Node aclnnConcatNode;
    atb_speed::common::AclNNConcatParam aclNNConcatParam;
    aclNNConcatParam.dim = 1;
    std::vector<std::string> concatInTensorNames;
    std::string concatOutTensorName;
    concatInTensorNames.push_back("intermediate_router_weights_topk_reduced_fp32");
    concatInTensorNames.push_back("mix_shared_routing_weight");
    concatOutTensorName = "intermediate_router_weights_topk_reduced_mix_shared";
    aclnnConcatNode.operation = new atb_speed::common::ConcatOperation("concatWeight", aclNNConcatParam);
    aclnnConcatNode.inTensorIds = GetTensorIdxList(tensorMap, concatInTensorNames);
    aclnnConcatNode.outTensorIds = {GetTensorIdx(tensorMap, concatOutTensorName)};
    opGraph.nodes.push_back(aclnnConcatNode);
    ATB_SPEED_LOG_DEBUG("AclNNConcat weight success");
    return atb::NO_ERROR;
}

atb::Status CreateConcatExpertOperation(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param,
    atb::GraphParam &opGraph)
{
    ATB_SPEED_LOG_DEBUG("AclNNConcat expert start");
    atb::Node aclnnConcatNode;
    atb_speed::common::AclNNConcatParam aclNNConcatParam;
    aclNNConcatParam.dim = 1;
    std::vector<std::string> concatInTensorNames;
    std::string concatOutTensorName;
    concatInTensorNames.push_back(param.enableGatingDp ? "intermediate_selected_experts_all" : "intermediate_selected_experts");
    concatInTensorNames.push_back("mix_shared_routing_expert");
    concatOutTensorName = "intermediate_selected_experts_mix_shared";
    aclnnConcatNode.operation = new atb_speed::common::ConcatOperation("concatExpert", aclNNConcatParam);
    aclnnConcatNode.inTensorIds = GetTensorIdxList(tensorMap, concatInTensorNames);
    aclnnConcatNode.outTensorIds = {GetTensorIdx(tensorMap, concatOutTensorName)};
    opGraph.nodes.push_back(aclnnConcatNode);
    ATB_SPEED_LOG_DEBUG("AclNNConcat expert success");
    return atb::NO_ERROR;
}

atb::Status CreateRecord(const SparseMoeParam &param, atb::GraphParam &opGraph,
                         atb_speed::EventAction eventAction, const std::string &cvKey)
{
    if (param.enableCVOverlap || param.enableGatingOverlap) {
        atb::Node recordNode;
        recordNode.inTensorIds = {};
        recordNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
            recordNode.operation,
            eventAction,
            cvKey));
        if (param.enableGatingOverlap) {
            atb::SetExecuteStreamId(recordNode.operation, STREAM1);
        }
        opGraph.nodes.push_back(recordNode);
        ATB_SPEED_LOG_DEBUG("Record event success");
    }
    return atb::NO_ERROR;
}

atb::Status CreateWait(const SparseMoeParam &param, atb::GraphParam &opGraph,
                       atb_speed::EventAction eventAction, const std::string &cvKey)
{
    if (param.enableCVOverlap || param.enableGatingOverlap) {
        atb::Node waitNode;
        waitNode.inTensorIds = {};
        waitNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
            waitNode.operation,
            eventAction,
            cvKey));
        if (param.enableGatingOverlap) {
            atb::SetExecuteStreamId(waitNode.operation, STREAM1);
        }
        opGraph.nodes.push_back(waitNode);
        ATB_SPEED_LOG_DEBUG("Wait event success");
    }
    return atb::NO_ERROR;
}

atb::Status SetTPAllGatherNode(std::map<std::string, uint32_t> tensorMap, const SparseMoeParam &param,
    atb::GraphParam &opGraph, bool isTopk)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.mlpTpParallelInfo.rank;
    allGatherParam.rankSize = param.mlpTpParallelInfo.rankIds.size();
    allGatherParam.backend = param.mlpTpParallelInfo.defaultBackend;
    param.mlpTpParallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    allGatherParam.rankTableFile = "";
    
    CreateOperation(allGatherParam, &allGatherNode.operation);

    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {(isTopk) ? "intermediate_selected_experts" : (param.enableFusedTopk || param.mixSharedRouting) ? \
            "intermediate_router_weights_topk_reduced_fp16" : "intermediate_router_weights_topk_reduced"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {(isTopk) ? "intermediate_selected_experts_with_padding_all" : \
            "intermediate_router_weights_topk_reduced_with_padding_all"});
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(allGatherNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    return atb::NO_ERROR;
}

atb::Status SetTPUnPadding(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, bool isTopk)
{
    atb::Node unpadNode;
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &unpadNode.operation));
    unpadNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {(isTopk) ? "intermediate_selected_experts_with_padding_all" : \
            "intermediate_router_weights_topk_reduced_with_padding_all", "in_attn_unpadding_idx"});
    unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {(isTopk) ? "intermediate_selected_experts_all" : "intermediate_router_weights_topk_reduced_all"});
    unpadNode.inTensorReshapeFuncs.reserve(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs.resize(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
    newShape.dimNum = 2; // 2：新shape维度为2
        if (oldShape.dimNum == 3) { // 3：旧shape维度为3
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0, 0, 1： 新shape前两维合轴
            newShape.dims[1] = oldShape.dims[2]; // 1, 2: 新shape最后一维不变
        } else {
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[1]; // 1, 2: 新shape最后一维不变
        }
    };
    opGraph.nodes.push_back(unpadNode);
    ATB_SPEED_LOG_DEBUG("AllGather calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSliceTopkOperation(std::map<std::string, uint32_t> tensorMap, atb::GraphParam &opGraph)
{
    atb::infer::SliceParam sliceParam;
    atb::Node sliceNode;
    // topk data shape is [batch_num, topk_num]. Slice topk data of the first token
    sliceParam.offsets.resize(2); // 2: Slice offset dim
    sliceParam.offsets[0] = 0;
    sliceParam.offsets[1] = 0;

    sliceParam.size.resize(2); // 2: Slice Size dim
    sliceParam.size[0] = 1;
    sliceParam.size[1] = -1;
    CreateOperation(sliceParam, &sliceNode.operation);

    sliceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_selected_experts"});
    sliceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out_selected_experts"});

    opGraph.nodes.push_back(sliceNode);
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoeOperation(const SparseMoeParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "SparseMoe";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum" << opGraph.internalTensorNum);
    if (param.enableCVOverlap) {
        CHECK_OPERATION_STATUS_RETURN(CreateRecord(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::CV_START));
    }
    if (param.routingMethod == "noAuxTc" || param.routingMethod == "topkFused") {
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoemoeGateFp32(tensorMap, param, opGraph));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoemoeGate(tensorMap, param, opGraph));
    }
    if (param.enableCVOverlap) {
        CHECK_OPERATION_STATUS_RETURN(CreateWait(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::VECTOR_CONTROL));
        CHECK_OPERATION_STATUS_RETURN(CreateRecord(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::CUBE_CONTROL));
    }
    if (param.useStdNorm) {
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoeStd(tensorMap, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoeNorm(tensorMap, opGraph));
    }
    if (param.enableGatingShift) {
        CheckPositive(param.deviceExpert.size());
        CHECK_OPERATION_STATUS_RETURN(CreateSplit(tensorMap, param, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateConcat(tensorMap, param, opGraph));
    }
    CHECK_OPERATION_STATUS_RETURN(RoutingBlock(tensorMap, param, opGraph));
    if (param.enableTopkOutput) {
        CHECK_OPERATION_STATUS_RETURN(CreateSliceTopkOperation(tensorMap, opGraph));
    }
    if (!param.enableFusedTopk && param.processLogits != "none") {
        if (param.processLogits == "normalization") {
            // In_tensor[0]: router_weights: Batch * Seq; 2
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoereduce(tensorMap, opGraph));
            // In_tensor[0]: router_weights: Batch * Seq; 2
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoedivide(tensorMap, opGraph));
        } else if (param.processLogits == "scaling") {
            CHECK_OPERATION_STATUS_RETURN(CreateElewiseMuls(tensorMap, param, opGraph));
        } else if (param.processLogits == "norm") {
            CHECK_OPERATION_STATUS_RETURN(CreateVectorNorm(tensorMap, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoedivide(tensorMap, opGraph));
        } else if (param.processLogits == "normScaling") {
            // In_tensor[0]: router_weights: Batch * Seq; 2
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoereduce(tensorMap, opGraph));
            // In_tensor[0]: router_weights: Batch * Seq; 2
            CHECK_OPERATION_STATUS_RETURN(CreateSparseMoedivide(tensorMap, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateElewiseMuls(tensorMap, param, opGraph));
        }
        CHECK_OPERATION_STATUS_RETURN(CreateFp32Cast(tensorMap, param, opGraph));
        if (param.mixSharedRouting) {
            CHECK_OPERATION_STATUS_RETURN(CreateConcatWeightOperation(tensorMap, opGraph));
        }
    }
    if (!param.enableMoeDistribute && (param.processLogits != "none")) {
        CHECK_OPERATION_STATUS_RETURN(CreateFp16Cast(tensorMap, param, opGraph));
    }
    if (param.processLogits == "none" && param.enableMoeDistribute && param.enableDispatchCombineV2) {
        CHECK_OPERATION_STATUS_RETURN(CreateFp32Cast(tensorMap, param, opGraph));
    }
    if (param.enableGatingOverlap) {
        CHECK_OPERATION_STATUS_RETURN(CreateRecord(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::COMP_CONTROL));
        CHECK_OPERATION_STATUS_RETURN(CreateWait(
            param, opGraph, atb_speed::EventAction::POP, atb_speed::common::COMM_CONTROL));
    }

    if (param.enableGatingDp) {
        CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(tensorMap, param, opGraph, true));
        CHECK_OPERATION_STATUS_RETURN(SetTPUnPadding(tensorMap, opGraph, true));
        CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(tensorMap, param, opGraph, false));
        CHECK_OPERATION_STATUS_RETURN(SetTPUnPadding(tensorMap, opGraph, false));
    }

    CHECK_OPERATION_STATUS_RETURN(GetoutTensorIdx(tensorMap, param, opGraph));

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableExpertCumSumOutput && param.enableTopkOutput) {
            SetOutTensorDescsForEPLB(outTensorDescs, 1, param.numOfDeviceExperts);
            SetOutTensorDescsForEPLB(outTensorDescs, 2, param.num.at(0), false);
        }  else if (param.enableExpertCumSumOutput) {
            SetOutTensorDescsForEPLB(outTensorDescs, 1, param.numOfDeviceExperts);
        } else if (param.enableTopkOutput) {
            SetOutTensorDescsForEPLB(outTensorDescs, 1, param.num.at(0), false);
        }
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}
}
} // namespace atb_speed