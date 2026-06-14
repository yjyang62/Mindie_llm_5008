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

#include "dynamic_ep_moe.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/aclnn/ops/argsort_operation.h"
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "operations/fusion/moe/moe_mlp.h"
#include "data_preparation.h"
#include "all_to_all_meta.h"
#include "all_to_all_dispatch.h"
#include "all_to_all_collect.h"
#include "operations/fusion/utils.h"
#include "operations/aclnn/utils/utils.h"
#include "fused_alltoall_gmm.h"


namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetDynamicEpMoEInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dynamicEpMoEInTensorCandidates = {
        {"default", {
            "in_hiddenstatus", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
            "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array", "in_selected_experts",
            "in_expert_weight", "in_one_hot", "in_zero_hot"}
        },
        {"ep", {
            "in_start_expert_idx", "in_device_expert_count", "in_padding_idx"}
        },
        {"dynamic_ep", {
            "in_buffer_idx", "in_moe_idx"}
        },
    };
    return dynamicEpMoEInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDynamicEpMoEInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dynamicEpMoEInterTensorCandidates = {
        {"default", {
            "intermediate_shuffle_idx", "intermediate_expert_shuffle_idx", "intermediate_valid_idx",
            "intermediate_buffer_idx", "intermediate_shuffle_idx_1",
            "intermediate_shuffle_idx_2", "intermediate_expert_shuffle_idx_1",
            "intermediate_group_count", "intermediate_shuffle_weight", "intermediate_recv_hiddenstatus",
            "intermediate_recv_selected_experts", "intermediate_experts_weight", "intermediate_moe_output"}
        },
        {"node_base", {
            "intermediate_selected_experts_multi_node", "intermediate_expert_weight_multi_node",
            "intermediate_hiddenstatus_multi_node", "intermediate_out_hiddenstatus_multi_node"}
        },
    };
    return dynamicEpMoEInterTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDynamicEpMoEOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> dynamicEpMoEOutTensorCandidates = {
        {"default", {
            "out_hiddenstates"}
        },
    };
    return dynamicEpMoEOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructDynamicEpTensorMap(
    const DynamicEpMoEParam &param, uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto dynamicEpMoEInTensorCandidates = GetDynamicEpMoEInTensorCandidates();
    auto dynamicEpMoEInterTensorCandidates = GetDynamicEpMoEInterTensorCandidates();
    auto dynamicEpMoEOutTensorCandidates = GetDynamicEpMoEOutTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {};
    AddTensorToList(dynamicEpMoEInTensorCandidates, "default", inTensorList);
    if (param.hasMoeEp) {
        AddTensorToList(dynamicEpMoEInTensorCandidates, "ep", inTensorList);
        if (param.isDynamicEp) {
            AddTensorToList(dynamicEpMoEInTensorCandidates, "dynamic_ep", inTensorList);
        }
    }
    if (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute && !param.enableLcocAll2All) {
        AddTensorToList(dynamicEpMoEInterTensorCandidates, "default", interTensorList);
        if (param.enableNodeBaseAll2All) {
            AddTensorToList(dynamicEpMoEInterTensorCandidates, "node_base", interTensorList);
        }
    }
    AddTensorToList(dynamicEpMoEOutTensorCandidates, "default", outTensorList);
    if (param.enableExpertCumSumOutput) {
        outTensorList.push_back("out_gmm_cumsum_list");
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();
    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateFusedAllToAllMlp(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &expertNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::All2AllMatmulParam mlpExpertParam;
    mlpExpertParam.topk = param.topk;
    mlpExpertParam.scaledTopk = param.scaledTopk;
    mlpExpertParam.numOfDeviceExperts = param.numOfDeviceExperts;
    mlpExpertParam.numOfExperts = param.numOfExperts;
    mlpExpertParam.gateUpTransposeB = param.gateUpTransposeB;
    mlpExpertParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;
    mlpExpertParam.downTransposeB = param.downTransposeB;
    mlpExpertParam.moeEpRank = param.moeEpParallelInfo.rank;
    mlpExpertParam.moeEpSize = param.moeEpParallelInfo.rankIds.size();
    mlpExpertParam.lcclMoeEpDomain = param.lcclMoeEpDomain;
    mlpExpertParam.lcclMoeEpHcclComm = param.lcclMoeEpHcclComm;
    atb_speed::common::CreateAll2AllMatmulOperation(mlpExpertParam, &expertNode.operation);
 
    expertNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_hiddenstatus"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_descale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_offset_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_compress_idx_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert"),
        GetTensorIdx(tensorMap, "in_expert_array"),
        GetTensorIdx(tensorMap, "in_selected_experts"),
        GetTensorIdx(tensorMap, "in_expert_weight"),
        GetTensorIdx(tensorMap, "in_moe_idx"),
    };
    expertNode.outTensorIds = {GetTensorIdx(tensorMap, "out_hiddenstates")};
    if (param.enableExpertCumSumOutput) {
        expertNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "out_gmm_cumsum_list"));
    }
    ATB_SPEED_LOG_DEBUG("Expert Group calculation success");
    return atb::NO_ERROR;
}
 

atb::Status CreateDataPreparation(std::map<std::string, uint32_t> &tensorMap,
    const DynamicEpMoEParam &param, size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &dataPreparationNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::DataPreparationParam dataPreparationParam;
    dataPreparationParam.topk = param.topk;
    dataPreparationParam.numOfExperts = param.numOfExperts;
    dataPreparationParam.mixSharedRouting = param.mixSharedRouting;
    dataPreparationParam.rank = param.moeEpParallelInfo.rank;
    dataPreparationParam.worldSize = param.moeEpParallelInfo.rankIds.size();
    if (param.enableNodeBaseAll2All) {
        dataPreparationParam.startRank = (
            param.moeEpInterNodeParallelInfo.rank * param.moeEpIntraNodeParallelInfo.rankIds.size());
        dataPreparationParam.rankSize = param.moeEpIntraNodeParallelInfo.rankIds.size();
    } else {
        dataPreparationParam.startRank = 0;
        dataPreparationParam.rankSize = param.moeEpParallelInfo.rankIds.size();
    }
    atb_speed::common::CreateDataPreparationOperation(dataPreparationParam, &dataPreparationNode.operation);
    dataPreparationNode.inTensorIds = {
        GetTensorIdx(tensorMap, param.enableNodeBaseAll2All ?
            "intermediate_selected_experts_multi_node" : "in_selected_experts"),
        GetTensorIdx(tensorMap, "in_buffer_idx"),
        GetTensorIdx(tensorMap, "in_one_hot"),
        GetTensorIdx(tensorMap, "in_zero_hot")
    };
    dataPreparationNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx"),
                                        GetTensorIdx(tensorMap, "intermediate_expert_shuffle_idx"),
                                        GetTensorIdx(tensorMap, "intermediate_group_count")};
    dataPreparationNode.inTensorChunks.resize(dataPreparationNode.inTensorIds.size());

    if (param.enableNodeBaseAll2All) {
        dataPreparationNode.inTensorReshapeFuncs.resize(dataPreparationNode.inTensorIds.size());
        dataPreparationNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = NUM2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[NUM2];
        };
    }

    ATB_SPEED_LOG_DEBUG("dataPreparationNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateAllToAllMeta(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &allToAllMetaNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AllToAllMetaParam allToAllMetaParam;
    allToAllMetaParam.topk = param.topk;
    allToAllMetaParam.numOfExperts = param.numOfExperts;
    allToAllMetaParam.rank = param.moeEpParallelInfo.rank;
    allToAllMetaParam.worldSize = (
        param.enableNodeBaseAll2All ? param.moeEpIntraNodeParallelInfo.rankIds.size() :
        param.moeEpParallelInfo.rankIds.size());
    allToAllMetaParam.isBF16 = param.isBF16;
    atb_speed::common::CreateAllToAllMetaOperation(allToAllMetaParam, &allToAllMetaNode.operation);
    allToAllMetaNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_count"),
                                    GetTensorIdx(tensorMap, "in_moe_idx"),
                                    GetTensorIdx(tensorMap, "in_zero_hot")};
    allToAllMetaNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_buffer_idx"),
                                     GetTensorIdx(tensorMap, "intermediate_shuffle_weight"),
                                     GetTensorIdx(tensorMap, "intermediate_valid_idx")};
    allToAllMetaNode.inTensorChunks.resize(allToAllMetaNode.inTensorIds.size());
    ATB_SPEED_LOG_DEBUG("allToAllMetaNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdxDE(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &gatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx"),
                              GetTensorIdx(tensorMap, "intermediate_buffer_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_1")};
    ATB_SPEED_LOG_DEBUG("CreateShuffleIdxDE");
    return atb::NO_ERROR;
}

atb::Status CreateExpertShuffleIdx(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &gatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_expert_shuffle_idx"),
                              GetTensorIdx(tensorMap, "intermediate_buffer_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_expert_shuffle_idx_1")};
    ATB_SPEED_LOG_DEBUG("CreateExpertShuffleIdx");
    return atb::NO_ERROR;
}

atb::Status SetAllToAllDispatchParam(
    atb_speed::common::AllToAllDispatchParam &allToAllDispatchParam,
    const DynamicEpMoEParam &param)
{
    allToAllDispatchParam.topk = param.topk;
    allToAllDispatchParam.numOfExperts = param.numOfExperts;

    auto parallelInfo = param.enableNodeBaseAll2All ? \
        param.moeEpIntraNodeParallelInfo : param.moeEpParallelInfo;
   
    allToAllDispatchParam.backend = parallelInfo.defaultBackend;
    allToAllDispatchParam.moeEpRank = parallelInfo.rank;
    allToAllDispatchParam.moeEpSize = parallelInfo.rankIds.size();
    parallelInfo.InitCommDomain(
        allToAllDispatchParam.hcclComm, allToAllDispatchParam.moeEpDomain);

    return atb::NO_ERROR;
}

atb::Status CreateAllToAllDispatch(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &allToAllDispatchNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AllToAllDispatchParam allToAllDispatchParam;

    SetAllToAllDispatchParam(allToAllDispatchParam, param);
    atb_speed::common::CreateAllToAllDispatchOperation(allToAllDispatchParam, &allToAllDispatchNode.operation);
    allToAllDispatchNode.inTensorIds = {
        GetTensorIdx(tensorMap, param.enableNodeBaseAll2All ?
            "intermediate_hiddenstatus_multi_node" : "in_hiddenstatus"),
        GetTensorIdx(tensorMap, param.enableNodeBaseAll2All ?
            "intermediate_selected_experts_multi_node" : "in_selected_experts"),
        GetTensorIdx(tensorMap, param.enableNodeBaseAll2All ?
            "intermediate_expert_weight_multi_node" : "in_expert_weight"),
        GetTensorIdx(tensorMap, "intermediate_shuffle_idx_1"),
        GetTensorIdx(tensorMap, "intermediate_expert_shuffle_idx_1"),
        GetTensorIdx(tensorMap, "in_zero_hot"),
        GetTensorIdx(tensorMap, "in_one_hot"),
    };

    allToAllDispatchNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_recv_hiddenstatus"),
        GetTensorIdx(tensorMap, "intermediate_recv_selected_experts"),
        GetTensorIdx(tensorMap, "intermediate_experts_weight"),
    };

    allToAllDispatchNode.inTensorChunks.resize(allToAllDispatchNode.inTensorIds.size());

    if (param.enableNodeBaseAll2All) {
        allToAllDispatchNode.inTensorReshapeFuncs.resize(allToAllDispatchNode.inTensorIds.size());
        allToAllDispatchNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = NUM2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[NUM2];
        };
        allToAllDispatchNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = NUM2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[NUM2];
        };
        allToAllDispatchNode.inTensorReshapeFuncs[NUM2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = NUM2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[NUM2];
        };
    }

    ATB_SPEED_LOG_DEBUG("allToAllDispatchNode calculation success");
    return atb::NO_ERROR;
}

atb::Status SetMoeMlpParam(atb_speed::common::MoeMlpParam &mlpExpertParam, const DynamicEpMoEParam &param)
{
    if (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) {
        mlpExpertParam.topk = 1;
    } else {
        mlpExpertParam.topk = param.topk;
        mlpExpertParam.scaledTopk = param.scaledTopk;
        mlpExpertParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
    }
    mlpExpertParam.numOfDeviceExperts = param.numOfDeviceExperts;
    mlpExpertParam.hasMoeEp = param.hasMoeEp;
    mlpExpertParam.deviceExpert = param.deviceExpert;
    mlpExpertParam.expertParallelDegree = param.expertParallelDegree;
    mlpExpertParam.transpose = param.transpose;
    mlpExpertParam.numOfExperts = param.numOfExperts;
    mlpExpertParam.supportSwiGLU = param.supportSwiGLU;
    mlpExpertParam.moeLinearQuantType = param.moeLinearQuantType;
    mlpExpertParam.packQuantType = param.packQuantType;
    mlpExpertParam.denseQuantType = param.denseQuantType;
    mlpExpertParam.isBF16 = param.isBF16;
    mlpExpertParam.gateUpTransposeB = param.gateUpTransposeB;
    mlpExpertParam.downTransposeB = param.downTransposeB;
    mlpExpertParam.enableFusedRouting = param.enableFusedRouting;
    mlpExpertParam.enableInitQuant = param.enableInitQuant;
    mlpExpertParam.enableSwigluQuant = param.enableSwigluQuant;
    mlpExpertParam.enableAtlasGMMFused = param.enableAtlasGMMFused;
    mlpExpertParam.quantGroupSize = param.quantGroupSize;
    mlpExpertParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    mlpExpertParam.enableCVOverlap = param.enableCVOverlap;

    mlpExpertParam.backend = param.moeEpParallelInfo.defaultBackend;
    mlpExpertParam.hasMoeEp = param.hasMoeEp;
    mlpExpertParam.moeEpRank = param.moeEpParallelInfo.rank;
    mlpExpertParam.moeEpSize = param.moeEpParallelInfo.rankIds.size();
    mlpExpertParam.moeEpDomain = param.dispatchAndCombinecommDomain;
    mlpExpertParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    if (param.expertParallelDegree == 1 && !param.enableEPWB) {
        mlpExpertParam.shiftedTopK = true;
    }
    mlpExpertParam.enableMoeDistribute = param.enableMoeDistribute && param.isDynamicEp;
    mlpExpertParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;
    mlpExpertParam.enableGatingDp = param.enableGatingDp;
    mlpExpertParam.enableDispatchCombineV2 = param.enableDispatchCombineV2;
    mlpExpertParam.numDanglingSharedExperts = param.numDanglingSharedExperts;
    mlpExpertParam.numOfRedundantExpert = param.numOfRedundantExpert;
    return atb::NO_ERROR;
}

atb::Status CreateMoeMlp(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &expertNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeMlpParam mlpExpertParam;
    SetMoeMlpParam(mlpExpertParam, param);
    atb_speed::common::CreateMoeMlpOperation(mlpExpertParam, &expertNode.operation);
    expertNode.outTensorIds = {GetTensorIdx(tensorMap,
        (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) ? \
        "intermediate_moe_output" : "out_hiddenstates")};
    if (param.enableExpertCumSumOutput) {
        expertNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "out_gmm_cumsum_list"));
    }

    expertNode.inTensorIds = {GetTensorIdx(tensorMap,
        (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) ? \
        "intermediate_recv_hiddenstatus" : "in_hiddenstatus"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_bias_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_descale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_offset_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_gateup_compress_idx_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
        GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert"),
        GetTensorIdx(tensorMap, "in_expert_array"),
        GetTensorIdx(tensorMap, (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) ? \
        "intermediate_recv_selected_experts" : "in_selected_experts"),
        GetTensorIdx(tensorMap, (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) ? \
        "in_expert_array" : "in_expert_weight")};
    if (param.hasMoeEp) {
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_zero_hot"));
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_start_expert_idx"));
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_device_expert_count"));
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_padding_idx"));
    }
    ATB_SPEED_LOG_DEBUG("Expert Group calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateShuffleIdxDE2(std::map<std::string, uint32_t> &tensorMap, size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &gatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx"),
                                GetTensorIdx(tensorMap, "intermediate_buffer_idx")};
    gatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_idx_2")};

    ATB_SPEED_LOG_DEBUG("CreateShuffleIdxDE");
    return atb::NO_ERROR;
}

atb::Status CreateShuffleWeight(std::map<std::string, uint32_t> &tensorMap,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    atb::Node &node = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &node.operation));

    node.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_experts_weight"),
        GetTensorIdx(tensorMap, "intermediate_shuffle_weight")};
    node.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shuffle_weight")};

    node.inTensorReshapeFuncs.resize(node.inTensorIds.size());
    node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("CreateShuffleWeight");
    return atb::NO_ERROR;
}

atb::Status CreateAllToAllCollect(std::map<std::string, uint32_t> &tensorMap, const DynamicEpMoEParam &param,
    size_t &nodeId, atb::GraphParam &opGraph)
{
    auto &allToAllCollectNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AllToAllCollectParam allToAllCollectParam;

    allToAllCollectParam.topk = param.topk;
    allToAllCollectParam.numOfExperts = param.numOfExperts;

    auto parallelInfo = param.enableNodeBaseAll2All ? \
        param.moeEpIntraNodeParallelInfo : param.moeEpParallelInfo;
    allToAllCollectParam.backend = parallelInfo.defaultBackend;
    allToAllCollectParam.moeEpRank = parallelInfo.rank;
    allToAllCollectParam.moeEpSize = parallelInfo.rankIds.size();
    parallelInfo.InitCommDomain(
        allToAllCollectParam.hcclComm, allToAllCollectParam.moeEpDomain);

    atb_speed::common::CreateAllToAllCollectOperation(allToAllCollectParam, &allToAllCollectNode.operation);
    allToAllCollectNode.inTensorIds = {
        GetTensorIdx(tensorMap, param.enableNodeBaseAll2All ? "intermediate_hiddenstatus_multi_node" : "in_hiddenstatus"),
        GetTensorIdx(tensorMap, "intermediate_moe_output"),
        GetTensorIdx(tensorMap, "intermediate_shuffle_weight"),
        GetTensorIdx(tensorMap, "intermediate_shuffle_idx_1"),
        GetTensorIdx(tensorMap, "intermediate_valid_idx")
    };
    allToAllCollectNode.outTensorIds = {
        GetTensorIdx(tensorMap, param.enableNodeBaseAll2All ?
            "intermediate_out_hiddenstatus_multi_node" : "out_hiddenstates")
    };
    allToAllCollectNode.inTensorChunks.resize(allToAllCollectNode.inTensorIds.size());

    if (param.enableNodeBaseAll2All) {
        allToAllCollectNode.inTensorReshapeFuncs.resize(allToAllCollectNode.inTensorIds.size());
        allToAllCollectNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = NUM2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[NUM2];
        };
    }

    ATB_SPEED_LOG_DEBUG("allToAllCollectNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateMultiNodeInfoGather(
    std::map<std::string, uint32_t> &tensorMap,
    const DynamicEpMoEParam &param,
    size_t &nodeId,
    atb::GraphParam &opGraph)
{
    auto &hAllGatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllGatherParam hAllGatherParam;
    hAllGatherParam.rank = param.moeEpInterNodeParallelInfo.rank;
    hAllGatherParam.rankSize = param.moeEpInterNodeParallelInfo.rankIds.size();
    hAllGatherParam.backend = param.moeEpInterNodeParallelInfo.defaultBackend;
    param.moeEpInterNodeParallelInfo.InitCommDomain(
        hAllGatherParam.hcclComm, hAllGatherParam.commDomain);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(hAllGatherParam, &hAllGatherNode.operation));
    hAllGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"in_hiddenstatus"});
    hAllGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_hiddenstatus_multi_node"});

    auto &kAllGatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllGatherParam kAllGatherParam;
    kAllGatherParam.rank = param.moeEpInterNodeParallelInfo.rank;
    kAllGatherParam.rankSize = param.moeEpInterNodeParallelInfo.rankIds.size();
    kAllGatherParam.backend = param.moeEpInterNodeParallelInfo.defaultBackend;
    param.moeEpInterNodeParallelInfo.InitCommDomain(
        kAllGatherParam.hcclComm, kAllGatherParam.commDomain);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(kAllGatherParam, &kAllGatherNode.operation));
    kAllGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"in_selected_experts"});
    kAllGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_selected_experts_multi_node"});

    auto &wAllGatherNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllGatherParam wAllGatherParam;
    wAllGatherParam.rank = param.moeEpInterNodeParallelInfo.rank;
    wAllGatherParam.rankSize = param.moeEpInterNodeParallelInfo.rankIds.size();
    wAllGatherParam.backend = param.moeEpInterNodeParallelInfo.defaultBackend;
    param.moeEpInterNodeParallelInfo.InitCommDomain(
        wAllGatherParam.hcclComm, wAllGatherParam.commDomain);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(wAllGatherParam, &wAllGatherNode.operation));
    wAllGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"in_expert_weight"});
    wAllGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_expert_weight_multi_node"});

    return atb::NO_ERROR;
}

atb::Status CreateMultiNodeInfoScatter(
    std::map<std::string, uint32_t> &tensorMap,
    const DynamicEpMoEParam &param,
    size_t &nodeId,
    atb::GraphParam &opGraph)
{
    auto &reduceScatterNode = opGraph.nodes.at(nodeId++);
    atb::infer::ReduceScatterParam reduceScatterParam;
    reduceScatterParam.rank = param.moeEpInterNodeParallelInfo.rank;
    reduceScatterParam.rankSize = param.moeEpInterNodeParallelInfo.rankIds.size();
    reduceScatterParam.backend = param.moeEpInterNodeParallelInfo.defaultBackend;
    param.moeEpInterNodeParallelInfo.InitCommDomain(
        reduceScatterParam.hcclComm, reduceScatterParam.commDomain);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(reduceScatterParam, &reduceScatterNode.operation));
    reduceScatterNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_out_hiddenstatus_multi_node"});
    reduceScatterNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"out_hiddenstates"});

    return atb::NO_ERROR;
}

atb::Status CreateAllToAll(
    std::map<std::string, uint32_t> &tensorMap,
    const DynamicEpMoEParam &param,
    size_t &nodeId,
    atb::GraphParam &opGraph)
{
    if (param.enableNodeBaseAll2All) {
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::AddDapEventsBeforeComm(opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateMultiNodeInfoGather(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::AddDapEventsAfterComm(opGraph));
    }
    CHECK_OPERATION_STATUS_RETURN(CreateDataPreparation(tensorMap, param, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateAllToAllMeta(tensorMap, param, nodeId, opGraph));

    CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdxDE(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateExpertShuffleIdx(tensorMap, nodeId, opGraph));

    CHECK_OPERATION_STATUS_RETURN(CreateAllToAllDispatch(tensorMap, param, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateMoeMlp(tensorMap, param, nodeId, opGraph));

    CHECK_OPERATION_STATUS_RETURN(CreateShuffleWeight(tensorMap, nodeId, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateShuffleIdxDE2(tensorMap, nodeId, opGraph));  // 内存复写
    CHECK_OPERATION_STATUS_RETURN(CreateAllToAllCollect(tensorMap, param, nodeId, opGraph));
    if (param.enableNodeBaseAll2All) {
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::AddDapEventsBeforeComm(opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateMultiNodeInfoScatter(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::AddDapEventsAfterComm(opGraph));
    }
    ATB_SPEED_LOG_DEBUG("CreateAllToAll");
    return atb::NO_ERROR;
}

atb::Status CreateDynamicEpMoEOperation(const DynamicEpMoEParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG("CreateDynamicEpMoEOperation Start");
    atb::GraphParam opGraph;
    opGraph.name = "DynamicEpMoE";
    std::map<std::string, uint32_t> tensorMap = ConstructDynamicEpTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    uint64_t nodeCount = 1;
    if (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute && !param.enableLcocAll2All) {
        nodeCount = 9; // 9: ep level2 时使用9个节点
        if (param.enableNodeBaseAll2All) {
            nodeCount += NUM4;
        }
    }
    opGraph.nodes.resize(nodeCount);

    size_t nodeId = 0;
    if (param.enableLcocAll2All && param.isDynamicEp) {
        // alltoall GMM operation
        /*
            1. InitRoutingQuant
            2. Allgather Matrix
            3. AlltoallGMM
            4  DequantSwigluQuant
            5  GMMAlltoall
            6  MoeTokenUnpermute
        */
        CHECK_OPERATION_STATUS_RETURN(CreateFusedAllToAllMlp(tensorMap, param, nodeId, opGraph));
    } else if (param.hasMoeEp && param.isDynamicEp && !param.enableMoeDistribute) {
        CHECK_OPERATION_STATUS_RETURN(CreateAllToAll(tensorMap, param, nodeId, opGraph));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateMoeMlp(tensorMap, param, nodeId, opGraph));
    }

    opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableExpertCumSumOutput) {
            outTensorDescs.at(1) = atb::TensorDesc{};
            outTensorDescs.at(1).format = ACL_FORMAT_ND;
            outTensorDescs.at(1).shape.dimNum = 1;
            outTensorDescs.at(1).dtype = ACL_INT64;
            outTensorDescs.at(1).shape.dims[0] = param.numOfDeviceExperts;
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    ATB_SPEED_LOG_DEBUG("CreateDynamicEpMoEOperation seccess");
    return atb::NO_ERROR;
}
}
} // namespace atb_speed