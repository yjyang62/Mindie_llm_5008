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
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/aclnn/ops/add_rms_norm_operation.h"
#include "models/qwen/operation/qwen_mlp_shared_expert.h"
#include "models/qwen/layer/moe_decoder_layer.h"

namespace atb_speed {
namespace qwen {
static const uint64_t IN_TENSOR_COUNT = 69;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 8;
static const uint64_t NUM2 = 2;

std::map<std::string, std::vector<std::string>> GetQwenMoeLayerInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenMoeLayerInTensorCandidates = {
        {"default_weight", {
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
            "in_qkv_weight_0", "in_qkv_bias_0", "in_qkv_descale_0",
            "in_qkv_offset_0", "in_qkv_scale_0", "in_qkv_compress_idx_0",
            "in_qkv_weight_1", "in_qkv_bias_1", "in_qkv_descale_1", "in_qkv_offset_1",
            "in_qkv_scale_1", "in_qkv_compress_idx_1",
            "in_qkv_weight_2", "in_qkv_bias_2", "in_qkv_descale_2", "in_qkv_offset_2",
            "in_qkv_scale_2", "in_qkv_compress_idx_2",
            "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale",
            "in_attention_out_offset", "in_attention_out_scale", "in_attention_out_compress_idx",
            "in_q_norm_weight", "in_k_norm_weight",
            "in_selfattention_out_norm_weight", "in_selfattention_out_norm_bias",
            "in_selfattention_out_new_norm_weight", "in_selfattention_out_new_norm_bias",
            "in_block_sparse_moe_gate_weight", "in_block_sparse_moe_gate_bias",
            "in_block_sparse_moe_gate_descale", "in_block_sparse_moe_gate_offset",
            "in_block_sparse_moe_gate_scale", "in_block_sparse_moe_gate_compress_idx",
            "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert", "in_mlp_gateup_descale_expert",
            "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert", "in_mlp_gateup_compress_idx_expert",
            "in_mlp_down_weight_expert", "in_mlp_down_bias_expert", "in_mlp_down_descale_expert",
            "in_mlp_down_offset_expert", "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert",
            "in_mlp_shared_gateup_weight", "in_mlp_shared_down_weight", "in_mlp_shared_expert_gate"
        }},
        {"default", {
            "in_hidden_states", "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
            "in_cos_table", "in_sin_table", "in_attention_mask", "in_k_cache", "in_v_cache",
            "in_seq_len", "in_place_holder", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"}},
        {"parallel_input", {
            "in_attn_padding_idx", "in_attn_unpadding_idx", "in_ffn_padding_idx",
            "in_ffn_unpadding_idx", "in_lm_head_skip_padding_token_indices",
            "in_attention_padding_idx_slice", "in_start_expert_idx",
            "in_device_expert_count", "in_lty_idx", "in_moe_idx"}},
        {"epwb", {
            "in_expert_routing_map"}},
        {"force_load_balance", {
            "in_fake_topk"
        }},
        {"q_len", {"in_q_len"}},
    };
    return qwenMoeLayerInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetQwenMoeLayerIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenMoeLayerIntermediateTensorCandidates = {
        {"default", {
            "intermediate_attention_out", "intermediate_selfattention_norm_out", "intermediate_attention_out_add"
        }},
        {"moe", {
            "intermediate_moe_out"
        }},
        {"mlp", {
            "intermediate_mlp_out"
        }},
        {"shared_expert", {
            "intermediate_shared_experts_out"
        }},
        {"enable_inter_layer_add_norm", {
            "intermediate_selfattention_norm_rstd_out"
        }},
        {"attn_comm", {
            "intermediate_attention_out_padding", "intermediate_selfattention_norm_out_fp32",
            "intermediate_selfattention_norm_out_padding"
        }},
        {"attn_reducescatter", {
            "intermediate_attention_out_scatter",
        }},
        {"ffn_allgather", {
            "intermediate_selfattention_norm_all_gather_out"
        }},
        {"ffn_comm", {
            "intermediate_moe_out_padding"
        }},
        {"ffn_reducescatter", {
            "intermediate_mlp_rs_out"
        }},
        {"attn_allgather", {
            "intermediate_mlp_all_gather_out"
        }},
        {"epwb", {
            "intermediate_expert_routing_map"
        }},

    };
    return qwenMoeLayerIntermediateTensorCandidates;
}

atb::Status QwenMoeConstructIntermediateTensorMap(const MoeDecoderLayerParam &param,
    std::map<std::string, std::vector<std::string>> &qwenMoeLayerIntermediateTensorCandidates,
    std::vector<std::string> &intermediateTensorList)
{
    atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates, "default", intermediateTensorList);
    if (param.hasMoe) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates, "moe", intermediateTensorList);
    }
    if (param.hasSharedExpert) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                           "shared_expert", intermediateTensorList);
    }
    if (!param.hasAttnComm && param.enableIntraLayerAddNorm) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                           "enable_inter_layer_add_norm", intermediateTensorList);
    }
    if (param.hasFfnComm || param.ffnAllreduce) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                           "mlp", intermediateTensorList);
    }
    if (param.hasAttnComm) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                           "attn_comm", intermediateTensorList);
        if (param.attnReduceScatter) {
            atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                               "attn_reducescatter", intermediateTensorList);
        }
        if (param.ffnAllGather) {
            atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                               "ffn_allgather", intermediateTensorList);
        }
    }
    if (param.hasFfnComm) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                           "ffn_comm", intermediateTensorList);
        if (param.ffnReduceScatter) {
            atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                               "ffn_reducescatter", intermediateTensorList);
        }
        if (param.attnAllGather) {
            atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates,
                                               "attn_allgather", intermediateTensorList);
        }
    }
    if (param.enableEPWB) {
        atb_speed::common::AddTensorToList(qwenMoeLayerIntermediateTensorCandidates, "epwb", intermediateTensorList);
    }
    return atb::NO_ERROR;
}

std::map<std::string, uint32_t> QwenMoeConstructTensorMap(
    const MoeDecoderLayerParam &param, uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto qwenMoeLayerInTensorCandidates = GetQwenMoeLayerInTensorCandidates();
    auto qwenMoeLayerIntermediateTensorCandidates = GetQwenMoeLayerIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out_decoder_layer"};

    if (!param.isDenseLayer && param.enableExpertCumSumOutput) {
        outTensorList.push_back("out_gmm_cumsum_list");
    }

    atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "default_weight", inTensorList);
    atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "default", inTensorList);
    atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "parallel_input", inTensorList);

    if (param.enableEPWB) {
        atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "epwb", inTensorList);
    }
    if (param.enableLoadBalance) {
        atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "force_load_balance", inTensorList);
    }
    // 添加prefix_cache所需的Tensor
    if (param.enableSpeculate || param.enableSplitFuse) {
        atb_speed::common::AddTensorToList(qwenMoeLayerInTensorCandidates, "q_len", inTensorList);
    }
    QwenMoeConstructIntermediateTensorMap(param, qwenMoeLayerIntermediateTensorCandidates, intermediateTensorList);

    inTensorNum = inTensorList.size();
    internalTensorNum = intermediateTensorList.size();
    outTensorNum = outTensorList.size();
    ATB_SPEED_LOG_DEBUG("ConstructTensorMap done");
    return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

void SetFusionAttentionAclNNIncreAttentionParam(
    const MoeDecoderLayerParam &param,
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    fusionAttentionParam.aclnnIncreAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.aclnnIncreAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.aclnnIncreAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.aclnnIncreAttentionParam.hasMask = true;
    fusionAttentionParam.aclnnIncreAttentionParam.isFA = param.isFA;
    fusionAttentionParam.aclnnIncreAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.aclnnIncreAttentionParam.hasKVQuant = param.enableKvQuant;
    if (param.enableKvQuant) {
        fusionAttentionParam.aclnnIncreAttentionParam.hasQuantOffset = param.kvQuantHasOffset;
    }
    if (param.enableSplitFuse && param.isPrefill) {
        // aclnnFusedInferAttentionScoreV3 use high precision
        fusionAttentionParam.aclnnIncreAttentionParam.inputLayoutPA = "TND";
        fusionAttentionParam.aclnnIncreAttentionParam.innerPrecise = 0;
    }
}

atb::Status SetFusionAttentionParam(
    const MoeDecoderLayerParam &param,
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = 2; // 2:设置新张量形状
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.layerLinearQuantType = param.attnLinearQuantType;
    fusionAttentionParam.layerLinearTransposeType = param.attnLinearTransposeType;
    fusionAttentionParam.packQuantType = param.packQuantType.at(0);
    fusionAttentionParam.supportLcoc = param.enableLcoc;
    fusionAttentionParam.enableSplitFuse = param.enableSplitFuse;
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.normEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.normEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.enableSpeculate || param.enableSplitFuse) {
        fusionAttentionParam.pageAttentionParam.calcType = \
            atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
        fusionAttentionParam.pageAttentionParam.maskType = \
            atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
    } else {
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    }
    fusionAttentionParam.useQKNorm = param.useQKNorm;
    fusionAttentionParam.qkvHasBias = param.linearHasBias.at(atb_speed::base::QKV_HASBIAS);

    // aclnnIncreAttention
    SetFusionAttentionAclNNIncreAttentionParam(param, fusionAttentionParam);

    return atb::NO_ERROR;
}

atb::Status SetFusionAttentionNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                                   std::map<std::string, uint32_t> &tensorMap)
{
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    SetFusionAttentionParam(param, fusionAttentionParam);
    atb::Node attentionNode;
    // 大ep场景，关闭fusion attention的输出 allreduce
    if (param.attnAllreduce) {
        fusionAttentionParam.selfOutLinearTensorParallelInfo = {
            param.rank, param.worldSize, param.backend, param.rankTableFile};
        if (param.mapping.isInitialized_) {
            atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
            parallelInfo.InitCommDomain(
                fusionAttentionParam.selfOutLinearTensorParallelInfo.hcommInfo,
                fusionAttentionParam.selfOutLinearTensorParallelInfo.commDomain);
        }
    } else {
        fusionAttentionParam.selfOutLinearTensorParallelInfo = {0, 1, param.backend, param.rankTableFile};
    }

    CHECK_OPERATION_STATUS_RETURN(Attention(fusionAttentionParam, &attentionNode.operation));
    std::vector<std::string> attnInTensorNames = {
            "in_hidden_states", "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight",
            "in_input_norm_new_bias", "in_qkv_weight_0", "in_qkv_scale_0", "in_qkv_offset_0",
            "in_qkv_descale_0", "in_qkv_bias_0", "in_qkv_compress_idx_0",
            "in_qkv_weight_1", "in_qkv_scale_1", "in_qkv_offset_1", "in_qkv_descale_1",
            "in_qkv_bias_1", "in_qkv_compress_idx_1", "in_qkv_weight_2", "in_qkv_scale_2",
            "in_qkv_offset_2", "in_qkv_descale_2", "in_qkv_bias_2", "in_qkv_compress_idx_2",
            "in_cos_table", "in_sin_table", "in_seq_len", "in_k_cache", "in_v_cache",
            "in_attention_mask", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots",
            "in_attention_out_weight", "in_attention_out_scale", "in_attention_out_offset",
            "in_attention_out_descale", "in_attention_out_bias", "in_attention_out_compress_idx"
            };
    if (param.enableSpeculate || param.enableSplitFuse) {
        attnInTensorNames.push_back("in_q_len");
    }
    if (fusionAttentionParam.useQKNorm) {
        attnInTensorNames.push_back("in_q_norm_weight");
        attnInTensorNames.push_back("in_k_norm_weight");
    }
            
    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
    opGraph.nodes.push_back(attentionNode);
    return atb::NO_ERROR;
}

atb::Status SetAttnOutPadding(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out",
                                                                             "in_attn_padding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_padding"});

    opGraph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("create SetPadding");
    return atb::NO_ERROR;
}

atb::Status SetAttnReduceScatter(atb::GraphParam &opGraph, const MoeDecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node rsNode;
    atb::infer::ReduceScatterParam rsParam;
    atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
    rsParam.rank = parallelInfo.rank;
    rsParam.rankSize = parallelInfo.rankIds.size();
    rsParam.backend = parallelInfo.defaultBackend;
    parallelInfo.InitCommDomain(rsParam.hcclComm, rsParam.commDomain);
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(rsParam, &rsNode.operation));
    rsNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_padding"});
    rsNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_scatter"});
    opGraph.nodes.push_back(rsNode);
    return atb::NO_ERROR;
}

atb::Status SetGatherPreNorm(atb::GraphParam &opGraph, const MoeDecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node gatherNormNode;
    atb::infer::GatherPreRmsNormParam gatherRmsNormParam;
    gatherRmsNormParam.epsilon = param.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherRmsNormParam, &gatherNormNode.operation));

    std::vector<std::string> outTensorNames;
    std::vector<std::string> inTensorNames;

    inTensorNames.push_back(
        param.attnReduceScatter ? "intermediate_attention_out_scatter" : "intermediate_attention_out_padding");
    inTensorNames.push_back("in_hidden_states");
    inTensorNames.push_back("in_attention_padding_idx_slice");

    if (param.normHasBias) { // FP
        inTensorNames.push_back("in_selfattention_out_norm_weight");
        inTensorNames.push_back("in_selfattention_out_new_norm_bias");
    } else {
        if (param.isAntiOutlier) {
            inTensorNames.push_back("in_selfattention_out_new_norm_weight");
        } else {
            inTensorNames.push_back("in_selfattention_out_norm_weight");
        }
    }

    outTensorNames.push_back("intermediate_selfattention_norm_out_fp32");
    outTensorNames.push_back("intermediate_attention_out_add");

    gatherNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, inTensorNames);
    gatherNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, outTensorNames);
    opGraph.nodes.push_back(gatherNormNode);
    ATB_SPEED_LOG_DEBUG("SetGatherPreNorm calculation success");

    return atb::NO_ERROR;
}

atb::Status SetCast(atb::GraphParam &opGraph, const MoeDecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node castNode;
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_selfattention_norm_out_fp32")};
    castNode.outTensorIds = {
        atb_speed::common::GetTensorIdx(tensorMap, "intermediate_selfattention_norm_out_padding")};

    opGraph.nodes.push_back(castNode);
    ATB_SPEED_LOG_DEBUG("Cast calculation success");
    return atb::NO_ERROR;
}


atb::Status SetFFNAllGather(atb::GraphParam &opGraph, const MoeDecoderLayerParam &param,
                            std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
    allGatherParam.rank = parallelInfo.rank;
    allGatherParam.rankSize = parallelInfo.rankIds.size();
    allGatherParam.backend = parallelInfo.defaultBackend;
    parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_selfattention_norm_out_padding"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_selfattention_norm_all_gather_out"});
    opGraph.nodes.push_back(allGatherNode);

    ATB_SPEED_LOG_DEBUG("AllGather calculation success");
    return atb::NO_ERROR;
}

int64_t SetFFNInUnpadding(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                          std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node unpadNode;
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &unpadNode.operation));
    unpadNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {param.ffnAllGather ? "intermediate_selfattention_norm_all_gather_out" :
            "intermediate_selfattention_norm_out_padding", "in_attn_unpadding_idx"});
    unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_selfattention_norm_out"});
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

    ATB_SPEED_LOG_DEBUG("SetFFNInUnpadding calculation success");
    return atb::NO_ERROR;
}

atb::Status SetAttentionResidualAddNode(atb::GraphParam &opGraph,
                                        std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    
    selfResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states", "intermediate_attention_out"});
    selfResidualAddNode.outTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_add"});
    opGraph.nodes.push_back(selfResidualAddNode);
    return atb::NO_ERROR;
}

atb::Status SetSelfNormNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                            std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfNormNode;
    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.normEps;
    CreateOperation(selfNormParam, &selfNormNode.operation);
    if (selfNormNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("selfNormNode op is nullptr: ");
    }
    selfNormNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_add",
                                                        "in_selfattention_out_norm_weight"});
    selfNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_selfattention_norm_out"});
    opGraph.nodes.push_back(selfNormNode);
    ATB_SPEED_LOG_DEBUG("create post normEps");
    return atb::NO_ERROR;
}

atb::Status SetAttentionResidualAddNormNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfAddNormNode;
    selfAddNormNode.operation = new atb_speed::common::AddRmsNormOperation("AddRmsNormNode", 1e-6);

    selfAddNormNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states",
                                                        "intermediate_attention_out",
                                                        "in_selfattention_out_norm_weight"});
    selfAddNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {
        "intermediate_selfattention_norm_out",
        "intermediate_selfattention_norm_rstd_out",
        "intermediate_attention_out_add"});

    opGraph.nodes.push_back(selfAddNormNode);
    ATB_SPEED_LOG_DEBUG("create SetAttentionResidualAddNormNode");
    return atb::NO_ERROR;
}

atb::Status SetMoeParam(const MoeDecoderLayerParam &param,
                        atb_speed::common::SparseMoeParam &sparseMoeParam)
{
    sparseMoeParam.enableEPWB = param.enableEPWB;
    sparseMoeParam.numOfRedundantExpert = param.numOfRedundantExpert;
    sparseMoeParam.transpose = param.transpose;
    sparseMoeParam.numOfExperts = param.numOfExperts;
    sparseMoeParam.num = param.numOfSelectedExperts;
    sparseMoeParam.isBF16 = param.isBF16;
    sparseMoeParam.expertParallelDegree = param.expertParallelDegree;
    sparseMoeParam.processLogits = param.processLogits;
    sparseMoeParam.supportSwiGLU = param.enableSwiGLU;
    sparseMoeParam.routingMethod = param.routingMethod;
    sparseMoeParam.enableFusedRouting = param.enableFusedRouting;
    sparseMoeParam.moeLinearQuantType = param.moeLinearQuantType;
    sparseMoeParam.gateUpTransposeB = param.moeLinearTransposeType[atb_speed::common::SparseMoeIdx::MOE_MLP_GATE_IDX];
    sparseMoeParam.downTransposeB = param.moeLinearTransposeType[atb_speed::common::SparseMoeIdx::MOE_MLP_DOWN_IDX];
    sparseMoeParam.packQuantType = param.packQuantType.at(1);
    sparseMoeParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    sparseMoeParam.enableMoeDistribute = !param.isPrefill && param.enableAllToAllMC2 && param.isDynamicEp;
    sparseMoeParam.isDynamicEp = param.isDynamicEp;
    sparseMoeParam.numOfDeviceExperts = param.numOfDeviceExperts;
    sparseMoeParam.enableDispatchCombineV2 = param.enableDispatchCombineV2;
    sparseMoeParam.enableInitQuant = param.enableInitQuant;
    sparseMoeParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    sparseMoeParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;

    sparseMoeParam.hasMoeEp = param.mapping.Get(base::MOE_EP).IsEnabled();
    sparseMoeParam.moeEpParallelInfo = param.mapping.Get(base::MOE_EP);
    sparseMoeParam.mlpTpParallelInfo = param.mapping.Get(base::MLP_TP);
    sparseMoeParam.moeEpInterNodeParallelInfo = param.mapping.Get(base::MOE_EP_INTER_NODE);
    sparseMoeParam.moeEpIntraNodeParallelInfo = param.mapping.Get(base::MOE_EP_INTRA_NODE);
    
    if (sparseMoeParam.enableMoeDistribute) {
        sparseMoeParam.dispatchAndCombinecommDomain = param.dispatchAndCombinecommDomain;
        sparseMoeParam.dispatchAndCombineHcclComm = param.dispatchAndCombineHcclComm;
    }
    sparseMoeParam.enableLoadBalance = param.enableLoadBalance;
    return atb::NO_ERROR;
}

int64_t SetExpertRoutingMapSlice(
    atb::GraphParam &opGraph,
    const MoeDecoderLayerParam &param, std::map<std::string,
    uint32_t> tensorMap)
{
    atb::Node sliceNode;
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets.resize(NUM2);
    sliceParam.offsets[0] = param.layerId;
    sliceParam.offsets[1] = 0;
    sliceParam.size.resize(NUM2);
    sliceParam.size[0] = 1;
    sliceParam.size[1] = -1;
    CreateOperation(sliceParam, &sliceNode.operation);
    sliceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"in_expert_routing_map"});
    sliceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_expert_routing_map"});
    opGraph.nodes.push_back(sliceNode);
    return atb::NO_ERROR;
}

atb::Status SetMoeNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                       std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node moeNode;
    atb_speed::common::SparseMoeParam sparseMoeParam;
    SetMoeParam(param, sparseMoeParam);
    atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation);
    if (moeNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("SparseMoe op is nullptr: ");
    }
    std::vector<std::string> moeInTensorNames = {
        "intermediate_selfattention_norm_out", "in_block_sparse_moe_gate_weight",
        "in_block_sparse_moe_gate_bias", "in_block_sparse_moe_gate_descale",
        "in_block_sparse_moe_gate_offset", "in_block_sparse_moe_gate_scale",
        "in_block_sparse_moe_gate_compress_idx", "in_mlp_gateup_weight_expert",
        "in_mlp_gateup_bias_expert", "in_mlp_gateup_descale_expert",
        "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
        "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
        "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
        "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array",
        "in_expert_group", "in_one_hot", "in_zero_hot"};
    if (param.enableLoadBalance) {
        moeInTensorNames.push_back("in_fake_topk");
    }
    if (param.mapping.Get(base::MOE_EP).IsEnabled()) {
        moeInTensorNames.push_back("in_start_expert_idx");
        moeInTensorNames.push_back("in_device_expert_count");
        moeInTensorNames.push_back("in_ffn_padding_idx");
        if (param.isDynamicEp) {
            moeInTensorNames.push_back("in_lty_idx");
            moeInTensorNames.push_back("in_moe_idx");
        }
    }
    if (param.enableEPWB) {
        SetExpertRoutingMapSlice(opGraph, param, tensorMap);
        moeInTensorNames.push_back("intermediate_expert_routing_map");
    }
    moeNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, moeInTensorNames);
    moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out"});
    if (param.enableExpertCumSumOutput) {
        moeNode.outTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap, {"out_gmm_cumsum_list"}));
    }
    ATB_SPEED_LOG_DEBUG("Moe sparse calculation success");
    opGraph.nodes.push_back(moeNode);
    return atb::NO_ERROR;
}

atb::Status SetShareExpertNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                               std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node shareExpertNode;
    atb_speed::qwen::QwenMlpSharedExpertParam sharedMlpExpertParam;
    sharedMlpExpertParam.transpose = param.transpose;
    sharedMlpExpertParam.hasSharedExpertGate = param.hasSharedExpertGate;
    ATB_SPEED_LOG_DEBUG("sharedMlpExpertParam success");
    qwen::CreateQwenMlpSharedExpertOperation(
        sharedMlpExpertParam, &shareExpertNode.operation);
    shareExpertNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_selfattention_norm_out",
                                                        "in_mlp_shared_gateup_weight",
                                                        "in_mlp_shared_down_weight", "in_mlp_shared_expert_gate"});
    shareExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_experts_out"});
    opGraph.nodes.push_back(shareExpertNode);
    ATB_SPEED_LOG_DEBUG("shared expert calculation success");
    return atb::NO_ERROR;
}

atb::Status SetShareAddSelectNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node shareAddSelectNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &shareAddSelectNode.operation);
    shareAddSelectNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_experts_out", "intermediate_moe_out"});
    shareAddSelectNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out"});
    ATB_SPEED_LOG_DEBUG("shared expert add success");
    opGraph.nodes.push_back(shareAddSelectNode);
    return atb::NO_ERROR;
}

atb::Status SetAllReduceNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                             std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node moeAllReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
    allReduceParam.rank = parallelInfo.rank;
    allReduceParam.rankSize = parallelInfo.rankIds.size();
    allReduceParam.backend = parallelInfo.defaultBackend;
    if (param.mapping.isInitialized_) {
        parallelInfo.InitCommDomain(allReduceParam.hcclComm, allReduceParam.commDomain);
    }
    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (moeAllReduceNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
    }
    moeAllReduceNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap,
            {param.hasMoe ? "intermediate_moe_out" : "intermediate_shared_experts_out"});
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    opGraph.nodes.push_back(moeAllReduceNode);
    ATB_SPEED_LOG_DEBUG("create all reduce");
    return atb::NO_ERROR;
}

atb::Status SetFFNOutpadding(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                             std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node unpadNode;
    atb::infer::GatherParam padParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(padParam, &unpadNode.operation));
    if (param.hasAttnComm) {
        unpadNode.inTensorIds = \
            atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out", "in_ffn_padding_idx"});
    } else {
        unpadNode.inTensorIds = \
            atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out", "in_attn_padding_idx"});
    }
    unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_padding"});
    unpadNode.inTensorReshapeFuncs.reserve(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs.resize(unpadNode.inTensorIds.size());
    opGraph.nodes.push_back(unpadNode);

    ATB_SPEED_LOG_DEBUG("SetFFNOutpadding calculation success");
    return atb::NO_ERROR;
}

atb::Status SetFFNReduceScatter(atb::GraphParam &opGraph, const MoeDecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node rsNode;
    atb::infer::ReduceScatterParam rsParam;
    atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
    rsParam.rank = parallelInfo.rank;
    rsParam.rankSize = parallelInfo.rankIds.size();
    rsParam.backend = parallelInfo.defaultBackend;
    parallelInfo.InitCommDomain(rsParam.hcclComm, rsParam.commDomain);
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(rsParam, &rsNode.operation));
    rsNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_padding"});
    rsNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_rs_out"});
    opGraph.nodes.push_back(rsNode);
    return atb::NO_ERROR;
}

atb::Status SetMlpResidualAddNode(atb::GraphParam &opGraph, const MoeDecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mlpResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    
    if (param.hasFfnComm) {
        mlpResidualAddNode.inTensorIds = \
            atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_add",
                param.ffnReduceScatter ? "intermediate_mlp_rs_out" :
                    param.hasAttnComm ? "intermediate_moe_out_padding" : "intermediate_moe_out"});
    } else {
        mlpResidualAddNode.inTensorIds = \
            atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_add",
                                        param.ffnAllreduce ? "intermediate_mlp_out" : "intermediate_moe_out"});
    }

    mlpResidualAddNode.outTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap,
            {param.hasFfnComm ? "intermediate_mlp_out" : "out_decoder_layer"});

    opGraph.nodes.push_back(mlpResidualAddNode);
    ATB_SPEED_LOG_DEBUG("decoder layer: residule create opgraph");

    return atb::NO_ERROR;
}

atb::Status SetTPAllGatherNode(atb::GraphParam &opGraph, const MoeDecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    
    if (!param.isLastLayer) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    } else {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::LM_HEAD_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    }
    CreateOperation(allGatherParam, &allGatherNode.operation);

    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {param.hasAttnComm ? "intermediate_mlp_out" : "intermediate_moe_out_padding"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_all_gather_out"});

    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(allGatherNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    return atb::NO_ERROR;
}

int64_t SetAttnInUnpadding(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node unpadNode;
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &unpadNode.operation));
    unpadNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {param.attnAllGather ? "intermediate_mlp_all_gather_out" : "intermediate_mlp_out",
         param.isLastLayer ? "in_lm_head_skip_padding_token_indices" : "in_ffn_unpadding_idx"});
    unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out_decoder_layer"});
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

    ATB_SPEED_LOG_DEBUG("SetAttnInUnpadding calculation success");
    return atb::NO_ERROR;
}


atb::Status SetPostAttnProcess(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                               std::map<std::string, uint32_t> &tensorMap)
{
    if (param.hasAttnComm) {
        // attn做reduce scatter之前进行padding
        CHECK_OPERATION_STATUS_RETURN(SetAttnOutPadding(opGraph, tensorMap));
        
        if (param.attnReduceScatter) {
            CHECK_OPERATION_STATUS_RETURN(SetAttnReduceScatter(opGraph, param, tensorMap));
        }
            // pregathernorm
        CHECK_OPERATION_STATUS_RETURN(SetGatherPreNorm(opGraph, param, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(SetCast(opGraph, param, tensorMap));
        if (param.ffnAllGather) {
            // 静态ep场景下，ffn输入做allgather
            CHECK_OPERATION_STATUS_RETURN(SetFFNAllGather(opGraph, param, tensorMap));
        }
        // attn做了rs后进行unpadding
        CHECK_OPERATION_STATUS_RETURN(SetFFNInUnpadding(param, opGraph, tensorMap));
    } else {
        if (param.enableIntraLayerAddNorm) {
            // node: residual + norm
            CHECK_OPERATION_STATUS_RETURN(SetAttentionResidualAddNormNode(opGraph, tensorMap));
        } else {
            // node1: residual
            CHECK_OPERATION_STATUS_RETURN(SetAttentionResidualAddNode(opGraph, tensorMap));
            // node2: norm
            CHECK_OPERATION_STATUS_RETURN(SetSelfNormNode(param, opGraph, tensorMap));
        }
    }
    ATB_SPEED_LOG_DEBUG("decoder layer: SetPostAttnProcess create opgraph");

    return atb::NO_ERROR;
}

atb::Status SetPostMoeProcess(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph,
                              std::map<std::string, uint32_t> &tensorMap)
{
    if (param.hasFfnComm) {
        if (param.hasAttnComm) {
            CHECK_OPERATION_STATUS_RETURN(SetFFNOutpadding(param, opGraph, tensorMap));
        }
        if (param.ffnReduceScatter) {
            // 静态ep场景下，ffn输出做reducescatter
            CHECK_OPERATION_STATUS_RETURN(SetFFNReduceScatter(opGraph, param, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAddNode(opGraph, param, tensorMap));

        if (param.attnAllGather) {
            if (!param.hasAttnComm) {
                // 如果attn部分没做padding，这里先residual add,再padding
                CHECK_OPERATION_STATUS_RETURN(SetFFNOutpadding(param, opGraph, tensorMap));
            }
            // layer最后的allgather，给下一层allgather，以及lastlayer输出使用
            CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(opGraph, param, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(SetAttnInUnpadding(param, opGraph, tensorMap));
    } else {
        if (param.ffnAllreduce) {
            CHECK_OPERATION_STATUS_RETURN(SetAllReduceNode(param, opGraph, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAddNode(opGraph, param, tensorMap));
    }
    
    return atb::NO_ERROR;
}

atb::Status CalculateDataPartition(MoeDecoderLayerParam &param)
{
    // ATTN
    param.attnDeviceNum = param.mapping.Get(base::ATTN_TP).rankIds.size();
    // FFN
    if (param.isDynamicEp) {
        param.ffnDeviceNum = 1; // 暂不支持MoE DP
    } else {
        param.ffnDeviceNum = param.mapping.Get(base::MOE_EP).rankIds.size() *
        param.mapping.Get(base::MOE_TP).rankIds.size();
    }
    // Lmhead
    ATB_SPEED_LOG_DEBUG("CalculateDataPartition done"
        << ". Attention Stream Num is " << param.attnDeviceNum
        << " . FFN Stream Num is " << param.ffnDeviceNum);
    return atb::NO_ERROR;
}

atb::Status CalculateCommType(MoeDecoderLayerParam &param)
{
    if (param.worldSize == 1) {
        ATB_SPEED_LOG_WARN("Current layerparam.WorldSize is "
            << param.worldSize << ". Please confirm your runtime configuration.");
        return atb::NO_ERROR;
    }
    
    // 2: dynamic ep
    if (!param.mapping.Get(base::ATTN_DP).IsEnabled() && param.expertParallelDegree == 2) {
        // attn纯tp+ep2场景，暂不支持，上层没生成对应kwargs
        ATB_SPEED_LOG_ERROR("Do not support eplevel=2 without attn_dp.");
        return atb::NO_ERROR;
    }

    // 进入attn前的allgather, 位于attn之前, 当前逻辑，放在layer最后
    param.attnAllGather = param.attnDeviceNum > 1 ||
        (param.isLastLayer && param.mapping.Get(base::LM_HEAD_TP).rankIds.size() > 1);
    param.attnReduceScatter = param.attnDeviceNum > 1; // 出去attn前的reducescatter，位于attn之后

    param.ffnAllGather = param.ffnDeviceNum > 1; // 进入moe的allgather，位于moe之前
    param.ffnReduceScatter = param.ffnDeviceNum > 1; // 出去moe的reducescatter，位于moe之后

    if (param.attnDeviceNum == param.ffnDeviceNum) {
        // 特殊场景判断
        // attn部分
        param.attnReduceScatter = false;
        param.ffnAllGather = false;
        if (param.attnDeviceNum > 1) {
            // 纯TP场景
            param.attnAllreduce = true;
        }

        param.ffnReduceScatter = false;
        param.attnAllGather = false;
        if (param.ffnDeviceNum > 1) {
            // 纯TP场景
            param.ffnAllreduce = true;
        } else if (param.isLastLayer && param.mapping.Get(base::LM_HEAD_TP).rankIds.size() > 1) {
            // 最后一层，需要allgather到lmhead数据域
            param.attnAllGather = true;
        }
    }

    param.hasAttnComm = param.attnReduceScatter || param.ffnAllGather; // 该flag只涉及模块后的reduce_scatter+allgather
    param.hasFfnComm = param.ffnReduceScatter || param.attnAllGather; // 该flag只涉及模块后的reduce_scatter+allgather

    ATB_SPEED_LOG_DEBUG("CalculateCommType done"
        << ". attnDeviceNum is " << param.attnDeviceNum
        << ". ffnDeviceNum is " << param.ffnDeviceNum
        << ". attnAllreduce is " << param.attnAllreduce << " . attnReduceScatter is " << param.attnReduceScatter
        << " . attnAllGather is " << param.attnAllGather
        << " . ffnAllreduce is " << param.ffnAllreduce << " . ffnReduceScatter is " << param.ffnReduceScatter
        << " . ffnAllGather is " << param.ffnAllGather);
    return atb::NO_ERROR;
}

atb::Status MoeDecoderLayer(MoeDecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    CalculateDataPartition(param);
    CalculateCommType(param);
    std::map<std::string, uint32_t> tensorMap = QwenMoeConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    std::stringstream ss;
    // 添加layer层 map打印
    for (auto tensor = tensorMap.cbegin(); tensor != tensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("layer map tensor:\n" << ss.str());

    // attention
    CHECK_OPERATION_STATUS_RETURN(SetFusionAttentionNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(SetPostAttnProcess(param, opGraph, tensorMap));

    // moe
    if (param.hasMoe) {
        CHECK_OPERATION_STATUS_RETURN(SetMoeNode(param, opGraph, tensorMap));
    }
    // shareExpert
    if (param.hasSharedExpert) {
        CHECK_OPERATION_STATUS_RETURN(SetShareExpertNode(param, opGraph, tensorMap));
    }
    // shareExperts add moe
    if (param.hasMoe && param.hasSharedExpert) {
        CHECK_OPERATION_STATUS_RETURN(SetShareAddSelectNode(opGraph, tensorMap));
    }

    CHECK_OPERATION_STATUS_RETURN(SetPostMoeProcess(param, opGraph, tensorMap));
    
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_hidden_states"));
        if (param.isLastLayer && param.mapping.Get(base::ATTN_DP).IsEnabled() &&
            param.mapping.Get(base::LM_HEAD_TP).rankIds.size() > 1) {
            outTensorDescs.at(0).shape.dims[0] = \
                inTensorDescs.at(
                    atb_speed::common::GetTensorIdx(tensorMap, "in_lm_head_skip_padding_token_indices")).shape.dims[0];
        }

        if (!param.isDenseLayer && param.enableExpertCumSumOutput) {
            outTensorDescs.at(1) = atb::TensorDesc{};
            outTensorDescs.at(1).format = ACL_FORMAT_ND;
            outTensorDescs.at(1).shape.dimNum = 1;
            outTensorDescs.at(1).dtype = ACL_INT64;
            outTensorDescs.at(1).shape.dims[0] = param.numOfDeviceExperts;
        }
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

MoeDecoderLayer::MoeDecoderLayer() {}
MoeDecoderLayer::~MoeDecoderLayer() {}
}  // namespace qwen
}  // namespace atb_speed