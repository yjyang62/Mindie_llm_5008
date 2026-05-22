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
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/hunyuan/operation/cross_layer_attention.h"
#include "models/hunyuan/layer/decoder_layer.h"

namespace atb_speed {
namespace hunyuan {

void DecoderLayerParam::PrintParam()
{
    atb_speed::moe::MoeLayerParam::PrintParam();
    std::stringstream ss;
    ss << "Huanyuan Layer Param: " << "isCrossLayer: " << this->isCrossLayer
       << ", softmaxScale: " << this->softmaxScale
       << ", hasSharedExpert: " << this->hasSharedExpert
       << ", hasSharedExpertGate: " << this->hasSharedExpertGate
       << ", moePackQuantType" << this->moePackQuantType;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

DecoderLayer::DecoderLayer(const DecoderLayerParam &param)
    : BaseMoeLayer(static_cast<atb_speed::moe::MoeLayerParam>(param))
{
    this->param = param;
    this->param.CheckParam();
    this->inTensorCandidates["cla_weight"] = {
        "in_qkv_proj_weight", "in_qkv_proj_bias", "in_qkv_proj_descale", "in_qkv_proj_offset", "in_qkv_proj_scale",
        "in_qkv_proj_compress_idx",
        "q_norm_weight", "q_norm_bias", "q_norm_new_weight", "q_norm_new_weight_bias",
        "k_norm_weight", "k_norm_bias", "k_norm_new_weight", "k_norm_new_weight_bias",

        "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale", "in_attention_out_offset",
        "in_attention_out_scale", "in_attention_out_compress_idx"
    };
    this->inTensorCandidates["cross"] = {"k_cross", "v_cross"};

    this->inTensorCandidates["shared_expert"] = {
        "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",

        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
        "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
        "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",

        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias",
        "in_shared_expert_gate_descale", "in_shared_expert_gate_offset",
        "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    };
    this->internalTensorCandidates["shared_expert"] = {"intermediate_shared_expert_out"};
}

void DecoderLayer::ConstructInTensorMap()
{
    this->inTensorList.clear();
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_norm_weight", this->inTensorList);     // base
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "cla_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_attn_norm_weight", this->inTensorList); // base
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "shared_expert", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "moe_weight", this->inTensorList);        // base moe
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default", this->inTensorList);           // base
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default_moe", this->inTensorList);       // base moe
    if (this->param.isCrossLayer) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "cross", this->inTensorList);
    }

    this->graph.inTensorNum = this->inTensorList.size();
}

void DecoderLayer::ConstructInternalTensorMap()
{
    atb_speed::base::DecoderLayer<NormType>::ConstructInternalTensorMap();
    atb_speed::common::AddTensorToList(
        this->internalTensorCandidates, "default_moe", this->intermediateTensorList);   // base moe
    atb_speed::common::AddTensorToList(
        this->internalTensorCandidates, "shared_expert", this->intermediateTensorList);
    this->graph.internalTensorNum = this->intermediateTensorList.size();
}

void DecoderLayer::ConstructOutTensorMap()
{
    this->outTensorList = {"out"};
    if (!param.isCrossLayer) {
        this->outTensorList.push_back("k_cross");
        this->outTensorList.push_back("v_cross");
    }
    this->graph.outTensorNum = this->outTensorList.size();
}


static atb::Status SetCrossLayerAttentionParam(
    atb_speed::common::CrossLayerAttentionParam<atb::infer::RmsNormParam> &claParam,
    const DecoderLayerParam &param)
{
    claParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    claParam.isCrossLayer = param.isCrossLayer;
    claParam.isBF16 = param.isBF16;
    claParam.attnLinearQuantType = param.attnLinearQuantType;
    claParam.packQuantType = param.packQuantType.at(0);
    claParam.attnLinearTransposeType = param.attnLinearTransposeType;
    claParam.supportLcoc = param.enableLcoc;

    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.normEps;
    claParam.normParamType = attenRmsNormParam;

    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.normEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    claParam.normQuantParamType = attenRmsNormQuantParam;
   
    claParam.headNum = param.numAttentionHeadsPerRank;
    claParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    claParam.ropeParam.rotaryCoeff = 2; // 2:设置新张量形状
    claParam.isPrefill = param.isPrefill;
    claParam.headDim = param.hiddenSizePerAttentionHead;
    claParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    claParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;

    CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
    claParam.selfAttentionParam.qkScale = param.softmaxScale;
    claParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    claParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    claParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    claParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    claParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    claParam.pageAttentionParam.mlaVHeadSize = 0;
    claParam.pageAttentionParam.qkScale = param.softmaxScale;
    claParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
    claParam.selfOutLinearTensorParallelInfo = param.tensorParallelInfo;
    claParam.reshapeCacheParm.kvCacheCfg = param.isCrossLayer
        ? atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS
        : atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE;
    return atb::NO_ERROR;
}

atb::Status DecoderLayer::AddCrossLayerAttention()
{
    ATB_SPEED_LOG_DEBUG("AddCrossLayerAttention start");
    atb::Node attentionNode;
    atb_speed::common::CrossLayerAttentionParam<atb::infer::RmsNormParam> claParam;
    SetCrossLayerAttentionParam(claParam, this->param);
    CHECK_OPERATION_STATUS_RETURN(CrossLayerAttention(claParam, &attentionNode.operation));
    std::vector<std::string> attnInTensorNames = {
        "in_hidden_states",
        "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",

        "q_norm_weight", "q_norm_bias", "q_norm_new_weight", "q_norm_new_weight_bias",
        "k_norm_weight", "k_norm_bias", "k_norm_new_weight", "k_norm_new_weight_bias",
        "in_qkv_proj_weight", "in_qkv_proj_bias", "in_qkv_proj_descale", "in_qkv_proj_offset", "in_qkv_proj_scale",
        "in_qkv_proj_compress_idx",
        "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale",
        "in_attention_out_offset",  "in_attention_out_scale", "in_attention_out_compress_idx",

        "in_cos_embedding", "in_sin_embedding", "in_seq_len", "in_k_cache", "in_v_cache",
        "in_attention_mask", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"
    };
    std::vector<std::string> attnOutTensorNames = {"intermediate_attn_out"};

    if (this->param.isCrossLayer) {
        attnInTensorNames.push_back("k_cross");
        attnInTensorNames.push_back("v_cross");
    } else {
        attnOutTensorNames.push_back("k_cross");
        attnOutTensorNames.push_back("v_cross");
    }
    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnOutTensorNames);
    this->graph.nodes.push_back(attentionNode);
    ATB_SPEED_LOG_DEBUG("AddCrossLayerAttention success");
    return atb::NO_ERROR;
}

void DecoderLayer::SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam)
{
    MoeDecoderLayer<atb::infer::RmsNormParam>::SetSparseMoeParam(sparseMoeParam);
    // load weight transpose to operator transpose
    auto &moeTransFlags = this->param.moeLinearTransposeType;
    sparseMoeParam.packQuantType = this->param.moePackQuantType;
    sparseMoeParam.gateUpTransposeB = moeTransFlags[atb_speed::common::SparseMoeIdx::MOE_MLP_GATE_IDX] ==
                                        atb_speed::common::TransposeType::TRANSPOSE;
    sparseMoeParam.downTransposeB = moeTransFlags[atb_speed::common::SparseMoeIdx::MOE_MLP_DOWN_IDX] ==
                                        atb_speed::common::TransposeType::TRANSPOSE;
}

atb::Status DecoderLayer::AddSharedExpert()
{
    ATB_SPEED_LOG_DEBUG("AddSharedExpert start");
    atb::Node sharedExpertNode;
    atb_speed::common::SharedExpertParam sharedExpertParam;
    sharedExpertParam.isBF16 = this->param.isBF16;
    sharedExpertParam.transposeGateup = this->param.transpose;
    sharedExpertParam.transposeDown = this->param.transpose;
    sharedExpertParam.hasSharedExpertGate = this->param.hasSharedExpertGate;
    sharedExpertParam.mlpLinearQuantType = this->param.mlpLinearQuantType;
    sharedExpertParam.mlpLinearTransposeType = this->param.mlpLinearTransposeType;
    sharedExpertParam.packQuantType = this->param.packQuantType.at(1);
    atb_speed::common::CreateSharedExpertOperation(sharedExpertParam, &sharedExpertNode.operation);
    std::vector<std::string> sharedExpertInTensorNames = {
        "norm_out",
        "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
        "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
        "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    };
    sharedExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertInTensorNames);
    sharedExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_expert_out"});
    this->graph.nodes.push_back(sharedExpertNode);
    ATB_SPEED_LOG_DEBUG("Shared expert calculation success");
    return atb::NO_ERROR;
}

atb::Status DecoderLayer::AddMoeAllReduce()
{
    atb::Node moeAllReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = this->param.tensorParallelInfo.rank;
    allReduceParam.rankSize = this->param.tensorParallelInfo.worldSize;
    allReduceParam.backend = this->param.tensorParallelInfo.backend;
    allReduceParam.rankTableFile = this->param.tensorParallelInfo.rankTableFile;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allReduceParam, &moeAllReduceNode.operation));
    moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out"});
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    this->graph.nodes.push_back(moeAllReduceNode);
    ATB_SPEED_LOG_DEBUG("hunyuan create all reduce");
    return atb::NO_ERROR;
}

atb::Status DecoderLayer::AddExpertAdd()
{
    atb::Node expertAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &expertAddNode.operation));
    if (param.tensorParallelInfo.worldSize > 1) {
        expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out"});
        expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
            this->tensorMap, {"moe_out", "intermediate_shared_expert_out"});
    } else {
        expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
            this->tensorMap, {"intermediate_mlp_out", "intermediate_shared_expert_out"});
        expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    }
    this->graph.nodes.push_back(expertAddNode);
    ATB_SPEED_LOG_DEBUG("create add operation success");
    return atb::NO_ERROR;
}

atb::Status DecoderLayer::BuildGraph(atb::Operation **operation)
{
    auto &layerparam = this->param;
    this->graph.name = layerparam.isPrefill ? (!layerparam.isCrossLayer ? "Prefill_Selflayer" : "Prefill_Crosslayer") :
                    (!layerparam.isCrossLayer ? "Decoder_Selflayer" : "Decoder_Crosslayer");
    this->ConstructInTensorMap();
    this->ConstructInternalTensorMap();
    this->ConstructOutTensorMap();
    this->tensorMap = atb_speed::common::GetTensorMap(
        this->inTensorList, this->outTensorList, this->intermediateTensorList);

    ATB_SPEED_LOG_DEBUG(this->graph.name);
    ATB_SPEED_LOG_DEBUG("layer graph inTensorNum: " << this->graph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph outTensorNum: "<< this->graph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph internalTensorNum: "<< this->graph.internalTensorNum);

    CHECK_OPERATION_STATUS_RETURN(this->AddCrossLayerAttention());
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttentionResidualAdd());
    CHECK_OPERATION_STATUS_RETURN(this->AddSelfNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddSharedExpert());
    CHECK_OPERATION_STATUS_RETURN(this->AddMoe());
    CHECK_OPERATION_STATUS_RETURN(this->AddExpertAdd());
    if (layerparam.tensorParallelInfo.worldSize > 1) {
        CHECK_OPERATION_STATUS_RETURN(this->AddMoeAllReduce());
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddMlpResidualAdd());

    this->graph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(WEIGHT_COUNT_PER_LAYER); // in hidden states
        if (!layerparam.isCrossLayer) {
            outTensorDescs.at(1) = outTensorDescs.at(0);
            outTensorDescs.at(1).shape.dims[1] =
             layerparam.numKeyValueHeadsPerRank * layerparam.hiddenSizePerAttentionHead;
            outTensorDescs.at(2) = outTensorDescs.at(1); // 2: out tensor index
        }
        return atb::NO_ERROR;
    };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(this->graph, operation));
    return atb::NO_ERROR;
}

} // namespace hunyuan
} // namespace atb_speed
