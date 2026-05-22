/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "models/glm41v/layer/decoder_layer.h"

namespace atb_speed {
namespace glm41v {

static const uint64_t NUM2 = 2;

DecoderLayer::DecoderLayer(
    const atb_speed::base::LayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->inTensorCandidates["post_self_attn_norm_weight"] = {
        "in_post_self_attn_norm_weight"
    };
    this->inTensorCandidates["post_mlp_norm_weight"] = {
        "in_post_mlp_norm_weight"
    };
};

void DecoderLayer::ConstructInTensorMap()
{
    this->inTensorList.clear();
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_self_attn_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_attn_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "mlp_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_mlp_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default", this->inTensorList);
    if (param.enableSpeculate) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "q_len", this->inTensorList);
    }
    if (param.enableFlashComm) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "flash_comm", this->inTensorList);
    }
}

void DecoderLayer::SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionParam(fusionAttentionParam);
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::HALF_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = this->param.hiddenSizePerAttentionHead / NUM2;
}

atb::Status DecoderLayer::AddPostSelfAttentionRMSNorm()
{
    atb::infer::RmsNormParam normParam;
    normParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    normParam.normParam.epsilon = this->param.normEps;
    atb::Node postSelfAttentionNormNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(normParam, &postSelfAttentionNormNode.operation));
    postSelfAttentionNormNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap,
                                            {"intermediate_attn_out", "in_post_self_attn_norm_weight"});
    postSelfAttentionNormNode.outTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_attn_out"});
    this->graph.nodes.push_back(postSelfAttentionNormNode);
    return atb::NO_ERROR;
}

atb::Status DecoderLayer::AddPostMlpRMSNorm()
{
    atb::infer::RmsNormParam normParam;
    normParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    normParam.normParam.epsilon = this->param.normEps;
    atb::Node postMlpNormNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(normParam, &postMlpNormNode.operation));
    postMlpNormNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap,
                                            {"intermediate_mlp_out", "in_post_mlp_norm_weight"});
    postMlpNormNode.outTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    this->graph.nodes.push_back(postMlpNormNode);
    return atb::NO_ERROR;
}

atb::Status DecoderLayer::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttention());
    CHECK_OPERATION_STATUS_RETURN(this->AddPostSelfAttentionRMSNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttentionResidualAdd());
    CHECK_OPERATION_STATUS_RETURN(this->AddMlp());
    CHECK_OPERATION_STATUS_RETURN(this->AddPostMlpRMSNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddMlpResidualAdd());
    return atb::NO_ERROR;
}


} // namespace glm41v
} // namespace atb_speed