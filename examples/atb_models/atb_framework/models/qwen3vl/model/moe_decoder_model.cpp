/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "models/qwen3vl/model/moe_decoder_model.h"

namespace atb_speed {
namespace qwen3vl {

MoeDecoderModel::MoeDecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    this->param.skipWordEmbedding = atb_speed::base::DecoderModel::param.skipWordEmbedding;
    this->inTensorCandidates["deepstack_visual_embeds"] = {
        "deepstack_visual_embeds_0", "deepstack_visual_embeds_1", "deepstack_visual_embeds_2"};
}

void MoeDecoderModel::ConstructInTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInTensorMap();
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default_moe", this->inTensorMap);
    if (this->param.isPrefill) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "deepstack_visual_embeds", this->inTensorMap);
    }
}

void MoeDecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    uint32_t inTensorId = 0;
    this->SetLayerNodeDefaultInput(layerNode, layerId, inTensorId);
    this->SetLayerNodeOptionalInput(layerNode, layerId, inTensorId);
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_array_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_group_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "one_hot_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "zero_hot_model"));
    if (this->param.isPrefill) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_0"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_1"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_2"));
    }
}

atb::Status MoeDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::moe::MoeLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    atb_speed::qwen3vl::MoeDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

} // namespace qwen3vl
} // namespace atb_speed