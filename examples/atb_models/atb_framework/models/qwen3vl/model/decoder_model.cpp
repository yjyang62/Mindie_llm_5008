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

#include "models/qwen3vl/model/decoder_model.h"
#include "operations/fusion/infer_shape_functions.h"

namespace atb_speed {
namespace qwen3vl {

const int DEEP_STACK_LAYER_NUM = 3;

DecoderModel::DecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->inTensorCandidates["deepstack_visual_embeds"] = {
        "deepstack_visual_embeds_0", "deepstack_visual_embeds_1", "deepstack_visual_embeds_2"};
}

void DecoderModel::ConstructInTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInTensorMap();
    if (this->param.isPrefill) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "deepstack_visual_embeds", this->inTensorMap);
    }
}

void DecoderModel::SetLayerNodeDefaultInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    for (uint32_t weightTensorId = 0; weightTensorId < this->weightCountPerLayer; ++weightTensorId) {
        layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
            CheckIntMulOverFlow(layerId, this->weightCountPerLayer) + weightTensorId + this->weightCountWordEmbedding);
    }
    if (this->param.enableFlashComm && layerId >= DEEP_STACK_LAYER_NUM) {
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap,
            "hidden_states_rank" + std::to_string(this->param.rank)));
    } else {
        layerNode.inTensors.at(inTensorId++) = this->param.skipWordEmbedding ? \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"));
    }
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embedding"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embedding"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "attention_mask"));
    layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "kv_cache_idx"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "block_tables"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "slots"));
    ATB_SPEED_LOG_DEBUG("LayerNode Default Input set success, inputs num: " << inTensorId);
}

void DecoderModel::SetLayerNodeOptionalInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    if (this->param.enableSpeculate) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len"));
    }
    if (this->param.enableFlashComm && layerId >= DEEP_STACK_LAYER_NUM) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "send_counts"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "sdispls"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "send_count"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "recv_counts"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "rdispls"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "recv_count"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "fake_rs_shape"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "fake_ag_shape"));
    }
    if (this->param.isPrefill) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_0"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_1"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "deepstack_visual_embeds_2"));
    }
    ATB_SPEED_LOG_DEBUG("LayerNode Optional Input set success, inputs num: " << inTensorId);
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::base::LayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    DecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

void DecoderModel::SetLayerParam(atb_speed::base::LayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    if (layerId < DEEP_STACK_LAYER_NUM) {
        layerParam.enableFlashComm = false;
    }
}

atb::Status DecoderModel::AddNodesBeforeLayer()
{
    if (!this->param.skipWordEmbedding) { CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding()); }
    CHECK_OPERATION_STATUS_RETURN(this->AddPositionalEmbedding());
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddSingleLayer(uint32_t layerId)
{
    if (this->param.enableFlashComm && layerId == DEEP_STACK_LAYER_NUM) {
        CHECK_OPERATION_STATUS_RETURN(this->AddSplitHiddenStates());
    }
    return atb_speed::base::DecoderModel::AddSingleLayer(layerId);
}

} // namespace qwen3vl
} // namespace atb_speed