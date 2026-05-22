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
#include "models/ernie_moe/layer/decoder_layer.h"
#include "models/ernie_moe/model/decoder_model.h"

namespace atb_speed {
namespace ernie_moe {

DecoderModel::DecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    this->param.FromString(param);
    if (this->param.hasSharedExpert) {
        this->weightCountMoeLayer = uint32_t(68); // 68: MoE layer with shared expert weight number
    } else {
        this->weightCountMoeLayer = uint32_t(50); // 50: MoE layer weight number
    }
}

uint32_t DecoderModel::CalcWeightTensorSize()
{
    const uint64_t weightTensorSize =
        this->weightCountWordEmbedding
        + CheckIntMulOverFlow(this->weightCountDenseLayer, uint32_t(this->param.firstKDenseReplace))
        + CheckIntMulOverFlow(
            this->weightCountMoeLayer,
            uint32_t(CheckPositive(this->param.numHiddenLayers - uint32_t(this->param.firstKDenseReplace))))
        + this->weightCountFinalNorm + this->weightCountLmHead;
    return weightTensorSize;
}

void DecoderModel::SetLayerNodeDefaultInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    const uint32_t ERNIE_WEIGHT_COUNT_BEFORE_MOE_LAYER = this->weightCountWordEmbedding
        + CheckIntMulOverFlow(uint32_t(this->param.firstKDenseReplace), this->weightCountDenseLayer);
    if (layerId < uint32_t(this->param.firstKDenseReplace)) {
        for (uint32_t weightTensorId = 0; weightTensorId < this->weightCountDenseLayer; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                weightTensorId + this->weightCountWordEmbedding
                + CheckIntMulOverFlow(layerId, this->weightCountDenseLayer));
        }
        this->weightCountPerLayer = this->weightCountDenseLayer;
    } else {
        for (uint32_t weightTensorId = 0; weightTensorId < this->weightCountMoeLayer; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                weightTensorId + ERNIE_WEIGHT_COUNT_BEFORE_MOE_LAYER
                + CheckIntMulOverFlow(uint32_t(layerId - this->param.firstKDenseReplace), this->weightCountMoeLayer));
        }
        this->weightCountPerLayer = this->weightCountMoeLayer;
    }

    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embedding"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embedding"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "attention_mask"));
    layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "kv_cache_idx"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "block_tables"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "slots"));
    if (inTensorId > layerNode.inTensors.size()) {
        std::stringstream ss;
        ss << "Layer inTensorId = " << inTensorId << " should not be greater than the size of layerNode inTensors "
           << "=" << layerNode.inTensors.size() << "." << std::endl;
        throw std::runtime_error(ss.str());
    }
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::moe::MoeLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    MoeDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

} // namespace ernie_moe
} // namespace atb_speed