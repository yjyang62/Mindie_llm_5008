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
#include "models/hunyuan/model/decoder_model.h"
#include <vector>
#include <nlohmann/json.hpp>
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace hunyuan {

void HunyuanModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::moe::MoeModelParam::ParseParam(paramJson);
    if (paramJson.contains("claShareFactor")) {
        this->claShareFactor = paramJson["claShareFactor"].get<int>();
    }
    if (paramJson.contains("softmaxScale")) {
        this->softmaxScale = paramJson["softmaxScale"].get<float>();
    }
    for (auto item : paramJson["attnLinearQuantType"]) {
        this->attnLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    if (paramJson.contains("moePackQuantType")) {
        this->moePackQuantType = atb_speed::base::FetchJsonParam<int>(paramJson, "moePackQuantType");
    }
    for (auto item : paramJson["attnLinearTransposeType"]) {
        this->attnLinearTransposeType.push_back(item.get<std::vector<int>>());
    }

    if (this->claShareFactor < 1) {
        throw std::runtime_error("claShareFactor must be greater than or equal to 1, please check.");
    }
}

void HunyuanModelParam::PrintParam()
{
    atb_speed::moe::MoeModelParam::PrintParam();
    std::stringstream ss;
    ss << ", claShareFactor: " << this->claShareFactor <<
        ", softmaxScale: " << this->softmaxScale;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

DecoderModel::DecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    this->param.FromString(param);
    this->weightCountPerLayer = WEIGHT_COUNT_PER_LAYER;
    this->internalTensorCandidates["cross_layer_attn"] = {"k_cross", "v_cross"};
}

void DecoderModel::ConstructInternalTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInternalTensorMap();
    atb_speed::common::AssignTensorIdx(
        this->internalTensorCandidates, "cross_layer_attn", this->internalTensorMap);
}

void DecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    constexpr uint32_t vCacheOffset = WEIGHT_COUNT_PER_LAYER + 5U; // 5U: offset of v_cache after layer weights
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
    layerNode.inTensors.at(vCacheOffset) = &graph_.vCacheTensors.at(
        layerId / this->param.claShareFactor * this->param.claShareFactor);
    if (isCrossLayer_) {
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap, "k_cross"));
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap, "v_cross"));
    }
}

void DecoderModel::SetLayerNodeOutput(atb_speed::Model::Node &layerNode)
{
    uint32_t outTensorId = 0U;
    layerNode.outTensors.resize(layerNode.operation->GetOutputNum());
    layerNode.outTensors.at(outTensorId++) = layerNode.inTensors.at(this->weightCountPerLayer); // 输出原地写在输入上
    if (!isCrossLayer_) {
        layerNode.outTensors.at(outTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap, "k_cross"));
        layerNode.outTensors.at(outTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap, "v_cross"));
    }
}

void DecoderModel::SetLayerParam(DecoderLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::moe::MoeDecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.hasSharedExpert = this->param.hasSharedExpert;
    layerParam.hasSharedExpertGate = this->param.hasSharedExpertGate;
    layerParam.numOfSharedExperts = this->param.numOfSharedExperts;
    layerParam.layerId = layerId;
    layerParam.isCrossLayer = layerId % this->param.claShareFactor != 0;
    layerParam.softmaxScale = this->param.softmaxScale;
    layerParam.attnLinearQuantType = this->param.attnLinearQuantType[layerId];
    layerParam.attnLinearTransposeType = this->param.attnLinearTransposeType[layerId];
    layerParam.moePackQuantType = this->param.moePackQuantType;
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    DecoderLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    DecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    isCrossLayer_ = layerParam.isCrossLayer;
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddLayer()
{
    atb::Operation *op = nullptr;
    for (uint32_t layerId = 0; layerId < this->param.numHiddenLayers; ++layerId) {
        atb_speed::Model::Node layerNode;
        CHECK_OPERATION_STATUS_RETURN(this->CreateLayerOperation(&op, layerId));
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        SetLayerNodeInput(layerNode, layerId);
        SetLayerNodeOutput(layerNode);
        graph_.nodes.push_back(layerNode);
        ATB_SPEED_LOG_DEBUG("layer" << layerId << " create success, type is " << isCrossLayer_);
    }
    ATB_SPEED_LOG_DEBUG("[+] add hunyuan layerNode num" << this->param.numHiddenLayers);
    return atb::NO_ERROR;
}

} // namespace hunyuan
} // namespace atb_speed