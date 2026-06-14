/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "models/baichuan2/13b/model/paged_attention_quant_model.h"

#include <vector>

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "atb_speed/log.h"
#include "models/baichuan2/13b/layer/paged_attention_quant_layer.h"

namespace atb_speed {
namespace baichuan2_13b {
REGISTER_MODEL(baichuan2_13b, PagedAttentionQuantModel);

void BaichuanModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("enableAlibiMaskFree")) {
        this->enableAlibiMaskFree = paramJson["enableAlibiMaskFree"].get<bool>();
    }
}

void BaichuanModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG("BaichuanModelParam: enableAlibiMaskFree: " << this->enableAlibiMaskFree);
}

PagedAttentionQuantModel::PagedAttentionQuantModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["alibi_mask_compress"] = {"in_slopes"};
}

void PagedAttentionQuantModel::ConstructInTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInTensorMap();
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "alibi_mask_compress", this->inTensorMap);
}

atb::Status PagedAttentionQuantModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    BaichuanLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    PagedAttentionQuantLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

void PagedAttentionQuantModel::SetLayerParam(BaichuanLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.enableAlibiMaskFree = this->param.enableAlibiMaskFree;
}

void PagedAttentionQuantModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerNodeInput(layerNode, layerId);
    layerNode.inTensors.at(layerNode.inTensors.size() - 1) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_slopes"));
}

void PagedAttentionQuantModel::SetFinalNormParam(atb::infer::RmsNormParam &normParam)
{
    atb_speed::base::DecoderModel::SetFinalNormParam(normParam);
    normParam.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
}

atb::Status PagedAttentionQuantModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("Baichuan BindParamHostTensor nodeId = " << nodeId);

    if (nodeId != 0) {
        return atb::NO_ERROR;
    }

    uint32_t tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->tokenOffset.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len");
    if (!this->param.isPrefill && this->param.enableCompressHead) {
        tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ra_seqlens");
    }
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->seqLen.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->qLen.data();
    }

    ATB_SPEED_LOG_DEBUG("Baichuan BindParamHostTensor end");
    return atb::NO_ERROR;
}

} // namespace baichuan2_13b
} // namespace atb_speed
