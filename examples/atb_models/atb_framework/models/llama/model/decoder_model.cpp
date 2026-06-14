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
#include "models/llama/model/decoder_model.h"
#include "models/llama/layer/decoder_layer.h"
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include <atb/types.h>

namespace atb_speed {
namespace llama {

const uint64_t RA_LAYER_SEQLEN_IDX = 62;

void LlamaModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("splitWithStride")) {
        this->splitWithStride = atb_speed::base::FetchJsonParam<bool>(paramJson, "splitWithStride");
    }
    if (paramJson.contains("isLongSeq")) { isLongSeq = atb_speed::base::FetchJsonParam<bool>(paramJson, "isLongSeq"); }
}

void LlamaModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG("LlamaModelParam:splitWithStride: " << this->splitWithStride
                  << ", isLongSeq:" << isLongSeq);
}

LlamaDecoderModel::LlamaDecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["default"] = {
        "input_ids", "input_embedding", "positional_ids", "cosine_table", "sine_table", "attention_mask",
        "block_tables", "slots", "kv_cache_idx", "token_offset", "place_holder", "seq_len", "logits_indices"};
    this->inTensorCandidates["long_seq"] = {"pos_ids_expanded", "inv_freq", "pos_lens"};
    this->internalTensorCandidates["long_seq"] = {"cosine_embed_table", "sine_embed_table"};
}

void LlamaDecoderModel::ConstructInTensorMap()
{
    DecoderModel::ConstructInTensorMap();
    // 添加DynamicNTK特性的输入Tensor
    if (this->param.isLongSeq) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "long_seq", this->inTensorMap);
    }
    // 添加后处理前置的Tensor
    if (this->param.enableGreedyPostProcessing) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "token_off_set", this->inTensorMap);
    }
}

void LlamaDecoderModel::ConstructInternalTensorMap()
{
    DecoderModel::ConstructInternalTensorMap();
    // 添加DynamicNTK特性的中间Tensor
    if (this->param.isLongSeq) {
        atb_speed::common::AssignTensorIdx(this->internalTensorCandidates, "long_seq", this->internalTensorMap);
    }
}

atb::Status LlamaDecoderModel::AddNodesBeforeLayer()
{
    if (!this->param.skipWordEmbedding) { CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding()); }
    if (this->param.isLongSeq) { CHECK_OPERATION_STATUS_RETURN(this->AddDynamicNTK()); }
    if (this->param.positionEmbeddingType == atb_speed::base::PositionEmbeddingType::ROPE) {
        CHECK_OPERATION_STATUS_RETURN(this->AddPositionalEmbedding());
    }
    if (!this->param.firstPpRank) {
        CHECK_OPERATION_STATUS_RETURN(AddRecv());
    }
    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::AddNodesAfterLayer()
{
    if (this->param.lastPpRank) {
        CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
        CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    } else {
        CHECK_OPERATION_STATUS_RETURN(AddSend());
    }
    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::AddDynamicNTK()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node dynamicNTKNode;
    atb::infer::DynamicNTKParam dynamicNTKParam;
    dynamicNTKParam.outDataType = this->param.isBF16 ? ACL_BF16 : ACL_FLOAT16;

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(dynamicNTKParam, &op));
    dynamicNTKNode.operation.reset(op);

    dynamicNTKNode.inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "pos_ids_expanded")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "inv_freq")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "pos_lens"))
    };
    dynamicNTKNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embed_table")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embed_table"))
    };
    ATB_SPEED_LOG_DEBUG("[+] dynamicNTKNode");
    graph_.nodes.push_back(dynamicNTKNode);
    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::AddPositionalEmbedding()
{
    CHECK_OPERATION_STATUS_RETURN(atb_speed::base::DecoderModel::AddPositionalEmbedding());
    if (this->param.isLongSeq) {
        graph_.nodes.at(graph_.nodes.size() - 1).inTensors = {
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids")),
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embed_table")),
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embed_table"))
        };
    }

    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    LlamaLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    LlamaDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

void LlamaDecoderModel::SetLayerParam(LlamaLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.splitWithStride = this->param.splitWithStride;
}

atb::Status LlamaDecoderModel::ParseParam(const std::string &paramString)
{
    CHECK_OPERATION_STATUS_RETURN(atb_speed::base::DecoderModel::ParseParam(paramString));
    nlohmann::json paramJson = atb_speed::base::StringToJson(paramString);

    this->blockNumsList_.clear();
    for (auto item : paramJson["blockNumsList"]) {
        this->blockNumsList_.push_back(atb_speed::base::FetchJsonParam<int>(item, "blockNumsList", true));
        ATB_SPEED_LOG_DEBUG("blockNumsList value: " << item);
    }

    return atb::NO_ERROR;
}

atb::Status LlamaDecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

    if (param.enableDap) {
        BindDapHostTensor(this->seqLenForDap, "seq_len");
        BindDapHostTensor(this->tokenOffsetForDap, "token_offset");
        BindDapHostTensor(this->qLenForDap, "q_len");
        return atb::NO_ERROR;
    }

    uint32_t tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = tokenOffset.data();
    }
    auto &node = graph_.nodes.at(nodeId);
    if (tensorIdx != UINT32_MAX) {
        if (!this->param.isPrefill && this->param.enableCompressHead) {
            // OPERATION_COUNT_BEFORE_LAYER_SKIP_EMBED = 1, OPERATION_COUNT_BEFORE_LAYER_SKIP_EMBED = 2
            int operationCountBeforeLayers = this->param.skipWordEmbedding ? 1 : 2;
            auto upperBound = operationCountBeforeLayers;
            auto lowerBound = upperBound + this->param.numHiddenLayers;
            if (nodeId < static_cast<uint32_t>(upperBound) || nodeId >= static_cast<uint32_t>(lowerBound)) {
                return atb::NO_ERROR;
            }
            auto layerNum = this->param.numHiddenLayers;
            auto layerId = nodeId - upperBound;
            tensorIdx = RA_LAYER_SEQLEN_IDX;
            node.variantPack.inTensors.at(tensorIdx).hostData = seqLen.data() + seqLen.size() / layerNum * layerId;
        } else {
            tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len");
            graph_.inTensors.at(tensorIdx).hostData = seqLen.data();
        }
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = qLen.data();
    }

    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}

} // namespace llama
} // namespace atb_speed