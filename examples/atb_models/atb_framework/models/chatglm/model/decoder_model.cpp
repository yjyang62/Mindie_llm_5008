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
#include "models/chatglm/model/decoder_model.h"
#include "models/chatglm/layer/decoder_layer.h"
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include <atb/types.h>

namespace atb_speed {
namespace chatglm {

void SliceTensor(atb::Tensor &tensor, int layerNum, int layerId)
{
    uint64_t offset = tensor.dataSize / layerNum;

    auto p = static_cast<char*>(tensor.deviceData);
    p += offset * layerId;
    tensor.deviceData = static_cast<void*>(p);
    tensor.dataSize = offset;
    tensor.desc.shape.dims[0] = tensor.desc.shape.dims[0] / layerNum;
}

// Weight count
const uint64_t BLOCK_TABLES_LAYER_ID = 59;
const uint64_t SLOTS_LAYER_ID = 60;
const uint64_t SEQ_LEN_LAYER_ID = 62;
const uint64_t PFFSET_INDEX_LAYER_ID = 63;
const uint64_t RAZOR_OFFSET_LAYER_ID = 64;

// Weight count
const uint64_t WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const uint64_t WEIGHT_COUNT_POST_NORM = 1;
const uint64_t WEIGHT_COUNT_LM_HEAD = 1;

// Operation count
const uint64_t OPERATION_COUNT_BEFORE_LAYER = 2;
const uint64_t OPERATION_COUNT_BEFORE_LAYER_SKIP_EMBED = 1;
const uint64_t OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead


ChatglmDecoderModel::ChatglmDecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
}

atb::Status ChatglmDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    ChatglmLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    ChatglmDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

atb::Status ChatglmDecoderModel::ParseParam(const std::string &paramString)
{
    CHECK_OPERATION_STATUS_RETURN(atb_speed::base::DecoderModel::ParseParam(paramString));
    nlohmann::json paramJson = atb_speed::base::StringToJson(paramString);

    this->blockNumsList_.clear();
    for (auto item : paramJson["blockNumsList"]) {
        this->blockNumsList_.push_back(item.get<int>());
        ATB_SPEED_LOG_DEBUG("blockNumsList value: " << item);
    }

    return atb::NO_ERROR;
}

atb::Status ChatglmDecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

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
            tensorIdx = SEQ_LEN_LAYER_ID;
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

void ChatglmDecoderModel::BuildNodeOutTensors(
    int nodeId, atb_speed::Model::Node &node, atb::SVector<atb::TensorDesc>& inTensorDescs)
{
    BuildNodeOutTensorImpl(nodeId, node, inTensorDescs);
}


} // namespace chatglm
} // namespace atb_speed