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
#include "models/qwen/model/decoder_model.h"
#include "models/qwen/layer/decoder_layer.h"
#include "operations/fusion/infer_shape_functions.h"

namespace atb_speed {
namespace qwen {

const uint64_t RA_LAYER_SEQLEN_IDX = 62;

void QwenModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("withEmbedding")) {
        this->withEmbedding = paramJson["withEmbedding"].get<bool>();
    }
    if (paramJson.contains("enableLogN")) {
        this->enableLogN = paramJson["enableLogN"].get<bool>();
    }
    if (paramJson.contains("isLongSeq")) {
        this->isLongSeq = paramJson["isLongSeq"].get<bool>();
    }
    if (paramJson.contains("isYarn")) {
        isYarn = paramJson["isYarn"].get<bool>();
    }
    if (paramJson.contains("mscale")) {
        this->mscale = paramJson["mscale"].get<float>();
    }
    if (paramJson.contains("enableQScale")) {
        this->enableQScale = paramJson["enableQScale"].get<bool>();
    }
    if (paramJson.contains("enableRopeQuantKvcache")) {
        this->enableRopeQuantKvcache = paramJson["enableRopeQuantKvcache"].get<bool>();
    }
}

void QwenModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG("QwenModelParam: withEmbedding: " << this->withEmbedding << ", enableLogN: " << this->enableLogN
                                                          << ", isLongSeq: " << this->isLongSeq
                                                          << ", isYarn:" << this->isYarn << ", mscale:" << this->mscale
                                                          << ", enableQScale: " << this->enableQScale
                                                          << ", enableFlashComm:" << this->enableFlashComm);
}

QwenDecoderModel::QwenDecoderModel(const std::string &param) : DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["long_seq"] = {"inv_freq", "pos_lens", "positional_ids_gather"};
    this->internalTensorCandidates["long_seq"] = {"cosine_embed_table", "sine_embed_table"};
}

bool QwenDecoderModel::ReuseEmbedTable()
{
    return atb_speed::base::DecoderModel::param.layerwiseDisaggregated &&
        atb_speed::base::DecoderModel::param.reuseEmbedTable;
}

bool QwenDecoderModel::OutputEmbedTable()
{
    return atb_speed::base::DecoderModel::param.layerwiseDisaggregated &&
        atb_speed::base::DecoderModel::param.outputEmbedTable;
}

void QwenDecoderModel::ConstructInTensorMap()
{
    this->inTensorMap.clear();
    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default", this->inTensorMap);
    
    // 添加后处理前置的Tensor
    if (this->param.enableGreedyPostProcessing) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "token_off_set", this->inTensorMap);
    }
    // 添加长序列所需Tensor
    if (this->param.isLongSeq) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "long_seq", this->inTensorMap);
    }

    // 添加并行解码特性或SplitFuse的Tensor
    if (this->param.enableSpeculate || this->param.enableSplitFuse) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "q_len", this->inTensorMap);
    }

    // 添加lora特性的Tensor
    if (this->param.enableLora) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "lora_common", this->inTensorMap);
        uint32_t currentTensorIdx = this->inTensorMap.size();
        for (uint32_t i = 0; i < this->param.numHiddenLayers; i++) {
            for (std::string loraWeightName : this->inTensorCandidates.at("lora_per_layer")) {
                this->inTensorMap["layer_" + std::to_string(i) + loraWeightName] = currentTensorIdx;
                currentTensorIdx++;
            }
        }
    }
    
    // 添加omniattention特性的Tensor
    if (this->param.enableOmniAttention) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "compress_head_rope_common", this->inTensorMap);
        uint32_t currentTensorIdx = this->inTensorMap.size();
        for (uint32_t i = 0; i < this->param.numHiddenLayers; ++i) {
            for (std::string raInputName : this->inTensorCandidates.at("compress_head_rope_per_layer")) {
                this->inTensorMap["layer_" + std::to_string(i) + "_" + raInputName] = currentTensorIdx;
                currentTensorIdx++;
            }
        }
    }

    // Add flashcomm intensor
    if (this->param.enableFlashComm) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "flash_comm", this->inTensorMap);
    }

    // 添加边云协同复用embed table的Tensor
    if (this->ReuseEmbedTable()) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "reuse_embed_table", this->inTensorMap);
    }
}

void QwenDecoderModel::ConstructInternalTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInternalTensorMap();
    // 添加长序列的中间Tensor
    // 边云场景下，如果开启ReuseEmbedTable或OutputEmbedTable，原本的中间变量会在input或output中存在，不需要添加中间Tensor
    if (this->param.isLongSeq && !this->ReuseEmbedTable() && !this->OutputEmbedTable()) {
        atb_speed::common::AssignTensorIdx(this->internalTensorCandidates, "long_seq", this->internalTensorMap);
    }
}

void QwenDecoderModel::ConstructOutTensorMap()
{
    atb_speed::base::DecoderModel::ConstructOutTensorMap();
    if (this->OutputEmbedTable()) {
        atb_speed::common::AssignTensorIdx(this->outTensorCandidates, "output_embed_table", this->outTensorMap);
    }
}

atb::Tensor *QwenDecoderModel::GetSineEmbedTable()
{
    if (this->ReuseEmbedTable()) {
        return &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "sine_embed_table"));
    }
    if (this->OutputEmbedTable()) {
        return &graph_.outTensors.at(atb_speed::common::GetTensorIdx(this->outTensorMap, "sine_embed_table"));
    }
    return &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embed_table"));
}

atb::Tensor *QwenDecoderModel::GetCosineEmbedTable()
{
    if (this->ReuseEmbedTable()) {
        return &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "cosine_embed_table"));
    }
    if (this->OutputEmbedTable()) {
        return &graph_.outTensors.at(atb_speed::common::GetTensorIdx(this->outTensorMap, "cosine_embed_table"));
    }
    return &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embed_table"));
}

atb::Status QwenDecoderModel::AddDynamicNTK()
{
    atb::Operation *op = nullptr;
    if (param.isLongSeq) {
        if (this->ReuseEmbedTable()) {
            ATB_SPEED_LOG_DEBUG("reuse embed table, skip add dynamicNKNode");
            return atb::NO_ERROR;
        }
        atb_speed::Model::Node dynamicNTKNode;
        atb::infer::DynamicNTKParam dynamicNTKParam;
        dynamicNTKParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;

        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(dynamicNTKParam, &op));
        dynamicNTKNode.operation.reset(op);

        dynamicNTKNode.inTensors = {
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids")),
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "inv_freq")),
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "pos_lens"))};
        dynamicNTKNode.outTensors = { this->GetSineEmbedTable(), this->GetCosineEmbedTable() };
        ATB_SPEED_LOG_DEBUG("[+] dynamicNTKNode");
        graph_.nodes.push_back(dynamicNTKNode);
    }
    return atb::NO_ERROR;
}

atb::Status QwenDecoderModel::AddMuls()
{
    atb::Operation *op = nullptr;

    if (param.isLongSeq && param.isYarn) {
        if (this->ReuseEmbedTable()) {
            ATB_SPEED_LOG_DEBUG("reuse embed table, skip add muls");
            return atb::NO_ERROR;
        }
        atb_speed::Model::Node mulsCosNode;
        atb::infer::ElewiseParam magnifyElewiseMulsParam;
        magnifyElewiseMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
        magnifyElewiseMulsParam.mulsParam.varAttr = param.mscale;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(magnifyElewiseMulsParam, &op));
        mulsCosNode.operation.reset(op);
        mulsCosNode.inTensors = { this->GetCosineEmbedTable() };
        mulsCosNode.outTensors = { this->GetCosineEmbedTable() };
        graph_.nodes.push_back(mulsCosNode);

        atb_speed::Model::Node mulsSinNode;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(magnifyElewiseMulsParam, &op));
        mulsSinNode.operation.reset(op);
        mulsSinNode.inTensors = { this->GetSineEmbedTable() };
        mulsSinNode.outTensors = { this->GetSineEmbedTable() };
        graph_.nodes.push_back(mulsSinNode);
    }

    return atb::NO_ERROR;
}

atb::Status QwenDecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node positionalEmbeddingGatherNode;
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    positionalEmbeddingGatherNode.operation.reset(op);
    if (param.isLongSeq) {
        positionalEmbeddingGatherNode.inTensors = {
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids_gather")),
            this->GetCosineEmbedTable(), this->GetSineEmbedTable(),
        };
    } else {
        positionalEmbeddingGatherNode.inTensors = {
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids")),
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "cosine_table")),
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "sine_table")),
        };
    }

    positionalEmbeddingGatherNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embedding")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embedding"))};
    ATB_SPEED_LOG_DEBUG("[+] positionalEmbeddingGatherNode");
    graph_.nodes.push_back(positionalEmbeddingGatherNode);

    return atb::NO_ERROR;
}

void QwenDecoderModel::SetLayerParam(QwenLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.enableLogN = param.enableLogN;
    layerParam.enableQScale = param.enableQScale;
    layerParam.enableRopeQuantKvcache = param.enableRopeQuantKvcache;
}

void QwenDecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    DecoderModel::SetLayerNodeInput(layerNode, layerId);

    uint32_t inTensorId = layerNode.inTensors.size() - 1;
    if (param.enableLogN) {
        layerNode.inTensors.at(inTensorId++) =
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "kv_cache_idx"));
    }
}

atb::Status QwenDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    QwenLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    QwenDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

atb::Status QwenDecoderModel::AddNodesBeforeLayer()
{
    if (!this->param.skipWordEmbedding) { CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding()); }
    CHECK_OPERATION_STATUS_RETURN(AddDynamicNTK());
    CHECK_OPERATION_STATUS_RETURN(AddMuls());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    if (this->param.enableFlashComm) {
        CHECK_OPERATION_STATUS_RETURN(this->AddSplitHiddenStates());
    }
    return atb::NO_ERROR;
}

atb::Status QwenDecoderModel::AddNodesAfterLayer()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
    if (this->param.enableFlashComm) {
        CHECK_OPERATION_STATUS_RETURN(this->AddAllGather());
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    return atb::NO_ERROR;
}

void QwenDecoderModel::SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam)
{
    atb_speed::base::DecoderModel::SetLmHeadParam(lmHeadParam);
    // 浮点场景：单卡lmhead无法走aclnn,固定走ATB接口
    lmHeadParam.linearParallelParam.fusionLinearParam.matmulBackend = atb_speed::common::OpBackend::ATB;
}

atb::Status QwenDecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);
    if (param.enableFlashComm) {
        BindDapHostTensor(this->sendCounts, "send_counts");
        BindDapHostTensor(this->sdispls, "sdispls");
        BindDapHostTensor(this->sendCount, "send_count");
        BindDapHostTensor(this->recvCounts, "recv_counts");
        BindDapHostTensor(this->rdispls, "rdispls");
        BindDapHostTensor(this->recvCount, "recv_count");
    }
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
        if (!this->param.isPrefill && this->param.enableOmniAttention) {
            auto upperBound = this->param.skipWordEmbedding ? 1 : 2;
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
    if (tensorIdx != UINT32_MAX) { graph_.inTensors.at(tensorIdx).hostData = qLen.data(); }

    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}

} // namespace qwen
} // namespace atb_speed