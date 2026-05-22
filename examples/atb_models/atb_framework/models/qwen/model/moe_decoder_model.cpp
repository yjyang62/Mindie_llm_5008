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
#include "nlohmann/json.hpp"
#include "vector"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "models/qwen/layer/moe_decoder_layer.h"
#include "models/qwen/model/moe_decoder_model.h"

namespace atb_speed {
namespace qwen {

// Weight count
const int WEIGHT_COUNT_PER_LAYER = 55;
const int WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
const int WEIGHT_COUNT_POST_NORM = 1;
const int WEIGHT_COUNT_LM_HEAD = 1;

// Operation count
const int OPERATION_COUNT_BEFORE_LAYER = 2;  // wte(wordEmbed) + gather(cos/sin embedding)
const int OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

constexpr size_t ATTN_LINEAR_TYPE_LENGTH = 6;
constexpr size_t MLP_LINEAR_TYPE_LENGTH = 4;
constexpr size_t MOE_LINEAR_TYPE_LENGTH = 4;


HcclComm QwenMoeParam::dispatchAndCombineHcclComm = nullptr;
std::string QwenMoeParam::dispatchAndCombinecommDomain = "";

void QwenMoeParam::SetHcclComm() const
{
    if (!isPrefill && enableAllToAllMC2 && expertParallelDegree == 2) { // 2: dynamic ep level
        // Assign commDomain by rankIds and rank
        if (dispatchAndCombineHcclComm != nullptr) {
            ATB_SPEED_LOG_DEBUG("Reuse the hccl communication group for dispatch and combine.");
        } else {
            atb_speed::common::ParallelInfo moeEpParallelInfo = mapping.Get(base::MOE_EP);
            dispatchAndCombinecommDomain = GetSingleton<ExternalCommManager>().GetCommDomain(
                moeEpParallelInfo.groupId, moeEpParallelInfo.rankIds, moeEpParallelInfo.rank,
                moeEpParallelInfo.defaultBackend, moeEpParallelInfo.bufferSize, 0, false);

            dispatchAndCombineHcclComm = \
                GetSingleton<ExternalCommManager>().GetCommPtr(dispatchAndCombinecommDomain);
            ATB_SPEED_LOG_DEBUG("Create the hccl communication group for dispatch and combine.");
        }
    }
}

void QwenMoeParam::ParseBasicParams(const nlohmann::json &paramJson)
{
    numAttentionHeadsPerRank = CheckPositive(paramJson["numAttentionHeadsPerRank"].get<int>());
    hiddenSizePerAttentionHead = CheckPositive(paramJson["hiddenSizePerAttentionHead"].get<int>());
}

void QwenMoeParam::AddLogInfo()
{
    ATB_SPEED_LOG_DEBUG("MoeDecoderModel param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill
                  << ", isBF16:" << isBF16
                  << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: "
                  << isLmHeadParallel << ", lmHeadTransposeType: " << lmHeadTransposeType
                  << ", enableSwiGLU: " << enableSwiGLU << "enableLcoc: " << enableLcoc
                  << ", normEps:" << normEps << ", numAttentionHeadsPerRank:"
                  << numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                  << ", numHiddenLayers:" << numHiddenLayers
                  << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank
                  << ", rank:" << rank << ", worldSize:" << worldSize << ", backend:" << backend
                  << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen << ", rankTableFile:" << rankTableFile
                  << ", numOfExperts:" << numOfExperts << ", expertParallelDegree:" << expertParallelDegree
                  << ", numOfSelectedExperts:" << numOfSelectedExperts << "routingMethod: " << routingMethod
                  << ", packQuantType: " << packQuantType << ", attnLinearQuantType" << attnLinearQuantType
                  << ", mlpLinearQuantType: " << mlpLinearQuantType << ", moeLinearQuantType: " << moeLinearQuantType
                  << ", attnLinearTransposeTyp: " << attnLinearTransposeType
                  << ", mlpLinearTransposeType: " << mlpLinearTransposeType
                  << ", moeLinearTransposeType: " << moeLinearTransposeType);
}

void QwenMoeParam::AddParamJson(const std::string &param)
{
    nlohmann::json paramJson = atb_speed::base::StringToJson(param);
    if (paramJson.contains("enableEPWB")) {
        enableEPWB = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableEPWB");
    }
    if (paramJson.contains("numOfRedundantExpert")) {
        numOfRedundantExpert = atb_speed::base::FetchJsonParam<int>(paramJson, "numOfRedundantExpert");
    }
    if (paramJson.contains("hasSharedExpert")) {
        hasSharedExpert = paramJson.at("hasSharedExpert").get<bool>();
    }
    if (paramJson.contains("enableAllToAllMC2")) {
        enableAllToAllMC2 = paramJson.at("enableAllToAllMC2").get<bool>();
    }
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    for (auto item : paramJson["isDenseLayer"]) {
        isDenseLayer.push_back(item.get<bool>());
    }
    if (paramJson.contains("enableExpertCumSumOutput")) {
        enableExpertCumSumOutput = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableExpertCumSumOutput");
    }
    if (paramJson.contains("enableLoadBalance")) {
        enableLoadBalance = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableLoadBalance");
    }
}

void QwenMoeParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = atb_speed::base::StringToJson(param);
    ParseParam(paramJson);
    ParseBasicParams(paramJson);
    if (rank >= worldSize) {
        std::stringstream ss;
        ss << "worldSize must be greater than rank, please check." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    backend = paramJson["backend"].get<std::string>();
    AddParamJson(param);

    for (auto item : paramJson["attnLinearQuantType"]) {
        attnLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(attnLinearQuantType, numHiddenLayers, ATTN_LINEAR_TYPE_LENGTH);

    for (auto item : paramJson["attnLinearTransposeType"]) {
        attnLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearParamsSufficient(attnLinearTransposeType, numHiddenLayers, ATTN_LINEAR_TYPE_LENGTH);
    SetHcclComm();
    AddLogInfo();
}

MoeDecoderModel::MoeDecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

std::map<std::string, std::vector<std::string>> GetQwenMoeModelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenMoeModelInTensorCandidates = {
        {"default", {
            "in_tensor_input_ids", "in_tensor_position_ids", "in_tensor_cos_table", "in_tensor_sin_table",
            "in_tensor_attention_mask", "in_tensor_block_tables", "in_tensor_slots", "in_tensor_kvcache_idx",
            "in_tensor_token_offset", "in_tensor_place_holder", "in_tensor_seq_len", "in_tensor_logits_indices",
            "in_expert_array_model", "in_expert_group_model", "in_one_hot_model", "in_zero_hot_model"}},
        {"parallel_input", {
            "in_attn_padding_idx_model", "in_attn_unpadding_idx_model",
            "in_ffn_padding_idx_model", "in_ffn_unpadding_idx_model",
            "in_lm_head_skip_padding_token_indices_model",
            "in_attention_padding_idx_slice", "in_start_expert_idx_model",
            "in_device_expert_count_model",
            "in_lty_idx_model", "in_moe_idx_model", "in_post_lmhead_unpadding_indices"}},
        {"epwb", {
            "in_expert_routing_map_model"}},
        {"force_load_balance", {
            "in_fake_topk_model"}},
        {"q_len", {"q_len"}},
    };
    return qwenMoeModelInTensorCandidates;
}

void MoeDecoderModel::ConstructInTensorMap()
{
    auto qwenMoeModelInTensorCandidates = GetQwenMoeModelInTensorCandidates();
    atb_speed::common::AssignTensorIdx(qwenMoeModelInTensorCandidates, "default", this->inTensorMap);
    atb_speed::common::AssignTensorIdx(qwenMoeModelInTensorCandidates, "parallel_input", this->inTensorMap);
    if (param_.enableEPWB) {
        atb_speed::common::AssignTensorIdx(qwenMoeModelInTensorCandidates, "epwb", this->inTensorMap);
    }
    if (param_.enableLoadBalance) {
        atb_speed::common::AssignTensorIdx(qwenMoeModelInTensorCandidates, "force_load_balance", this->inTensorMap);
    }
    // 添加perfix_cache所需的Tensor
    ATB_SPEED_LOG_INFO("model enableSplitFuse: " << param_.enableSplitFuse);
    if (param_.enableSpeculate || param_.enableSplitFuse) {
        atb_speed::common::AssignTensorIdx(qwenMoeModelInTensorCandidates, "q_len", this->inTensorMap);
    }
}

std::map<std::string, std::vector<std::string>> GetQwenMoeModelInternalTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenMoeModelInternalTensorCandidates = {
        {"default", {
            "internal_tensor_hidden_states", "internal_tensor_cos_emb",
            "internal_tensor_sin_emb", "internal_tensor_layer_out_base", "last_hidden_states"}},
    };
    return qwenMoeModelInternalTensorCandidates;
}


void MoeDecoderModel::ConstructInternalTensorMap()
{
    auto qwenMoeModelInternalTensorCandidates = GetQwenMoeModelInternalTensorCandidates();
    atb_speed::common::AssignTensorIdx(
        qwenMoeModelInternalTensorCandidates, "default", this->internalTensorMap);
}

std::map<std::string, std::vector<std::string>> GetQwenMoeModelOutensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenMoeModelOutTensorCandidates = {
        {"default", {"logits"}},
    };
    return qwenMoeModelOutTensorCandidates;
}

void MoeDecoderModel::ConstructOutTensorMap()
{
    this->outTensorMap.clear();
    // 添加默认的Tensor
    auto qwenMoeModelOutTensorCandidates = GetQwenMoeModelOutensorCandidates();
    atb_speed::common::AssignTensorIdx(
        qwenMoeModelOutTensorCandidates, "default", this->outTensorMap);

    if (param_.enableExpertCumSumOutput) {
        uint32_t currentTensorIdx = this->outTensorMap.size();
        uint32_t moeLayerNum = param_.numHiddenLayers - param.firstKDenseReplace;
        for (uint32_t i = 0; i < moeLayerNum; i++) {
            this->outTensorMap["layer_" + std::to_string(i) + "_activation_count_per_expert"] = currentTensorIdx;
            currentTensorIdx++;
        }
    }
}

atb::Status MoeDecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    ATB_SPEED_LOG_DEBUG("Enter MoeDecoderModel InferShape");
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;

    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
    
    uint32_t logitsIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_logits_indices");
    if (param_.isFA) {  // unpadInputs = false
        outTensorDescs.at(0).shape.dims[1] =
            param_.isPrefill ? inTensorDescs.at(logitsIndicesIdx).shape.dims[0] : 1;
    } else {  // unpadInputs = true
        if (param.mapping.Get(base::ATTN_DP).IsEnabled() && !param.enableDpOut) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
        } else if (param_.isPrefill) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
        } else {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        }
    }

    if (param_.isLmHeadParallel) {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = \
            CheckIntMulOverFlow(vocabSizePerRank,
                static_cast<int64_t>(param.mapping.Get(base::LM_HEAD_TP).rankIds.size()));;
    } else {
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = vocabSizePerRank;
    }

    uint32_t outTensorIdx = 1;
    if (param_.enableExpertCumSumOutput) {  // 不支持和DAP同时开启
        uint32_t moeLayernum = param_.numHiddenLayers - static_cast<uint32_t>(param.firstKDenseReplace);
        for (uint32_t i = 0; i < moeLayernum; i++) {
            outTensorDescs.at(outTensorIdx) = atb::TensorDesc{};
            outTensorDescs.at(outTensorIdx).format = ACL_FORMAT_ND;
            outTensorDescs.at(outTensorIdx).dtype = ACL_INT64;
            outTensorDescs.at(outTensorIdx).shape.dimNum = 1;
            outTensorDescs.at(outTensorIdx).shape.dims[0] = param.numOfDeviceExperts;
            outTensorIdx++;
        }
    }
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddNodesBeforeLayer()
{
    CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddNodesAfterLayer()
{
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    return atb::NO_ERROR;
}

uint32_t MoeDecoderModel::CalcWeightTensorSize()
{
    const int weightTensorSize = WEIGHT_COUNT_WORD_EMBEDDINGNODE + WEIGHT_COUNT_PER_LAYER * param_.numHiddenLayers +
                                 WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    return weightTensorSize;
}

atb::Status MoeDecoderModel::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    auto wordEmbeddingNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
    wordEmbeddingParam.unpadInputs = !param_.isFA;

    if (param_.isEmbeddingParallel) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::WORD_EMBED_TP);
        wordEmbeddingParam.tensorParallelInfo.rank = parallelInfo.rank;
        wordEmbeddingParam.tensorParallelInfo.worldSize = parallelInfo.rankIds.size();
        wordEmbeddingParam.tensorParallelInfo.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(
            wordEmbeddingParam.tensorParallelInfo.hcommInfo,
            wordEmbeddingParam.tensorParallelInfo.commDomain);
    };
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
    wordEmbeddingNode->operation.reset(op);
    wordEmbeddingNode->inTensors = {
        &graph_.weightTensors.at(0),                    // shape: [vocabSize + 1, hiddenSize]
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_input_ids"))
    };
    wordEmbeddingNode->outTensors = {&graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_hidden_states"))};
    ATB_SPEED_LOG_DEBUG("wordEmbeddingNode is doing!");
    graph_.nodes.push_back(*wordEmbeddingNode);
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    auto posEmbeddingNode = std::make_unique<atb_speed::Model::Node>();
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    posEmbeddingNode->operation.reset(op);
    posEmbeddingNode->inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_position_ids")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_cos_table")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_sin_table")),
    };
    posEmbeddingNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_cos_emb")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_sin_emb"))
    };
    graph_.nodes.push_back(*posEmbeddingNode);
    ATB_SPEED_LOG_DEBUG("posEmbeddingNode is doing!");
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::SetLayerParam(atb_speed::qwen::MoeDecoderLayerParam &layerParam, const int layerId)
{
    layerParam.isFA = param_.isFA;
    layerParam.isPrefill = param_.isPrefill;
    layerParam.isBF16 = param_.isBF16;
    layerParam.enableSwiGLU = param_.enableSwiGLU;
    layerParam.enableLcoc = param_.enableLcoc;
    layerParam.enableFusedRouting = param_.enableFusedRouting;
    layerParam.packQuantType = param_.packQuantType[layerId];
    layerParam.attnLinearQuantType = param_.attnLinearQuantType[layerId];
    layerParam.mlpLinearQuantType = param_.mlpLinearQuantType[layerId];
    layerParam.moeLinearQuantType = param_.moeLinearQuantType[layerId];
    layerParam.attnLinearTransposeType = param_.attnLinearTransposeType[layerId];
    layerParam.mlpLinearTransposeType = param_.mlpLinearTransposeType[layerId];
    layerParam.moeLinearTransposeType = param_.moeLinearTransposeType[layerId];
    layerParam.normEps = param_.normEps;
    layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
    layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
    layerParam.rank = param_.rank;
    layerParam.worldSize = param_.worldSize;
    layerParam.backend = param_.backend;
    layerParam.rankTableFile = param_.rankTableFile;
    layerParam.layerId = layerId;
    layerParam.useQKNorm = param.useQKNorm;
    layerParam.linearHasBias = param.linearHasBias[layerId];
    layerParam.enableIntraLayerAddNorm = param.enableIntraLayerAddNorm;
    layerParam.mapping = param_.mapping;
    layerParam.enableDpOut = param_.enableDpOut;
    if (layerId == static_cast<int>(param.numHiddenLayers - 1)) {
        layerParam.isLastLayer = true;
    }
    layerParam.enableLoadBalance = param_.enableLoadBalance;
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::SetLayerMoeParam(atb_speed::qwen::MoeDecoderLayerParam &layerParam, const int layerId)
{
    layerParam.enableEPWB = param_.enableEPWB;
    layerParam.numOfRedundantExpert = param_.numOfRedundantExpert;
    layerParam.numOfSelectedExperts = param_.numOfSelectedExperts;
    layerParam.expertParallelDegree = param_.expertParallelDegree;
    layerParam.numOfExperts = param_.numOfExperts;
    layerParam.routingMethod = param_.routingMethod;
    layerParam.processLogits = param.processLogits;
    layerParam.hasSharedExpert = param_.hasSharedExpert;
    layerParam.numOfDeviceExperts = param_.numOfDeviceExperts;
    layerParam.enableInitQuant = param.enableInitQuant;
    layerParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    layerParam.enableAllToAllMC2 = param_.enableAllToAllMC2;
    layerParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    if (param.expertParallelDegree == 2) { // 2: dynamic ep level
        layerParam.isDynamicEp = true;
    }
    layerParam.dispatchAndCombineHcclComm = param_.dispatchAndCombineHcclComm;
    layerParam.dispatchAndCombinecommDomain = param_.dispatchAndCombinecommDomain;
    layerParam.enableDispatchCombineV2 = param_.enableDispatchCombineV2;
    layerParam.isDenseLayer = param_.isDenseLayer[layerId];
    if (param_.isDenseLayer[layerId]) {
        layerParam.hasMoe = false;
        layerParam.hasSharedExpert = true;
        layerParam.hasSharedExpertGate = false;
    }
    layerParam.enableExpertCumSumOutput = param_.enableExpertCumSumOutput;
    layerParam.enableSplitFuse = param_.enableSplitFuse;
    layerParam.enableSpeculate = param_.enableSpeculate;
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddParallelHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
{
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attn_padding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attn_unpadding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ffn_padding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ffn_unpadding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_lm_head_skip_padding_token_indices_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attention_padding_idx_slice"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_start_expert_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_device_expert_count_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_lty_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_moe_idx_model"));
    if (param_.enableEPWB) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_expert_routing_map_model"));
    }
    if (param_.enableLoadBalance) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_fake_topk_model"));
    }
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddLayerTensor(atb_speed::Model::Node &layerNode, size_t &inTensorId,
                                            const int &layerId)
{
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_hidden_states"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_expert_array_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_expert_group_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_one_hot_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_zero_hot_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_cos_emb"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_sin_emb"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_attention_mask"));
    layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_seq_len"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_token_offset"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_kvcache_idx"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_block_tables"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_slots"));
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddSingleLayer(uint32_t layerId)
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node layerNode;
    atb_speed::qwen::MoeDecoderLayerParam layerParam;
    SetLayerParam(layerParam, layerId);
    SetLayerMoeParam(layerParam, layerId);
    CHECK_OPERATION_STATUS_RETURN(atb_speed::qwen::MoeDecoderLayer(layerParam, &op));
    layerNode.operation.reset(op);
    layerNode.inTensors.resize(layerNode.operation->GetInputNum());
    size_t inTensorId = 0;

    for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
        layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
            layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
    }
    AddLayerTensor(layerNode, inTensorId, layerId);
    AddParallelHostWeight(layerNode, inTensorId);
    if (param_.enableSpeculate || param_.enableSplitFuse) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len")
        );
    }

    if (layerId == param_.numHiddenLayers - 1) {
        layerNode.outTensors = {
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "last_hidden_states"))
        };
    } else {
        layerNode.outTensors = {
            &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_hidden_states"))
        };
    }
    if (layerId >= static_cast<uint32_t>(param.firstKDenseReplace) && param_.enableExpertCumSumOutput) {
        uint32_t ExpertCumSumStartIdx = 1;
        uint32_t moeLayerId = layerId - static_cast<uint32_t>(param.firstKDenseReplace);
        layerNode.outTensors.push_back(&graph_.outTensors.at(ExpertCumSumStartIdx + moeLayerId));
    }
    ATB_SPEED_LOG_DEBUG("layerNode_" << layerId << " is doing!");
    graph_.nodes.push_back(layerNode);
    ATB_SPEED_LOG_DEBUG("layerNode is doing!");
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    auto finalNormNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    finalNormNode->operation.reset(op);
    const size_t finalLayerNormWeightTensorId =
        this -> graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    finalNormNode->inTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "last_hidden_states")),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId)
    };
    finalNormNode->outTensors = {
        // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "last_hidden_states"))
    };

    ATB_SPEED_LOG_DEBUG("finalNormNode is doing!");
    graph_.nodes.push_back(*finalNormNode);
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::AddLmhead()
{
    atb::Operation *op = nullptr;

    auto lmHeadNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param_.isFA;
    lmHeadParam.gatherAhead = param_.isPrefill || (param.mapping.Get(base::ATTN_DP).IsEnabled() && !param.enableDpOut);
    lmHeadParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = param_.isBF16;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = param_.lmHeadTransposeType;
    lmHeadParam.linearParallelParam.unpadInputs = !param_.isFA;
    if (param.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::LM_HEAD_TP);
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = parallelInfo.rank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = parallelInfo.rankIds.size();
        lmHeadParam.linearParallelParam.tensorParallelInfo.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(
            lmHeadParam.linearParallelParam.tensorParallelInfo.hcommInfo,
            lmHeadParam.linearParallelParam.tensorParallelInfo.commDomain);
    }
    CHECK_OPERATION_STATUS_RETURN(LmHead(lmHeadParam, &op));
    ATB_SPEED_LOG_DEBUG("lmHeadNode is doing!");

    lmHeadNode->operation.reset(op);
    const size_t finalLinearWeightTensorId = this -> graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    lmHeadNode->inTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "last_hidden_states")),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_logits_indices"))
    };
    if (this->param_.enableGreedyPostProcessing) {
        lmHeadNode->inTensors.emplace_back(&graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "logits_offset_tensor")));
    } else {
        lmHeadNode->inTensors.emplace_back(&graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder")));
    }

    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode->outTensors = {&graph_.outTensors.at(atb_speed::common::GetTensorIdx(this->outTensorMap, "logits"))};

    ATB_SPEED_LOG_DEBUG("MoeDecoderModel build graph success");
    graph_.nodes.push_back(*lmHeadNode);
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::ParseParam(const std::string &param)
{
    ATB_SPEED_LOG_DEBUG("ParseParam start.");
    CHECK_PARAM_LT(param.size(), MAX_PARAM_STRING_LENGTH);
    nlohmann::json paramJson = atb_speed::base::StringToJson(param);

    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        int tokenOffset = item.get<int>();
        CHECK_PARAM_LT(tokenOffset, MAX_PARAM_VALUE);
        tokenOffset_.push_back(tokenOffset);
        ATB_SPEED_LOG_DEBUG("tokenOffset value: " << item);
    }
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        int seqLen = item.get<int>();
        CHECK_PARAM_LT(seqLen, MAX_PARAM_VALUE);
        seqLen_.push_back(seqLen);
        ATB_SPEED_LOG_DEBUG("Prefill" << paramJson["isPrefill"] << "seqLen value: " << item);
    }
    qLen_.clear();
    for (auto item : paramJson["qLen"]) {
        this->qLen_.push_back(item.get<int>());
        ATB_SPEED_LOG_DEBUG("qLen value: " << item);
    }
    ATB_SPEED_LOG_DEBUG("ParseParam end.");
    return atb::NO_ERROR;
}

atb::Status MoeDecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

    if (nodeId != 0) {
        // 仅需在graph的intensor中bind一次
        return atb::NO_ERROR;
    }

    uint32_t tokenOffsetTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_token_offset");
    if (tokenOffsetTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tokenOffsetTensorIdx).hostData = tokenOffset_.data();
    }
    uint32_t seqLenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_seq_len");
    if (seqLenTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(seqLenTensorIdx).hostData = seqLen_.data();
    }
    uint32_t qLenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len");
    if (qLenTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(qLenTensorIdx).hostData = qLen_.data();
    }

    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}
} // namespace qwen
} // namespace atb_speed
