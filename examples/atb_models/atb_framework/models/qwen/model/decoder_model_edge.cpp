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
#include "models/qwen/model/decoder_model_edge.h"

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"
#include "vector"

#include "atb_speed/log.h"
#include "models/qwen/layer/decoder_layer_edge.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"

namespace atb_speed {
namespace qwen {

// Weight count
const int WEIGHT_COUNT_WORD_EMBEDDING = 1;
const int WEIGHT_COUNT_POST_NORM = 1;
const int WEIGHT_COUNT_LM_HEAD = 1;
static const uint64_t OUT_TENSOR_COUNT = 1;
const uint64_t RESULT_DIM_4 = 4;
static const int QKNORM_WEIGHT_PER_LAYER = 52;
uint64_t HEAD_DIM = 128;

void DecoderModelEdge::Param::ParseBasicParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("withEmbedding")) {
        withEmbedding = paramJson["withEmbedding"].get<bool>();
    }
    isFA = paramJson["isFA"].get<bool>();
    isPrefill = paramJson["isPrefill"].get<bool>();
    isBF16 = paramJson["isBF16"].get<bool>();
    isEmbeddingParallel = paramJson["isEmbeddingParallel"].get<bool>();
    isLmHeadParallel = paramJson["isLmHeadParallel"].get<bool>();
    lmHeadTransposeType = CheckPositive(paramJson["lmHeadTransposeType"].get<int>());
    supportSwiGLU = paramJson["supportSwiGLU"].get<bool>();
    supportLcoc = paramJson["supportLcoc"].get<bool>();
    kvQuant = paramJson["kvQuant"].get<bool>();
    if (paramJson.contains("enableFA3")) {
        this->enableFA3 = paramJson["enableFA3"].get<bool>();
    }
    if (paramJson.contains("enableQScale")) {
        this->enableQScale = paramJson["enableQScale"].get<bool>();
    }
    rmsNormEps = paramJson["rmsNormEps"].get<float>();
    numAttentionHeadsPerRank = CheckPositive(paramJson["numAttentionHeadsPerRank"].get<int>());
    hiddenSizePerAttentionHead = CheckPositive(paramJson["hiddenSizePerAttentionHead"].get<int>());
    numHiddenLayers = CheckNumHiddenLayersValid(paramJson["numHiddenLayers"].get<int>());
    numKeyValueHeadsPerRank = CheckPositive(paramJson["numKeyValueHeadsPerRank"].get<int>());
    rank = paramJson["rank"].get<int>();
    worldSize = CheckPositive(paramJson["worldSize"].get<int>());
    backend = paramJson["backend"].get<std::string>();
    if (paramJson.contains("vocabSize")) {
        vocabSize = paramJson["vocabSize"].get<int>();
    }
    if (paramJson.contains("head_dim")) {
        HEAD_DIM = paramJson["head_dim"].get<int>();
    }
    if (paramJson.contains("hiddenSize")) {
        hiddenSize = paramJson["hiddenSize"].get<int>();
    }
    if (paramJson.contains("isQuant")) {
        isQuant = paramJson["isQuant"].get<bool>();
    }
    if (paramJson.contains("useQKNorm")) {
        useQKNorm = paramJson["useQKNorm"].get<bool>();
    }
}

void DecoderModelEdge::Param::AddParamJson(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    if (paramJson.contains("quantGroupSize")) {
        quantGroupSize = paramJson["quantGroupSize"].get<uint32_t>();
    }
    for (auto item : paramJson["packQuantType"]) {
        packQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckPackQuantParamsSufficient(packQuantType, numHiddenLayers);
    for (auto item : paramJson["linearQuantType"]) {
        linearQuantType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearPackParamsSufficient(linearQuantType, numHiddenLayers);
    for (auto item : paramJson["linearTransposeType"]) {
        linearTransposeType.push_back(item.get<std::vector<int>>());
    }
    CheckLinearPackParamsSufficient(linearTransposeType, numHiddenLayers);
    if (paramJson.contains("enableAddNorm")) {
        enableAddNorm = paramJson["enableAddNorm"].get<bool>();
    }
}

void DecoderModelEdge::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    ParseBasicParams(paramJson);
    if (rank >= worldSize) {
        std::stringstream ss;
        ss << "worldSize must be greater than rank, please check." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    backend = paramJson["backend"].get<std::string>();
    AddParamJson(param);
    PrintParam();
}

void DecoderModelEdge::Param::PrintParam()
{
    ATB_SPEED_LOG_DEBUG("DecoderModel param"
                        << ", isFA:" << isFA << ", isPrefill:" << isPrefill << ", isBF16:" << isBF16
                        << ", withEmbedding: " << withEmbedding << ", isEmbeddingParallel: " << isEmbeddingParallel
                        << ", isLmHeadParallel: " << isLmHeadParallel << ", supportSwiGLU: " << supportSwiGLU
                        << ", rmsNormEps:" << rmsNormEps << ", numAttentionHeadsPerRank:" << numAttentionHeadsPerRank
                        << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead
                        << ", numHiddenLayers:" << numHiddenLayers
                        << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank << ", supportLcoc:" << supportLcoc
                        << ", rank:" << rank << ", worldSize:" << worldSize << ", backend:" << backend
                        << ", kvQuant: " << kvQuant  << ", enableAddNorm:" << enableAddNorm);
}

DecoderModelEdge::DecoderModelEdge(const std::string &param) : Model("DecoderModel", param)
{
    param_.FromString(param);
    if (param_.useQKNorm) {
        weightCountPerLayer_ = QKNORM_WEIGHT_PER_LAYER;
    }
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

DecoderModelEdge::~DecoderModelEdge() {}

std::map<std::string, std::vector<std::string>> GetQwenModelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenInTensorCandidates = {
        {"default",
         {"input_ids_or_embedding", "positional_ids", "cosine_table", "sine_table", "attention_mask", "block_tables",
          "slots", "kv_cache_idx", "token_offset", "seq_len", "logits_indices", "place_holder", "in_tensor_past_key"}},
    };
    return qwenInTensorCandidates;
}

void DecoderModelEdge::ConstructInTensorMap()
{
    auto qwenInTensorCandidates = GetQwenModelInTensorCandidates();
    uint32_t tensorIdx = 0;

    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(qwenInTensorCandidates, "default", tensorIdx, this->inTensorMap_);

    this->inTensorCount_ = tensorIdx;

    std::stringstream ss;
    for (auto tensor = this->inTensorMap_.cbegin(); tensor != this->inTensorMap_.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("tensor map" << ss.str());
}

std::map<std::string, std::vector<std::string>> GetQwenModelInternalTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenInternalTensorCandidates = {
        {"default", {"hidden_states"}},
        {"rope", {"cosine_embedding", "sine_embedding"}},
    };
    return qwenInternalTensorCandidates;
}

void DecoderModelEdge::ConstructInternalTensorMap()
{
    auto qwenInternalTensorCandidates = GetQwenModelInternalTensorCandidates();
    uint32_t tensorIdx = 0;

    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(qwenInternalTensorCandidates, "default", tensorIdx, this->internalTensorMap_);

    // 添加rope的Tensor
    atb_speed::common::AssignTensorIdx(qwenInternalTensorCandidates, "rope", tensorIdx, this->internalTensorMap_);
    
    this->internalTensorCount_ = tensorIdx;
    std::stringstream ss;
    for (auto tensor = this->internalTensorMap_.cbegin(); tensor != this->internalTensorMap_.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("inteltensor map" << ss.str());
}

std::map<std::string, std::vector<std::string>> GetQwenModelOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenoutTensorCandidates = {
        {"default", {"hidden_states"}},
        {"isEdgeHardware", {"out_past_key", "out_past_value"}}};
    return qwenoutTensorCandidates;
}

void DecoderModelEdge::ConstructOutTensorMap()
{
    auto qwenoutTensorCandidates = GetQwenModelOutTensorCandidates();
    uint32_t tensorIdx = 0;

    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(qwenoutTensorCandidates, "default", tensorIdx, this->outTensorMap_);
    for (uint32_t layerID = 0; layerID < param_.numHiddenLayers; layerID++) {
        atb_speed::common::AssignTensorIdx(qwenoutTensorCandidates,
                                           "isEdgeHardware", tensorIdx, this->outTensorMap_);
    }

    this->outTensorCount_ = tensorIdx;

    std::stringstream ss;
    for (auto tensor = this->outTensorMap_.cbegin(); tensor != this->outTensorMap_.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("OUTtensor map" << ss.str());
}

uint32_t DecoderModelEdge::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t DecoderModelEdge::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status DecoderModelEdge::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                         std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    CHECK_TENSORDESC_DIMNUM_VALID(graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dimNum);
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = (inTensorDescs.at(0).shape.dimNum + 1);
    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // bs
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // seqlen
    outTensorDescs.at(0).shape.dims[2] = param_.vocabSize; // dimes[2]=vocabSize

    if (param_.isPrefill) {
        for (uint32_t keyId = 0; keyId < param_.numHiddenLayers; ++keyId) {
            outTensorDescs.at(1 + keyId) = outTensorDescs.at(0);
            outTensorDescs.at(1 + keyId).shape.dimNum = RESULT_DIM_4; // 四维
            outTensorDescs.at(1 + keyId).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // bs
            outTensorDescs.at(1 + keyId).shape.dims[1] = param_.numKeyValueHeadsPerRank;
            outTensorDescs.at(1 + keyId).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // dims[2]=seqlen
            outTensorDescs.at(1 + keyId).shape.dims[3] = HEAD_DIM; // dims[3]head dim
        }
        for (uint32_t valueId = 0; valueId < param_.numHiddenLayers; ++valueId) {
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId) = outTensorDescs.at(0);
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dimNum = RESULT_DIM_4; // 四维
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            // key head
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[1] = param_.numKeyValueHeadsPerRank;
            // dims[2]=seqlen
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[2] = inTensorDescs.at(0).shape.dims[1];
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[3] = HEAD_DIM; // dims[3]=head dim
        }
    } else {
        uint32_t inTensorPastKey = atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_past_key");
        const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(inTensorPastKey);
        for (uint32_t keyId = 0; keyId < param_.numHiddenLayers; ++keyId) {
            outTensorDescs.at(1 + keyId) = keyTensorDesc;
            outTensorDescs.at(1 + keyId).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
            outTensorDescs.at(1 + keyId).format = graph_.weightTensors.at(0).desc.format;
            outTensorDescs.at(1 + keyId).shape.dimNum = keyTensorDesc.shape.dimNum;
            outTensorDescs.at(1 + keyId).shape.dims[2] += 1; // dims[2] + 1
        }
        const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(inTensorPastKey + param_.numHiddenLayers);
        for (uint32_t valueId = 0; valueId < param_.numHiddenLayers; ++valueId) {
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId) = valueTensorDesc;
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).dtype =
                graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).format = graph_.weightTensors.at(0).desc.format;
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dimNum = valueTensorDesc.shape.dimNum;
            outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[2] += 1; // dims[2] + 1
        }
    }

    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    if (param_.withEmbedding) {
        atb_speed::Model::Node wordEmbeddingNode;
        atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
        wordEmbeddingParam.unpadInputs = !param_.isFA;
        atb_speed::common::WordEmbedding(wordEmbeddingParam, &op);
        wordEmbeddingNode.operation.reset(op);
        wordEmbeddingNode.inTensors = {
            &graph_.weightTensors.at(0), // shape: [vocabSize + 1, hiddenSize]
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "input_ids_or_embedding"))};
        wordEmbeddingNode.outTensors = {
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "hidden_states"))};
        ATB_SPEED_LOG_DEBUG("[+] wordEmbeddingNode");
        graph_.nodes.push_back(wordEmbeddingNode);
    }

    return atb::NO_ERROR;
}


int64_t DecoderModelEdge::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node positionalEmbeddingGatherNode;
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    positionalEmbeddingGatherNode.operation.reset(op);
    positionalEmbeddingGatherNode.inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "positional_ids")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "cosine_table")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "sine_table")),
    };

    positionalEmbeddingGatherNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "cosine_embedding")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "sine_embedding"))};
    ATB_SPEED_LOG_DEBUG("[+] positionalEmbeddingGatherNode");
    graph_.nodes.push_back(positionalEmbeddingGatherNode);

    return atb::NO_ERROR;
}

void DecoderModelEdge::SetLayerParam(DecoderLayerParam &layerParam, uint32_t layerId)
{
    layerParam.isFA = param_.isFA;
    layerParam.isPrefill = param_.isPrefill;
    layerParam.isBF16 = param_.isBF16;
    layerParam.supportSwiGLU = param_.supportSwiGLU;
    layerParam.supportLcoc = param_.supportLcoc;
    layerParam.supportSpeculate = param_.supportSpeculate;
    layerParam.enableSplitFuse = param_.enableSplitFuse;
    layerParam.supportLora = param_.supportLora;
    layerParam.loraEnableGMM = param_.loraEnableGMM;
    layerParam.packQuantType = param_.packQuantType[layerId];
    layerParam.linearQuantType = param_.linearQuantType[layerId];
    layerParam.linearTransposeType = param_.linearTransposeType[layerId];
    layerParam.kvQuant = param_.kvQuant;
    layerParam.enableFA3 = param_.enableFA3;
    layerParam.enableLogN = param_.enableLogN;
    layerParam.enableQScale = param_.enableQScale;
    layerParam.rmsNormEps = param_.rmsNormEps;
    layerParam.quantGroupSize = param_.quantGroupSize;
    layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
    layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
    layerParam.numAttentionHeads = param_.numAttentionHeadsPerRank;
    layerParam.numKeyValueHeads = param_.numAttentionHeadsPerRank;
    layerParam.hiddenSize = param_.hiddenSize;
    layerParam.rank = param_.rank;
    layerParam.worldSize = param_.worldSize;
    layerParam.backend = param_.backend;
    layerParam.enableAddNorm = param_.enableAddNorm;
    layerParam.layerId = layerId;
    layerParam.isQuant = param_.isQuant;
    layerParam.useQKNorm = param_.useQKNorm;
}

int64_t DecoderModelEdge::SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId,
                                                   uint32_t &inTensorId)
{
    for (uint32_t weightTensorId = 0; weightTensorId < weightCountPerLayer_; ++weightTensorId) {
        layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
            CheckIntMulOverFlow(layerId, weightCountPerLayer_) + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDING);
    }
    layerNode.inTensors.at(inTensorId++) =
        param_.withEmbedding
            ? &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "hidden_states"))
            : &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "input_ids_or_embedding"));

    layerNode.inTensors.at(inTensorId++) =
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "cosine_embedding"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "sine_embedding"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "attention_mask"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "seq_len"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "token_offset"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "kv_cache_idx"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "block_tables"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "slots"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "positional_ids"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "place_holder"));
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_past_key") + layerId);
    layerNode.inTensors.at(inTensorId++) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_past_key")
            + layerId + param_.numHiddenLayers);
    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::SetLayerNodeoutput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    layerNode.outTensors.at(0) =
           layerNode.inTensors.at(weightCountPerLayer_);
    // k cache in outtensors 1
    layerNode.outTensors.at(1) =
           &graph_.outTensors.at(layerId + 1);
    // v cache in outtensors 2
    layerNode.outTensors.at(2) =
           &graph_.outTensors.at(1 + layerId + param_.numHiddenLayers);
    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::AddLayer()
{
    atb::Operation *op = nullptr;
    for (uint32_t layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        atb_speed::Model::Node layerNode;
        atb_speed::qwen::DecoderLayerParam layerParam;
        SetLayerParam(layerParam, layerId);
        CHECK_OPERATION_STATUS_RETURN(atb_speed::qwen::DecoderLayer(layerParam, &op));
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        uint32_t inTensorId = 0;
        CHECK_OPERATION_STATUS_RETURN(SetLayerNodeDefaultInput(layerNode, layerId, inTensorId));
        layerNode.outTensors.resize(3); // EdgeHardware layer output num is 3
        CHECK_OPERATION_STATUS_RETURN(SetLayerNodeoutput(layerNode, layerId));
        ATB_SPEED_LOG_DEBUG("[+] layerNode_" << layerId);
        graph_.nodes.push_back(layerNode);
    }

    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node finalNormNode;
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    finalNormNode.operation.reset(op);
    const uint32_t finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    finalNormNode.inTensors = {
        param_.withEmbedding
            ? &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "hidden_states"))
            : &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "input_ids_or_embedding")),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {finalNormNode.inTensors.at(0)}; // 输出原地写在输入上
    ATB_SPEED_LOG_DEBUG("[+] finalNormNode");
    graph_.nodes.push_back(finalNormNode);

    return atb::NO_ERROR;
}


int64_t DecoderModelEdge::AddLmhead()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node lmHeadNode;
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    linearParam.transposeA = false;
    linearParam.transposeB = true;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParam, &op));
    lmHeadNode.operation.reset(op);
    lmHeadNode.inTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "hidden_states")),
        &graph_.weightTensors.at(graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD),
    };
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};
    graph_.nodes.push_back(lmHeadNode);
    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::BuildGraph()
{
    // 准备inTensor
    ConstructInTensorMap();
    const uint64_t inTensorCount = this->inTensorCount_ + param_.numHiddenLayers * 2 - 1;
    this->graph_.inTensors.resize(inTensorCount);
    ATB_SPEED_LOG_DEBUG("graph_.inTensorCount_ " << this->inTensorCount_);

    // 准备internalTensor
    ConstructInternalTensorMap();
    this->graph_.internalTensors.resize(this->internalTensorCount_);
    ATB_SPEED_LOG_DEBUG("graph_.internalTensorCount_ " << this->internalTensorCount_);
    ConstructOutTensorMap();
    
    // 准备outTensor
    const uint64_t outTensorCount = OUT_TENSOR_COUNT + param_.numHiddenLayers * 2;
    graph_.outTensors.resize(outTensorCount);

    const uint64_t weightTensorSize = WEIGHT_COUNT_WORD_EMBEDDING +
                                      CheckIntMulOverFlow(weightCountPerLayer_, param_.numHiddenLayers) +
                                      WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    ATB_SPEED_LOG_DEBUG("WEIGHT_SIZE:"<<weightTensorSize);
    graph_.weightTensors.resize(weightTensorSize);

    ATB_SPEED_LOG_DEBUG("weightTensors.size=" << graph_.weightTensors.size()
                                              << ", inTensors.size=" << graph_.inTensors.size()
                                              << ", outTensors.size=" << graph_.outTensors.size()
                                              << ", internalTensor.size=" << graph_.internalTensors.size());
    ATB_SPEED_LOG_DEBUG("DecoderModel build graph begin");
    CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddLayer());
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    ATB_SPEED_LOG_DEBUG("DecoderModel build graph success");
    return atb::NO_ERROR;
}

atb::Status DecoderModelEdge::ParseParam(const std::string &param)
{
    ATB_SPEED_LOG_DEBUG("ParseParam start.");
    CHECK_PARAM_LT(param.size(), MAX_PARAM_STRING_LENGTH);
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }

    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        int tokenOffset = item.get<int>();
        CHECK_PARAM_LT(tokenOffset, MAX_PARAM_VALUE);
        tokenOffset_.push_back(tokenOffset);
        ATB_SPEED_LOG_DEBUG("token offset value: " << item);
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
        int qLen = item.get<int>();
        CHECK_PARAM_LT(qLen, MAX_PARAM_VALUE);
        qLen_.push_back(qLen);
        ATB_SPEED_LOG_DEBUG("qLen value: " << item);
    }
    ATB_SPEED_LOG_DEBUG("ParseParam end.");
    return atb::NO_ERROR;
}

} // namespace qwen
} // namespace atb_speed