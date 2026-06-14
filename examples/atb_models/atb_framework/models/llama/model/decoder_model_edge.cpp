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
#include <vector>
#include <atb/types.h>
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "operations/fusion/utils.h"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/base/model.h"
#include "models/llama/model/decoder_model_edge.h"

namespace atb_speed {
namespace llama {

// Weight count
const uint32_t WEIGHT_COUNT_WORD_EMBEDDING = 1;
const uint32_t WEIGHT_COUNT_POST_NORM = 1;
const uint32_t WEIGHT_COUNT_LM_HEAD = 1;
static const uint64_t OUT_TENSOR_COUNT = 1;
const uint64_t RESULT_DIM_4 = 4;

// 全局变量
uint64_t g_headDim = 128;

void DecoderModelEdge::Param::ParseBasicParams(const nlohmann::json &paramJson)
{
    skipWordEmbedding = paramJson["skipWordEmbedding"].get<bool>();
    isFA = paramJson["isFA"].get<bool>();
    isPrefill = paramJson["isPrefill"].get<bool>();
    isBF16 = paramJson["isBF16"].get<bool>();
    lmHeadTransposeType = paramJson["lmHeadTransposeType"].get<int>();
    supportSwiGLU = paramJson["enableSwiGLU"].get<bool>();
    if (paramJson.contains("attnBackend")) {
        attnBackend = paramJson["attnBackend"].get<int>();
    }
    rmsNormEps = paramJson["normEps"].get<float>();
    numAttentionHeadsPerRank = paramJson["numAttentionHeadsPerRank"].get<int>();
    hiddenSizePerAttentionHead = paramJson["hiddenSizePerAttentionHead"].get<int>();
    numHiddenLayers = paramJson["numHiddenLayers"].get<int>();
    numKeyValueHeadsPerRank = paramJson["numKeyValueHeadsPerRank"].get<int>();
    hiddenSize = paramJson["hiddenSize"].get<int>();
    rank = paramJson["rank"].get<int>();
    worldSize = CheckPositive(paramJson["worldSize"].get<int>());
    if (paramJson.contains("vocabSize")) {
        vocabSize = paramJson["vocabSize"].get<int>();
    }
    if (paramJson.contains("head_dim")) {
        g_headDim = paramJson["head_dim"].get<int>();
    }
    if (paramJson.contains("isQuant")) {
        isQuant = paramJson["isQuant"].get<bool>();
    }
    if (paramJson.contains("outputHiddenStates")) {
        outputHiddenStates = paramJson["outputHiddenStates"].get<bool>();
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
        throw std::runtime_error(ss.str());
    }
    if (paramJson.contains("enableAddNorm")) { enableAddNorm = paramJson["enableAddNorm"].get<bool>(); }
    if (paramJson.contains("rankTableFile")) { rankTableFile = paramJson["rankTableFile"].get<std::string>(); }
    if (paramJson.contains("positionEmbeddingType")) {
        positionEmbeddingType = paramJson["positionEmbeddingType"].get<std::string>();
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
    if (paramJson.contains("linearHasBias")) {
        linearHasBias.clear();
        for (auto item : paramJson["linearHasBias"]) {
            linearHasBias.push_back(item.get<bool>());
        }
    }
    CheckLinearPackParamsSufficient(linearTransposeType, numHiddenLayers);
    if (paramJson.contains("quantGroupSize")) { quantGroupSize = paramJson["quantGroupSize"].get<uint32_t>(); }
}

void DecoderModelEdge::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        throw std::runtime_error(ss.str());
    }
    ParseBasicParams(paramJson);
    if (rank >= worldSize) {
        std::stringstream ss;
        ss << "worldSize must be greater than rank, please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    backend = paramJson["backend"].get<std::string>();
    AddParamJson(param);
    PrintParam();
}

void DecoderModelEdge::Param::PrintParam()
{
    ATB_SPEED_LOG_DEBUG("DecoderModel param" << ", skipWordEmbedding:" << skipWordEmbedding << ", isFA:" << isFA
                  << ", isPrefill:" << isPrefill << ", isBF16:" << isBF16
                  << ", lmHeadTransposeType: " << lmHeadTransposeType <<", supportSwiGLU: " << supportSwiGLU
                  << ", rmsNormEps:"
                  << rmsNormEps << ", numAttentionHeadsPerRank:" << numAttentionHeadsPerRank
                  << ", hiddenSizePerAttentionHead:" << hiddenSizePerAttentionHead << ", numHiddenLayers:"
                  << numHiddenLayers << ", numKeyValueHeadsPerRank:" << numKeyValueHeadsPerRank << ", rank:" << rank
                  << ", worldSize:" << worldSize << ", backend:" << backend << ", rankTableFile" << rankTableFile
                  << ", attnBackend:" << attnBackend
                  << ", positionEmbeddingType" << positionEmbeddingType
                  << ", enableAddNorm" << enableAddNorm
                  << ", quantGroupSize:" << quantGroupSize);
}

DecoderModelEdge::DecoderModelEdge(const std::string &param) : Model("DecoderModelEdge", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

DecoderModelEdge::~DecoderModelEdge() {}

std::map<std::string, std::vector<std::string>> GetLlamaModelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> llamaInTensorCandiadates = {
        {"default", {
            "input_ids", "positional_ids", "cosine_table", "sine_table", "attention_mask",
            "place_holder", "seq_len", "in_tensor_past_key"}
        },
    };
    return llamaInTensorCandiadates;
}

void DecoderModelEdge::ConstructInTensorMap()
{
    auto llamaInTensorCandiadates = GetLlamaModelInTensorCandidates();
    if (param_.skipWordEmbedding) {
        llamaInTensorCandiadates["default"].at(0) = "input_embedding"; // skipWordEmbedding时，第0个输入为input_embedding
    }
    uint32_t tensorIdx = 0;

    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(llamaInTensorCandiadates, "default", tensorIdx, this->inTensorMap_);

    this->inTensorCount_ = tensorIdx;

    std::stringstream ss;
    for (auto tensor = this->inTensorMap_.cbegin(); tensor != this->inTensorMap_.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("tensor map" << ss.str());
}

std::map<std::string, std::vector<std::string>> GetLlamaModelInternalTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> llamaInternalTensorCandiadates = {
        {"default", {"internal_tensor_hidden_states"}},
        {"rope", {"cosine_embedding", "sine_embedding"}},
    };
    return llamaInternalTensorCandiadates;
}

void DecoderModelEdge::ConstructInternalTensorMap()
{
    auto llamaInternalTensorCandiadates = GetLlamaModelInternalTensorCandidates();
    uint32_t tensorIdx = 0;

    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(
        llamaInternalTensorCandiadates, "default", tensorIdx, this->internalTensorMap_);

    // 添加rope的Tensor
    if (this->param_.positionEmbeddingType == "ROPE") {
        atb_speed::common::AssignTensorIdx(
            llamaInternalTensorCandiadates, "rope", tensorIdx, this->internalTensorMap_);
    }

    this->internalTensorCount_ = tensorIdx;

    std::stringstream ss;
    for (auto tensor = this->internalTensorMap_.cbegin(); tensor != this->internalTensorMap_.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("tensor map" << ss.str());
}

uint32_t DecoderModelEdge::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t DecoderModelEdge::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status DecoderModelEdge::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs, \
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    CHECK_TENSORDESC_DIMNUM_VALID(graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dimNum);
    outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    if (!param_.skipWordEmbedding) {
        outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1; // 扩展维度
    }
    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // bs
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // seqlen
    uint32_t dimSize = param_.outputHiddenStates ? param_.hiddenSize : param_.vocabSize;
    outTensorDescs.at(0).shape.dims[2] = dimSize; // dims[2]: hidden_size or vocabSize
    if (param_.isPrefill) {
        PrefillInferShape(inTensorDescs, outTensorDescs);
    } else {
        DecodeInferShape(inTensorDescs, outTensorDescs);
    }

    return atb::NO_ERROR;
}

atb::Status DecoderModelEdge::PrefillInferShape(const std::vector<atb::TensorDesc> &inTensorDescs, \
    std::vector<atb::TensorDesc> &outTensorDescs)
{
    for (uint32_t keyId = 0; keyId < param_.numHiddenLayers; ++keyId) {
            outTensorDescs.at(1 + keyId) = outTensorDescs.at(0);
            outTensorDescs.at(1 + keyId).shape.dimNum = RESULT_DIM_4; // 四维
            outTensorDescs.at(1 + keyId).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // bs
            outTensorDescs.at(1 + keyId).shape.dims[1] = param_.numKeyValueHeadsPerRank;
            outTensorDescs.at(1 + keyId).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // dims[2]=seqlen
            outTensorDescs.at(1 + keyId).shape.dims[3] = g_headDim; // dims[3]head dim
    }
    for (uint32_t valueId = 0; valueId < param_.numHiddenLayers; ++valueId) {
        outTensorDescs.at(1 + param_.numHiddenLayers + valueId) = outTensorDescs.at(0);
        outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dimNum = RESULT_DIM_4; // 四维
        outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        // key head
        outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[1] = param_.numKeyValueHeadsPerRank;
        // dims[2]=seqlen
        outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[2] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(1 + param_.numHiddenLayers + valueId).shape.dims[3] = g_headDim; // dims[3]=head dim
    }
    return atb::NO_ERROR;
}

atb::Status DecoderModelEdge::DecodeInferShape(const std::vector<atb::TensorDesc> &inTensorDescs, \
    std::vector<atb::TensorDesc> &outTensorDescs)
{
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
    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    if (!param_.skipWordEmbedding) {
        atb_speed::Model::Node wordEmbeddingNode;
        atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
        wordEmbeddingParam.unpadInputs = !param_.isFA;
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
        wordEmbeddingNode.operation.reset(op);
        wordEmbeddingNode.inTensors = {
            &graph_.weightTensors.at(0),                    // shape: [vocabSize + 1, hiddenSize]
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "input_ids"))
        };
        wordEmbeddingNode.outTensors = {
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, \
                "internal_tensor_hidden_states"))
        };
        graph_.nodes.push_back(wordEmbeddingNode);
    }

    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    if (param_.positionEmbeddingType == "ROPE") {
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
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, "sine_embedding"))
        };
        graph_.nodes.push_back(positionalEmbeddingGatherNode);
    }

    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::AddSlice(atb::SVector<int64_t> offsets, atb::SVector<int64_t> size, \
    const std::string &inTensorName, const std::string &internalTersorMap)
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node sliceNode;
    atb::infer::SliceParam sliceNodeParam;
    sliceNodeParam.offsets = offsets;
    sliceNodeParam.size = size;
    CREATE_OPERATION(sliceNodeParam, &op);
    sliceNode.operation.reset(op);
    sliceNode.inTensors = {&graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, inTensorName))};
    sliceNode.outTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_, \
        internalTersorMap))};
    graph_.nodes.push_back(sliceNode);
    return atb::NO_ERROR;
}

void DecoderModelEdge::SetLayerParam(DecoderLayerParam &layerParam, uint32_t layerId)
{
    layerParam.hiddenSize = param_.hiddenSize;
    layerParam.numHiddenLayers = param_.numHiddenLayers;
    layerParam.seqLength = param_.seqLength;
    layerParam.isFA = param_.isFA;
    layerParam.isPrefill = param_.isPrefill;
    layerParam.isBF16 = param_.isBF16;
    layerParam.supportSwiGLU = param_.supportSwiGLU;
    layerParam.supportLcoc = param_.supportLcoc;
    layerParam.packQuantType = param_.packQuantType.at(layerId);
    layerParam.linearQuantType = param_.linearQuantType.at(layerId);
    layerParam.linearTransposeType = param_.linearTransposeType.at(layerId);
    layerParam.isQuant = param_.isQuant;
    layerParam.linearHasBias = param_.linearHasBias;
    layerParam.attnBackend = param_.attnBackend;
    layerParam.enableAddNorm = param_.enableAddNorm;
    layerParam.rmsNormEps = param_.rmsNormEps;
    layerParam.quantGroupSize = param_.quantGroupSize;
    layerParam.numAttentionHeadsPerRank = param_.numAttentionHeadsPerRank;
    layerParam.hiddenSizePerAttentionHead = param_.hiddenSizePerAttentionHead;
    layerParam.numKeyValueHeadsPerRank = param_.numKeyValueHeadsPerRank;
    layerParam.tensorParallelInfo = {param_.rank, param_.worldSize, param_.backend, param_.rankTableFile};
    if (param_.positionEmbeddingType == "ROPE") {
        layerParam.positionEmbeddingType = atb_speed::llama::ROPE;
    } else if (param_.positionEmbeddingType == "ALIBI") {
        layerParam.positionEmbeddingType = atb_speed::llama::ALIBI;
    }
}

int64_t DecoderModelEdge::SetLayerNodeDefaultInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    for (uint32_t weightTensorId = 0; weightTensorId < weightCountPerLayer_; ++weightTensorId) {
        layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
            CheckIntMulOverFlow(layerId, weightCountPerLayer_) + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDING);
    }
    layerNode.inTensors.at(inTensorId++) = param_.skipWordEmbedding && layerId == 0 ? \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "input_embedding")) : \
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
            this->internalTensorMap_, "internal_tensor_hidden_states"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "attention_mask"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "positional_ids"));
    if (param_.positionEmbeddingType == "ROPE") {
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap_, "cosine_embedding"));
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap_, "sine_embedding"));
    }
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "seq_len"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "place_holder"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_past_key") + layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_past_key")
        + layerId + param_.numHiddenLayers);
        
    layerNode.outTensors = {&graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")),
        &graph_.outTensors.at(layerId + 1), &graph_.outTensors.at(layerId + 1 + param_.numHiddenLayers)};
    return atb::NO_ERROR;
}

int64_t DecoderModelEdge::AddLayer()
{
    atb::Operation *op = nullptr;
    for (uint32_t layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
        atb_speed::Model::Node layerNode;
        atb_speed::llama::DecoderLayerParam layerParam;
        SetLayerParam(layerParam, layerId);
        CHECK_OPERATION_STATUS_RETURN(atb_speed::llama::DecoderLayer(layerParam, &op));
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());
        uint32_t inTensorId = 0;
        CHECK_OPERATION_STATUS_RETURN(SetLayerNodeDefaultInput(layerNode, layerId, inTensorId));
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
        &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId)
    };
    if (!param_.outputHiddenStates) {
        finalNormNode.outTensors = {
            &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states"))
        };
    } else {
        finalNormNode.outTensors = {&graph_.outTensors.at(0)};
    }
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
        &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")),
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

    const uint64_t weightTensorSize =
        WEIGHT_COUNT_WORD_EMBEDDING +
        CheckIntMulOverFlow(weightCountPerLayer_, param_.numHiddenLayers) +
        WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    graph_.weightTensors.resize(weightTensorSize);

    const uint64_t outTensorCount = OUT_TENSOR_COUNT + param_.numHiddenLayers * 2;
    graph_.outTensors.resize(outTensorCount);

    ATB_SPEED_LOG_DEBUG("DecoderModelEdge build graph begin");
    CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddLayer());
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    if (!param_.outputHiddenStates) {
        CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    }
    ATB_SPEED_LOG_DEBUG("DecoderModelEdge build graph success");
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
        throw std::runtime_error(ss.str());
    }

    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        int tokenOffset = item.get<int>();
        tokenOffset_.push_back(tokenOffset);
        ATB_SPEED_LOG_DEBUG("token offset value: " << item);
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        int seqLen = item.get<int>();
        seqLen_.push_back(seqLen);
        ATB_SPEED_LOG_DEBUG("seqLen value: " << item);
    }

    qLen_.clear();
    for (auto item : paramJson["qLen"]) {
        int qLen = item.get<int>();
        qLen_.push_back(qLen);
        ATB_SPEED_LOG_DEBUG("qLen value: " << item);
    }

    blockNumsList_.clear();
    for (auto item : paramJson["blockNumsList"]) {
        int blockNumsList = item.get<int>();
        blockNumsList_.push_back(blockNumsList);
        ATB_SPEED_LOG_DEBUG("blockNumsList value: " << item);
    }
    
    ATB_SPEED_LOG_DEBUG("ParseParam end.");
    return atb::NO_ERROR;
}

atb::Status DecoderModelEdge::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

    uint32_t tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap_, "token_offset");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = tokenOffset_.data();
    }
    if (tensorIdx != UINT32_MAX) {
        tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap_, "seq_len");
        graph_.inTensors.at(tensorIdx).hostData = seqLen_.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap_, "q_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = qLen_.data();
    }

    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}

void DecoderModelEdge::BuildNodeOutTensors(
    int nodeId, atb_speed::Model::Node &node, atb::SVector<atb::TensorDesc>& inTensorDescs)
{
    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.reserve(node.operation->GetOutputNum());
    outTensorDescs.resize(node.operation->GetOutputNum());
    atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);
    if (st != 0) {
        ATB_SPEED_LOG_DEBUG(modelName_ << " nodes[" << nodeId << "] "
                                       << " infer shape fail, error code: " << st);
    }
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_SPEED_LOG_DEBUG(modelName_ << " nodes[" << nodeId << "] outTensorDescs[" << i
                      << "]:" << TensorUtil::TensorDescToString(outTensorDescs.at(i)));
    }
    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        CHECK_THROW(node.outTensors.at(i) == nullptr,
            modelName_ << " nodes[" << nodeId << "] "
                       << "outTensor " << i << "is NULL");
        node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
        if (node.outTensorTypes.at(i) == Model::TensorType::INTERMEDIATE_TENSOR) {
            node.variantPack.outTensors.at(i)
                = Model::MallocInternalTensor(node.outTensors.at(i), nodeId, i, outTensorDescs.at(i));
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
        }
        if (!TensorUtil::TensorDescEqual(node.variantPack.outTensors.at(i).desc, outTensorDescs.at(i))) {
            ATB_SPEED_LOG_DEBUG(modelName_ << "  nodes[" << nodeId << "] new outTensorDescs[" << i
                           << "]:" << TensorUtil::TensorDescToString(outTensorDescs.at(i))
                           << ", node.variantPack.outTensors.at[" << i
                           << "].desc:" << TensorUtil::TensorDescToString(node.variantPack.outTensors.at(i).desc));
        }
    }
}

void DecoderModelEdge::BuildNodeVariantPack(int nodeId)
{
    int operationCountBeforeLayers =
        param_.skipWordEmbedding ? 1 : 2;
    int upperBound = operationCountBeforeLayers;
    int lowerBound = upperBound + param_.numHiddenLayers;
    if (nodeId < upperBound || nodeId >= lowerBound) {
        Model::BuildNodeVariantPack(nodeId);
    } else {
        auto &node = graph_.nodes.at(nodeId);
        atb::SVector<atb::TensorDesc> inTensorDescs;
        inTensorDescs.reserve(node.variantPack.inTensors.size());
        inTensorDescs.resize(node.variantPack.inTensors.size());

        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            CHECK_THROW(node.inTensors.at(i) == nullptr,
                modelName_ << " nodes[" << nodeId << "] "
                           << "inTensor " << i << "is NULL");
            node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
            inTensorDescs.at(i) = node.inTensors.at(i)->desc;

            ATB_SPEED_LOG_DEBUG(modelName_ << " nodes[" << nodeId << "] inTensors[" << i
                          << "]:" << TensorUtil::TensorToString(node.variantPack.inTensors.at(i)));
        }

        DecoderModelEdge::BuildNodeOutTensors(nodeId, node, inTensorDescs);

        auto it = graph_.maxNodeIdTensorMap.find(nodeId);
        if (it != graph_.maxNodeIdTensorMap.end()) {
            for (auto tensorIt : it->second) {
                Model::FreeInternalTensor(tensorIt);
            }
        }
    }
}
} // namespace llama
} // namespace atb_speed
