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
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "operations/fusion/utils.h"

#include "models/minicpm/model/decoder_model_edge.h"

namespace atb_speed {
    namespace minicpm {

        const uint64_t WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
        const uint64_t WEIGHT_COUNT_POST_NORM = 1;
        const uint64_t WEIGHT_COUNT_LM_HEAD = 1;
        const uint64_t WEIGHT_COUNT_QUANT_INC = 5;
        const uint64_t WEIGHT_COUNT_QUANT_INC_1 = 4;
        const uint64_t WEIGHT_COUNT_QUANT_INC_2 = 3;
        const uint64_t WEIGHT_COUNT_QUANT = 50;
        static const uint64_t OUT_TENSOR_COUNT = 1;
        const uint64_t HEAD_DIM = 64;
        const uint64_t RESULT_DIM_4 = 4;

        atb::Status DecoderModelEdge::ParseParam(const std::string &param)
        {
            ATB_SPEED_LOG_DEBUG("ParseParam start.");
            nlohmann::json paramJson;
            try {
                paramJson = nlohmann::json::parse(param);
            } catch (const std::exception &e) {
                std::stringstream ss;
                ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
                ATB_SPEED_LOG_ERROR(ss.str());
                throw std::runtime_error(ss.str());
            }
            if (paramJson.contains("seqLength")) {
                param_.seqLength = paramJson["seqLength"].get<int>();
            }
            if (paramJson.contains("isPrefill")) {
                param_.isPrefill = paramJson["isPrefill"].get<int>();
            }
            ATB_SPEED_LOG_DEBUG("ParseParam end.");
            return atb::NO_ERROR;
        }

        void DecoderModelEdge::Param::ParseBasicParams(const nlohmann::json &paramJson)
        {
            if (paramJson.contains("rmsNormEps")) {
                rmsNormEps = paramJson["rmsNormEps"].get<float>();
            }
            if (paramJson.contains("hiddenSize")) {
                hiddenSize = paramJson["hiddenSize"].get<int>();
            }
            if (paramJson.contains("scaleEmb")) {
                scaleEmb = paramJson["scaleEmb"].get<float>();
            }
            if (paramJson.contains("scaleDepth")) {
                scaleDepth = paramJson["scaleDepth"].get<float>();
            }
            if (paramJson.contains("dimModelBase")) {
                dimModelBase = paramJson["dimModelBase"].get<int>();
            }
            if (paramJson.contains("numHiddenLayers")) {
                numHiddenLayers = paramJson["numHiddenLayers"].get<int>();
            }
            if (paramJson.contains("numAttentionHeads")) {
                numAttentionHeads = paramJson["numAttentionHeads"].get<int>();
            }
            if (paramJson.contains("numKeyValueHeads")) {
                numKeyValueHeads = paramJson["numKeyValueHeads"].get<int>();
            }
            if (paramJson.contains("vocabSize")) {
                vocabSize = paramJson["vocabSize"].get<int>();
            }
            if (paramJson.contains("seqLength")) {
                seqLength = paramJson["seqLength"].get<int>();
            }
            if (paramJson.contains("isGQA")) {
                isGQA = paramJson["isGQA"].get<bool>();
            }
            if (paramJson.contains("isPrefill")) {
                isPrefill = paramJson["isPrefill"].get<bool>();
            }
            if (paramJson.contains("isQuant")) {
                isQuant = paramJson["isQuant"].get<bool>();
            }
            for (auto item : paramJson["packQuantType"]) {
                packQuantType.push_back(item.get<std::vector<int>>());
            }
        }

        void DecoderModelEdge::Param::CheckParam() const
        {
            if (this->numHiddenLayers == 0) {
                std::stringstream ss;
                ss << "Cannot be devided by zero. Param numHiddenLayers is zero!" << std::endl;
                throw std::runtime_error(ss.str());
            }
            if (this->dimModelBase == 0) {
                std::stringstream ss;
                ss << "Cannot be devided by zero. Param dimModelBase is zero!" << std::endl;
                throw std::runtime_error(ss.str());
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
                throw std::runtime_error(ss.str());
            }
            ParseBasicParams(paramJson);
            CheckParam();
            CheckPackQuantParamsSufficient(packQuantType, numHiddenLayers);
            for (auto item : paramJson["linearQuantType"]) {
                linearQuantType.push_back(item.get<std::vector<int>>());
            }
            CheckLinearPackParamsSufficient(linearQuantType, numHiddenLayers);
            for (auto item : paramJson["linearTransposeType"]) {
                linearTransposeType.push_back(item.get<std::vector<int>>());
            }
        }

        DecoderModelEdge::DecoderModelEdge(const std::string &param) : Model("DecoderModelEdge", param)
        {
            param_.FromString(param);
            modelName_ += "_Decoder";
        }

        DecoderModelEdge::~DecoderModelEdge() {}

        std::map<std::string, std::vector<std::string>> GetMiniCpmModelInTensorCandidates()
        {
            std::map<std::string, std::vector<std::string>> minicpmInTensorCandidates = {
                {"default", {
                    "in_tensor_input_ids", "in_tensor_attention_mask", "in_tensor_position_ids",
                    "in_tensor_cos_table", "in_tensor_sin_table", "in_tensor_seq_len",
                    "in_model_place_holder", "in_tensor_past_key"}
                }
            };
            return minicpmInTensorCandidates;
        }

        void DecoderModelEdge::ConstructInTensorMap()
        {
            auto miniCpmInTensorCandidates = GetMiniCpmModelInTensorCandidates();
            uint32_t tensorIdx = 0;

            atb_speed::common::AssignTensorIdx(miniCpmInTensorCandidates, "default", tensorIdx, this->inTensorMap_);

            this->inTensorCount_ = tensorIdx;

            std::stringstream ss;
            for (auto tensor = this->inTensorMap_.cbegin(); tensor != this->inTensorMap_.cend(); ++tensor) {
                ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
            }
            ATB_SPEED_LOG_DEBUG("tensor map" << ss.str());
        }

        std::map<std::string, std::vector<std::string>> GetMiniCpmModelInternalTensorCandidates()
        {
            std::map<std::string, std::vector<std::string>> miniCpmInternalTensorCandidates = {
                {"default", {
                    "internal_tensor_hidden_states", "internal_tensor_cos_emb", "internal_tensor_sin_emb"}},
                {"quant", {
                    "internal_tensor_hidden_states_quant"}}
            };
            return miniCpmInternalTensorCandidates;
        }

        void DecoderModelEdge::ConstructInternalTensorMap()
        {
            auto miniCpmInternalTensorCandidates = GetMiniCpmModelInternalTensorCandidates();
            uint32_t tensorIdx = 0;

            // 添加默认的Tensor
            atb_speed::common::AssignTensorIdx(miniCpmInternalTensorCandidates, "default", tensorIdx,
                                               this->internalTensorMap_);
            if (param_.isQuant) {
                atb_speed::common::AssignTensorIdx(miniCpmInternalTensorCandidates, "quant", tensorIdx,
                                                   this->internalTensorMap_);
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

        atb::Status DecoderModelEdge::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                 std::vector<atb::TensorDesc> &outTensorDescs)
        {
            outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
            outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
            outTensorDescs.at(0).shape.dimNum = (inTensorDescs.at(0).shape.dimNum + 1);
            CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
            CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // 第0维bs
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // 第1维seqLen
            outTensorDescs.at(0).shape.dims[2] = param_.vocabSize; // 第2维vocabSize

            uint32_t inTensorSeqLen = atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_seq_len");
            const atb::TensorDesc &inTensorSeqLenDesc = inTensorDescs.at(inTensorSeqLen);
            static const int LAYER_NUM = param_.numHiddenLayers;

            if (param_.isPrefill) {
                for (int keyId = 0; keyId < param_.numHiddenLayers; ++keyId) {
                    outTensorDescs.at(1 + keyId) = outTensorDescs.at(0);
                    outTensorDescs.at(1 + keyId).shape.dimNum = RESULT_DIM_4; // 四维
                    outTensorDescs.at(1 + keyId).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // 第0维bs
                    outTensorDescs.at(1 + keyId).shape.dims[1] = param_.numKeyValueHeads; // 第1维KeyValueHeads
                    outTensorDescs.at(1 + keyId).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // 第1维赋值第2维
                    outTensorDescs.at(1 + keyId).shape.dims[3] = HEAD_DIM; // 第3维头数
                }
                for (int valueId = 0; valueId < param_.numHiddenLayers; ++valueId) {
                    outTensorDescs.at(1 + LAYER_NUM + valueId) = outTensorDescs.at(0);
                    outTensorDescs.at(1 + LAYER_NUM + valueId).shape.dimNum = RESULT_DIM_4; // 四维
                    outTensorDescs.at(1 + LAYER_NUM + valueId).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // 0维
                    outTensorDescs.at(1 + LAYER_NUM + valueId).shape.dims[1] = param_.numKeyValueHeads;
                    outTensorDescs.at(1 + LAYER_NUM + valueId).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // 2维
                    outTensorDescs.at(1 + LAYER_NUM + valueId).shape.dims[3] = HEAD_DIM; // 第3维
                }
            } else {
                uint32_t inTensorPastKey = atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_past_key");
                const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(inTensorPastKey);
                for (int keyId = 0; keyId < param_.numHiddenLayers; ++keyId) {
                    outTensorDescs.at(1 + keyId) = keyTensorDesc;
                    outTensorDescs.at(1 + keyId).dtype =
                            graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
                    outTensorDescs.at(1 + keyId).format = graph_.weightTensors.at(0).desc.format;
                    outTensorDescs.at(1 + keyId).shape.dimNum = keyTensorDesc.shape.dimNum;
                    outTensorDescs.at(1 + keyId).shape.dims[2] += inTensorSeqLenDesc.shape.dims[0]; // 第2维
                }
                const atb::TensorDesc &valueTensorDesc = inTensorDescs.at(inTensorPastKey + param_.numHiddenLayers);
                for (int valueId = 0; valueId < LAYER_NUM; ++valueId) {
                    outTensorDescs.at(1 + LAYER_NUM + valueId) = valueTensorDesc;
                    outTensorDescs.at(1 + LAYER_NUM + valueId).dtype =
                            graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
                    outTensorDescs.at(1 + LAYER_NUM + valueId).format = graph_.weightTensors.at(0).desc.format;
                    outTensorDescs.at(1 + LAYER_NUM + valueId).shape.dimNum = valueTensorDesc.shape.dimNum;
                    outTensorDescs.at(1 + LAYER_NUM + valueId).shape.dims[2] += inTensorSeqLenDesc.shape.dims[0]; // 2维
                }
            }
            return atb::NO_ERROR;
        }

        int64_t DecoderModelEdge::AddWordEmbedding()
        {
            atb::Operation *op = nullptr;
            atb_speed::Model::Node wordEmbeddingNode;
            atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
            wordEmbeddingParam.unpadInputs = false;
            CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
            wordEmbeddingNode.operation.reset(op);
            wordEmbeddingNode.inTensors = {
                &graph_.weightTensors.at(0), // shape: [vocabSize + 1, hiddenSize]
                &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_input_ids"))
            };
            wordEmbeddingNode.outTensors = {
                &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap_,
                                                    "internal_tensor_hidden_states"))};
            graph_.nodes.push_back(wordEmbeddingNode);
            return atb::NO_ERROR;
        }

        int64_t DecoderModelEdge::AddMuls()
        {
            atb::Operation *op = nullptr;
            atb_speed::Model::Node mulsNode;
            float embedingScale = param_.scaleEmb;
            atb::infer::ElewiseParam embedingScaleMulsParam;
            embedingScaleMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
            embedingScaleMulsParam.mulsParam.varAttr = embedingScale;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(embedingScaleMulsParam, &op));
            mulsNode.operation.reset(op);
            mulsNode.inTensors = {&graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states"))};
            mulsNode.outTensors = {&graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states"))};
       
            graph_.nodes.push_back(mulsNode);
            return atb::NO_ERROR;
        }

        int64_t DecoderModelEdge::AddPositionalEmbedding()
        {
            atb::Operation *op = nullptr;
            atb_speed::Model::Node positionalEmbeddingGatherNode;
            CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
            positionalEmbeddingGatherNode.operation.reset(op);
            positionalEmbeddingGatherNode.inTensors = {
                &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_position_ids")),
                &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_cos_table")),
                &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_sin_table")),
            };
            positionalEmbeddingGatherNode.outTensors = {
                &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_cos_emb")),
                &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_sin_emb"))
            };
            graph_.nodes.push_back(positionalEmbeddingGatherNode);
            return atb::NO_ERROR;
        }

        int64_t DecoderModelEdge::SetLayerParam(int layerId, atb_speed::minicpm::DecoderLayerParam &layerParam)
        {
            layerParam.normEps = param_.rmsNormEps;
            layerParam.scaleDepth = param_.scaleDepth;
            layerParam.numHiddenLayers = param_.numHiddenLayers;
            layerParam.numAttentionHeads = param_.numAttentionHeads;
            layerParam.numKeyValueHeads = param_.numKeyValueHeads;
            layerParam.hiddenSize = param_.hiddenSize;
            layerParam.isGQA = param_.isGQA;
            layerParam.isPrefill = param_.isPrefill;
            layerParam.layerId = layerId;
            layerParam.seqLength = param_.seqLength;
            layerParam.isQuant = param_.isQuant;
            layerParam.packQuantType = param_.packQuantType[layerId];
            layerParam.linearQuantType = param_.linearQuantType[layerId];
            layerParam.linearTransposeType = param_.linearTransposeType[layerId];
            return atb::NO_ERROR;
        }

        int64_t DecoderModelEdge::AddLayer()
        {
            atb::Operation *op = nullptr;
            for (int layerId = 0; layerId < param_.numHiddenLayers; ++layerId) {
                atb_speed::Model::Node layerNode;
                atb_speed::minicpm::DecoderLayerParam layerParam;
                SetLayerParam(layerId, layerParam);
                CHECK_OPERATION_STATUS_RETURN(atb_speed::minicpm::DecoderLayer(layerParam, &op));
                layerNode.operation.reset(op);
                layerNode.inTensors.resize(layerNode.operation->GetInputNum());

                size_t inTensorId = 0;
                layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states"));

                if (param_.isQuant) {
                    weightCountPerLayer = WEIGHT_COUNT_QUANT;
                }

                for (size_t weightTensorId = 0; weightTensorId < weightCountPerLayer; ++weightTensorId) {
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                        layerId * weightCountPerLayer + weightTensorId + WEIGHT_COUNT_WORD_EMBEDDINGNODE);
                }

                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                    atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_attention_mask"));
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                    atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_position_ids"));
                layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_cos_emb"));
                layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_sin_emb"));
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                    atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_seq_len"));
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                    atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_model_place_holder"));
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                    atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_past_key") + layerId);
                layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                    atb_speed::common::GetTensorIdx(this->inTensorMap_, "in_tensor_past_key")
                        + layerId + param_.numHiddenLayers);
                layerNode.outTensors = {
                    &graph_.internalTensors.at(
                        atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")),
                    &graph_.outTensors.at(layerId + 1),
                    &graph_.outTensors.at(layerId + 1 + param_.numHiddenLayers)};

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
            uint32_t finalLayerNormWeightTensorId = 0;
            if (param_.isQuant) {
                finalLayerNormWeightTensorId =
                    graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD -
                    WEIGHT_COUNT_QUANT_INC;
            } else {
                finalLayerNormWeightTensorId =
                    graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
            }
            finalNormNode.inTensors = {
                &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")),
                &graph_.weightTensors.at(finalLayerNormWeightTensorId)
            };
            finalNormNode.outTensors = {
                &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states"))
            };
            graph_.nodes.push_back(finalNormNode);
            return atb::NO_ERROR;
        }

        int64_t DecoderModelEdge::AddLmMuls()
        {
            atb::Operation *op = nullptr;
            atb_speed::Model::Node lmMulsNode;
            float lmScale = 1.0 / (param_.hiddenSize / param_.dimModelBase) ;
            atb::infer::ElewiseParam lmMulsParam;
            lmMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
            lmMulsParam.mulsParam.varAttr = lmScale;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(lmMulsParam, &op));
            lmMulsNode.operation.reset(op);
            lmMulsNode.inTensors = { &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")) };
            lmMulsNode.outTensors = { &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap_, "internal_tensor_hidden_states")) };
            graph_.nodes.push_back(lmMulsNode);
            return atb::NO_ERROR;
        }

        int64_t DecoderModelEdge::QuantLmHeadInput()
        {
            atb::Operation *op = nullptr;
            atb_speed::Model::Node quantLmNode;
            atb::infer::ElewiseParam quantParam;
            quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(quantParam, &op));
            quantLmNode.operation.reset(op);
            quantLmNode.inTensors = {
                &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_,
                                                                           "internal_tensor_hidden_states")),
                &graph_.weightTensors.at(graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD - 1), // 相对偏差1
                &graph_.weightTensors.at(graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD - 2) // 相对偏差2
            };
            quantLmNode.outTensors = { &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap_,
                                                "internal_tensor_hidden_states_quant")) };
            graph_.nodes.push_back(quantLmNode);
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

            if (param_.isQuant) {
                QuantLmHeadInput();
                linearParam.hasBias = true;
                linearParam.outDataType = ACL_FLOAT16;
                CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParam, &op));
                lmHeadNode.operation.reset(op);
                lmHeadNode.inTensors = {
                    &graph_.internalTensors.at(
                        atb_speed::common::GetTensorIdx(this->internalTensorMap_,
                                                        "internal_tensor_hidden_states_quant")),
                    &graph_.weightTensors.at(
                        graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD - WEIGHT_COUNT_QUANT_INC),
                    &graph_.weightTensors.at(
                        graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD - WEIGHT_COUNT_QUANT_INC_1),
                    &graph_.weightTensors.at(
                        graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD - WEIGHT_COUNT_QUANT_INC_2),
                };
            } else {
                CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParam, &op));
                lmHeadNode.operation.reset(op);
                lmHeadNode.inTensors = {
                    &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap_,
                                                                               "internal_tensor_hidden_states")),
                    &graph_.weightTensors.at(graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD),
                };
            }
            lmHeadNode.outTensors = {&graph_.outTensors.at(0)};
            graph_.nodes.push_back(lmHeadNode);
            return atb::NO_ERROR;
        }

        int64_t DecoderModelEdge::BuildGraph()
        {
            // 准备inTensor
            ConstructInTensorMap();
            const uint64_t inTensorCount = this->inTensorCount_ + param_.numHiddenLayers * 2 - 1; // 相对偏差1和2
            graph_.inTensors.resize(inTensorCount);
            ATB_SPEED_LOG_DEBUG("graph_.inTensorCount_ " << this->inTensorCount_);

            // 准备internalTensor
            ConstructInternalTensorMap();
            this->graph_.internalTensors.resize(this->internalTensorCount_);
            ATB_SPEED_LOG_DEBUG("graph_.internalTensorCount_ " << this->internalTensorCount_);
            uint64_t weightTensorSize = 0;
            if (param_.isQuant) {
                weightCountPerLayer = WEIGHT_COUNT_QUANT;
                weightTensorSize =
                    WEIGHT_COUNT_WORD_EMBEDDINGNODE +
                    weightCountPerLayer * static_cast<uint64_t>(param_.numHiddenLayers) +
                    WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD + WEIGHT_COUNT_QUANT_INC;
            } else {
                weightTensorSize =
                    WEIGHT_COUNT_WORD_EMBEDDINGNODE +
                    weightCountPerLayer * static_cast<uint64_t>(param_.numHiddenLayers) +
                    WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
            }
            graph_.weightTensors.resize(weightTensorSize);

            const uint64_t outTensorCount = OUT_TENSOR_COUNT + param_.numHiddenLayers * 2; // key和value乘以2
            graph_.outTensors.resize(outTensorCount);

            ATB_SPEED_LOG_DEBUG("weightTensors.size=" << graph_.weightTensors.size()
                                                      << ", inTensors.size=" << graph_.inTensors.size()
                                                      << ", outTensors.size=" << graph_.outTensors.size()
                                                      << ", internalTensor.size=" << graph_.internalTensors.size());
            ATB_SPEED_LOG_DEBUG("DecoderModelEdge build graph begin");

            CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
            CHECK_OPERATION_STATUS_RETURN(AddMuls());
            CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
            CHECK_OPERATION_STATUS_RETURN(AddLayer());
            CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
            CHECK_OPERATION_STATUS_RETURN(AddLmMuls());
            CHECK_OPERATION_STATUS_RETURN(AddLmhead());
            ATB_SPEED_LOG_DEBUG("DecoderModelEdge build graph success");
            return atb::NO_ERROR;
        }

    } // namespace minicpm
} // namespace atb_speed