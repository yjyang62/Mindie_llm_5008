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
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/attention/attention_edge.h"

#include "atb_speed/log.h"

#include "models/minicpm/layer/decoder_layer_edge.h"

#include "mlp_edge.h"

namespace atb_speed {
    namespace minicpm {

        std::map<std::string, std::vector<std::string>> GetMiniCpmLayerInTensorCandidates()
        {
            std::map<std::string, std::vector<std::string>> miniCpmLayerInTensorCandidates = {
                {"default", {
                    "in_hidden_states", "in_input_norm_weight", "in_qkv_weight", "in_attention_out_weight",
                    "in_mlp_gate_up_weight", "in_mlp_down_weight", "in_post_attention_norm_weight",
                    "in_attention_mask", "in_position_id", "in_cos_emb", "in_sin_emb", "in_seq_len",
                    "in_place_holder", "in_past_key", "in_past_value"}}
            };
            return miniCpmLayerInTensorCandidates;
        }

        std::map<std::string, std::vector<std::string>> GetQuantMiniCpmLayerInTensorCandidates()
        {
            std::map<std::string, std::vector<std::string>> miniCpmLayerInTensorCandidates = {
                {"default", {
                    "in_hidden_states", "in_input_norm_weight", "in_place_holder",
                    "in_place_holder", "in_place_holder", "in_qkv_weight", "in_qkv_weight_quant_bias",
                    "in_qkv_weight_deq_scale", "in_qkv_weight_input_offset", "in_qkv_weight_input_scale",
                    "in_place_holder", "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder",
                    "in_attention_out_weight", "in_attention_out_weight_quant_bias",
                    "in_attention_out_weight_deq_scale",
                    "in_attention_out_weight_input_offset", "in_attention_out_weight_input_scale",
                    "in_place_holder", "in_post_attention_norm_weight",
                    "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_mlp_gate_up_weight", "in_mlp_gate_up_weight_quant_bias",
                    "in_mlp_gate_up_weight_deq_scale", "in_mlp_gate_up_weight_input_offset",
                    "in_mlp_gate_up_weight_input_scale", "in_place_holder", "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder", "in_mlp_down_weight",
                    "in_mlp_down_weight_quant_bias", "in_mlp_down_weight_deq_scale",
                    "in_mlp_down_weight_input_offset",
                    "in_mlp_down_weight_input_scale", "in_place_holder",
                    "in_attention_mask", "in_position_id", "in_cos_emb", "in_sin_emb", "in_seq_len",
                    "in_place_holder", "in_past_key", "in_past_value"}
                    }};
            return miniCpmLayerInTensorCandidates;
        }

        std::map<std::string, std::vector<std::string>> GetMiniCpmLayerIntermediateTensorCandidates()
        {
            std::map<std::string, std::vector<std::string>> miniCpmLayerIntermediateTensorCandidates = {
                {"default", {"intermediate_attention_out", "intermediate_mlp_out"}},
            };
            return miniCpmLayerIntermediateTensorCandidates;
        }

        std::map<std::string, std::vector<std::string>> GetMiniCpmLayerQuantIntermediateTensorCandidates()
        {
            std::map<std::string, std::vector<std::string>> miniCpmLayerIntermediateTensorCandidates = {
                {"default", {"intermediate_attention_out", "intermediate_mlp_out", "intermediate_mlp_norm_out"}},
            };
            return miniCpmLayerIntermediateTensorCandidates;
        }

        int64_t SetAttentionParam(atb_speed::common::AttentionParam &attentionParam, const DecoderLayerParam &param)
        {
            attentionParam.normEps = param.normEps;
            attentionParam.layerId = param.layerId;
            attentionParam.numHiddenLayers = param.numHiddenLayers;
            attentionParam.numAttentionHeads = param.numAttentionHeads;
            attentionParam.numKeyValueHeads = param.numKeyValueHeads;
            attentionParam.hiddenSize = param.hiddenSize;
            attentionParam.seqLength = param.seqLength;
            attentionParam.isGQA = param.isGQA;
            attentionParam.isPrefill = param.isPrefill;
            attentionParam.isQuant = param.isQuant;
            return atb::NO_ERROR;
        }

        int64_t AddAttention(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                             std::map<std::string, uint32_t> &tensorMap)
        {
            atb_speed::common::AttentionParam attentionParam;
            SetAttentionParam(attentionParam, param);
            atb::Node attentionNode;

            atb_speed::common::AttentionEdge(attentionParam, &attentionNode.operation);
            std::vector<std::string> attentionInTensorNames = {};
            if (attentionParam.isQuant) {
                attentionInTensorNames = { "in_hidden_states",
                                           "in_input_norm_weight",
                                           "in_qkv_weight",
                                           "in_qkv_weight_input_scale",
                                           "in_qkv_weight_input_offset",
                                           "in_qkv_weight_deq_scale",
                                           "in_qkv_weight_quant_bias",
                                           "in_attention_out_weight",
                                           "in_attention_out_weight_input_scale",
                                           "in_attention_out_weight_input_offset",
                                           "in_attention_out_weight_deq_scale",
                                           "in_attention_out_weight_quant_bias",
                                           "in_attention_mask",
                                           "in_position_id",
                                           "in_cos_emb",
                                           "in_sin_emb",
                                           "in_seq_len",
                                           "in_place_holder",
                                           "in_past_key",
                                           "in_past_value"};
            } else {
                attentionInTensorNames = { "in_hidden_states",
                                           "in_input_norm_weight",
                                           "in_qkv_weight",
                                           "in_attention_out_weight",
                                           "in_mlp_gate_up_weight",
                                           "in_mlp_down_weight",
                                           "in_post_attention_norm_weight",
                                           "in_attention_mask",
                                           "in_position_id",
                                           "in_cos_emb",
                                           "in_sin_emb",
                                           "in_seq_len",
                                           "in_place_holder",
                                           "in_past_key",
                                           "in_past_value"};
            }

            attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attentionInTensorNames);
            std::vector<std::string> attentionOutTensorNames = {"intermediate_attention_out",
                                                                "out_present_key",
                                                                "out_present_value"};
            attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attentionOutTensorNames);
            opGraph.nodes.push_back(attentionNode);
            return atb::NO_ERROR;
        }

        void SetMlpParam(atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam,
                         const DecoderLayerParam &param)
        {
            mlpParam.isBF16 = false;
            mlpParam.layerLinearQuantType = param.linearQuantType;
            mlpParam.layerLinearTransposeType = param.linearTransposeType;
            mlpParam.packQuantType = param.packQuantType.at(1);
            mlpParam.isEdgeHardware = true;

            mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_PACK;
            mlpParam.enableAddNorm = false;
            atb::infer::RmsNormParam mlpRmsNormParam;

            mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
            mlpRmsNormParam.normParam.epsilon = param.normEps;
            mlpParam.normParamType = mlpRmsNormParam;
            atb::infer::RmsNormParam mlpRmsNormQuantParam;

            mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
            mlpRmsNormQuantParam.normParam.epsilon = param.normEps;
            mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;

            mlpParam.normQuantParamType = mlpRmsNormQuantParam;
            mlpParam.supportLora = false;
            mlpParam.loraEnableGMM = false;
            mlpParam.supportLcoc = false;
        }

        void SetMlpParamQuant(atb_speed::common::MlpLiteParam &mlpParam, const DecoderLayerParam &param)
        {
            mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
            if (param.isGQA) {
                mlpParam.transposeB = true;
            } else {
                mlpParam.transposeB = false;
            }

            mlpParam.isBias = false;
            mlpParam.noGate = false;
            mlpParam.isPack = true;
            mlpParam.isQuant = true;
        }

        int64_t AddMlp(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                       std::map<std::string, uint32_t> &tensorMap)
        {
            if (!param.isQuant) {
                atb::Node mlpParallelNode;
                atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
                SetMlpParam(mlpParam, param);
                CHECK_OPERATION_STATUS_RETURN(MlpSwiGLU(mlpParam, &mlpParallelNode.operation));

                std::vector <std::string> mlpInTensorNames = {
                    "intermediate_attention_out", "in_post_attention_norm_weight", "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_mlp_gate_up_weight", "in_place_holder", "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_place_holder", "in_place_holder", "in_mlp_down_weight", "in_place_holder",
                    "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder"};

                std::vector <std::string> mlpOutTensorName = {"intermediate_mlp_out"};
                mlpParallelNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpInTensorNames);
                mlpParallelNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpOutTensorName);
                opGraph.nodes.push_back(mlpParallelNode);
            } else {
                atb::Node mlpRmsNode;
                atb::infer::RmsNormParam rmsNormParam;
                rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
                rmsNormParam.normParam.epsilon = param.normEps;
                CreateOperation(rmsNormParam, &mlpRmsNode.operation);

                mlpRmsNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
                                                                             {"intermediate_attention_out",
                                                                              "in_post_attention_norm_weight"});

                mlpRmsNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_norm_out"});

                opGraph.nodes.push_back(mlpRmsNode);
                atb::Node mlpParallelNode;
                atb_speed::common::MlpLiteParam mlpParam;
                SetMlpParamQuant(mlpParam, param);
                atb_speed::common::MlpLiteLayer(mlpParam, &mlpParallelNode.operation);

                std::vector<std::string> mlpInTensorNames = {
                    "intermediate_mlp_norm_out", "in_post_attention_norm_weight", "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_mlp_gate_up_weight", "in_mlp_gate_up_weight_input_scale",
                    "in_mlp_gate_up_weight_input_offset", "in_mlp_gate_up_weight_deq_scale",
                    "in_mlp_gate_up_weight_quant_bias", "in_place_holder", "in_place_holder", "in_place_holder",
                    "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder", "in_mlp_down_weight",
                    "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder", "in_place_holder"};

                std::vector<std::string> mlpOutTensorName = {"intermediate_mlp_out"};
                mlpParallelNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpInTensorNames);
                mlpParallelNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpOutTensorName);
                opGraph.nodes.push_back(mlpParallelNode);
            }
            return atb::NO_ERROR;
        }

        std::map<std::string, uint32_t> ConstructTensorMap(const DecoderLayerParam &param, uint32_t &inTensorNum,
                                                           uint32_t &outTensorNum, uint32_t &internalTensorNum)
        {
            std::map<std::string, std::vector<std::string>> attnInTensor = {};
            auto miniCpmLayerInTensorCandidates  = attnInTensor;
            auto miniCpmLayerIntermediateTensorCandidates = attnInTensor;
            if (param.isQuant) {
                miniCpmLayerInTensorCandidates = GetQuantMiniCpmLayerInTensorCandidates();
                miniCpmLayerIntermediateTensorCandidates = GetMiniCpmLayerQuantIntermediateTensorCandidates();
            } else {
                miniCpmLayerInTensorCandidates = GetMiniCpmLayerInTensorCandidates();
                miniCpmLayerIntermediateTensorCandidates = GetMiniCpmLayerIntermediateTensorCandidates();
            }
            std::vector<std::string> inTensorList = {};
            std::vector<std::string> intermediateTensorList = {};
            std::vector<std::string> outTensorList = {"out_decoder_layer", "out_present_key", "out_present_value"};

            atb_speed::common::AddTensorToList(miniCpmLayerInTensorCandidates, "default", inTensorList);
            atb_speed::common::AddTensorToList(miniCpmLayerIntermediateTensorCandidates, "default",
                                               intermediateTensorList);

            inTensorNum = inTensorList.size();
            outTensorNum = outTensorList.size();
            internalTensorNum = intermediateTensorList.size();

            return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
        }

        atb::Status AddAttentionScale(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                                      std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node attMulsNode;
            float scale = param.scaleDepth / sqrt(param.numHiddenLayers);
            atb::infer::ElewiseParam scaleParam;
            scaleParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
            scaleParam.mulsParam.varAttr = scale;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(scaleParam, &attMulsNode.operation));
            attMulsNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
            attMulsNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
            opGraph.nodes.push_back(attMulsNode);
            return atb::NO_ERROR;
        }

        atb::Status AddAttentionResidual(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node selfResidualAddNode;
            atb::infer::ElewiseParam addParam;
            addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
            selfResidualAddNode.inTensorIds =
                    atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states", "intermediate_attention_out"});
            selfResidualAddNode.outTensorIds =
                    atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
            opGraph.nodes.push_back(selfResidualAddNode);
            return atb::NO_ERROR;
        }

        atb::Status AddMlpScale(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                                std::map<std::string, uint32_t> &tensorMap)
        {
            float scale = param.scaleDepth / sqrt(param.numHiddenLayers);
            atb::Node mlpMulsNode;
            atb::infer::ElewiseParam scaleMlpParam;
            scaleMlpParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
            scaleMlpParam.mulsParam.varAttr = scale;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(scaleMlpParam, &mlpMulsNode.operation));
            mlpMulsNode.inTensorIds =
                    atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
            mlpMulsNode.outTensorIds =
                    atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
            opGraph.nodes.push_back(mlpMulsNode);
            return atb::NO_ERROR;
        }

        atb::Status AddMlpResidual(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node mlpResidualAddNode;
            atb::infer::ElewiseParam addParam;
            addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
            mlpResidualAddNode.inTensorIds =
                    atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out",
                                                                    "intermediate_mlp_out"});
            mlpResidualAddNode.outTensorIds =
                    atb_speed::common::GetTensorIdxList(tensorMap, {"out_decoder_layer"});
            opGraph.nodes.push_back(mlpResidualAddNode);
            return atb::NO_ERROR;
        }

        atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
        {
            const uint64_t headDim = 64;
            constexpr int DIM_NUM_4 = 4; // 4维
            atb::GraphParam opGraph;
            std::map<std::string, uint32_t> tensorMap =
                    ConstructTensorMap(param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
            ATB_SPEED_LOG_DEBUG("layer graph inTensorNum " << opGraph.inTensorNum);
            ATB_SPEED_LOG_DEBUG("layer graph outTensorNum " << opGraph.outTensorNum);
            ATB_SPEED_LOG_DEBUG("layer graph internalTensorNum " << opGraph.internalTensorNum);

            opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";

            CHECK_OPERATION_STATUS_RETURN(AddAttention(opGraph, param, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddAttentionScale(opGraph, param, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddAttentionResidual(opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddMlp(opGraph, param, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddMlpScale(opGraph, param, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddMlpResidual(opGraph, tensorMap));

            if (param.isPrefill) {
                opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                             atb::SVector<atb::TensorDesc> &outTensorDescs) {
                    outTensorDescs.at(0) = inTensorDescs.at(0); // batch
                    outTensorDescs.at(1) = inTensorDescs.at(0); // bs
                    outTensorDescs.at(1).shape.dimNum = DIM_NUM_4; // 4维
                    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // 第0维bs
                    outTensorDescs.at(1).shape.dims[1] = param.numKeyValueHeads; // 第1维KeyValueHeads
                    outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // 第1维赋值第2维
                    outTensorDescs.at(1).shape.dims[3] = headDim; // 第3维

                    outTensorDescs.at(2) = inTensorDescs.at(0); // 第2个输出
                    outTensorDescs.at(2).shape.dimNum = 4; // 4维2
                    outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // 第0维bs2
                    outTensorDescs.at(2).shape.dims[1] = param.numKeyValueHeads; // 第1维KeyValueHeads2
                    outTensorDescs.at(2).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // 第1维赋值第2维
                    outTensorDescs.at(2).shape.dims[3] = headDim; // 第3维
                    return atb::NO_ERROR;
                };
            } else {
                opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                             atb::SVector<atb::TensorDesc> &outTensorDescs) {
                    outTensorDescs.at(0) = inTensorDescs.at(0);
                    outTensorDescs.at(1) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_past_key"));
                    outTensorDescs.at(1).shape.dims[2] += param.seqLength; // 第2维加1
                    outTensorDescs.at(2) = inTensorDescs.at(
                        atb_speed::common::GetTensorIdx(tensorMap, "in_past_value"));
                    outTensorDescs.at(2).shape.dims[2] += param.seqLength; // 第2维加1
                    return atb::NO_ERROR;
                };
            }

            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
            return atb::NO_ERROR;
        }
    } // namespace minicpm
} // namespace atb_speed