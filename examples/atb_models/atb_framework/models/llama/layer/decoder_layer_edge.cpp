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
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/attention/attention_edge.h"
#include "operations/fusion/mlp/mlp.h"
#include "models/llama/layer/decoder_layer_edge.h"

namespace atb_speed {
namespace llama {

std::map<std::string, std::vector<std::string>> GetLlamaLayerInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> llamaLayerInTensorCandiadates = {
        {"default_weight", {
            // shape: [hiddenSize]
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
            // Pack:
            // MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
            // GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
            // No pack:
            // (Q) shape: [numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
            "in_qkv_weight", "in_qkv_bias_0", "in_qkv_descale_0", "in_qkv_offset_0", "in_qkv_scale_0",
            "in_qkv_compress_idx_0",
            // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
            "in_qkv_weight_1", "in_qkv_bias_1", "in_qkv_descale_1", "in_qkv_offset_1", "in_qkv_scale_1",
            "in_qkv_compress_idx_1",
            // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
            "in_qkv_weight_2", "in_qkv_bias_2", "in_qkv_descale_2", "in_qkv_offset_2", "in_qkv_scale_2",
            "in_qkv_compress_idx_2",
            // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
            "in_attention_out_weight", "in_qkv_dense_bias", "in_qkv_dense_descale", "in_qkv_dense_offset",
            "in_qkv_dense_scale", "in_qkv_dense_compress_idx",
            // shape: [hiddenSize]
            "in_post_attention_norm_weight", "in_post_attn_norm_bias", "in_post_attn_norm_new_weight",
            "in_post_attn_norm_new_bias",
            // Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
            // No pack: (Gate) shape: [intermediateSizePerRank, hiddenSize]
            "in_mlp_gate_up_weight", "in_mlp_bias_0", "in_mlp_descale_0", "in_mlp_offset_0", "in_mlp_scale_0",
            "in_mlp_compress_idx_0",
            // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
            "in_mlp_weight_1", "in_mlp_bias_1", "in_mlp_descale_1", "in_mlp_offset_1", "in_mlp_scale_1",
            "in_mlp_compress_idx_1",
            // shape: [hiddenSize, intermediateSizePerRank]
            "in_mlp_down_weight", "in_mlp_down_bias", "in_mlp_down_descale", "in_mlp_down_offset",
            "in_mlp_down_scale", "in_mlp_down_compress_idx"}},
        {"default", {
            "in_hidden_states",
            "in_attention_mask", "in_position_id", "in_cos_emb", "in_sin_emb", "in_seq_len",
            "in_place_holder", "in_past_key", "in_past_value"}},
        };
    return llamaLayerInTensorCandiadates;
}

std::map<std::string, std::vector<std::string>> GetLlamaLayerIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> llamaLayerIntermediateTensorCandiadates = {
        {"default", {"intermediate_attention_out", "intermediate_mlp_out"}},
    };
    return llamaLayerIntermediateTensorCandiadates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto llamaLayerInTensorCandiadates = GetLlamaLayerInTensorCandidates();
    auto llamaLayerIntermediateTensorCandiadates = GetLlamaLayerIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out_decoder_layer", "out_present_key", "out_present_value"};

    // 添加默认的Tensor
    atb_speed::common::AddTensorToList(llamaLayerInTensorCandiadates, "default_weight", inTensorList);
    atb_speed::common::AddTensorToList(llamaLayerInTensorCandiadates, "default", inTensorList);
    atb_speed::common::AddTensorToList(llamaLayerIntermediateTensorCandiadates, "default", intermediateTensorList);

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

int64_t AddAttention(atb::Node &attentionNode, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb_speed::common::AttentionParam attentionParam;
    attentionParam.normEps = param.rmsNormEps;
    attentionParam.numHiddenLayers = param.numHiddenLayers;
    attentionParam.numAttentionHeads = param.numAttentionHeadsPerRank;
    attentionParam.hiddenSize = param.hiddenSize;
    attentionParam.isPrefill = param.isPrefill;
    attentionParam.numKeyValueHeads = param.numKeyValueHeadsPerRank;
    attentionParam.isGQA = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    attentionParam.seqLength = param.seqLength;
    attentionParam.isQuant = param.isQuant;
    
    atb_speed::common::AttentionEdge(attentionParam, &attentionNode.operation);
    std::vector<std::string> attentionInTensorNames = {};
    if (attentionParam.isQuant) {
        attentionInTensorNames = {"in_hidden_states", "in_input_norm_weight",
                                  "in_qkv_weight", "in_qkv_scale_0", "in_qkv_offset_0",
                                  "in_qkv_descale_0", "in_qkv_bias_0", "in_attention_out_weight",
                                  "in_qkv_dense_scale", "in_qkv_dense_offset", "in_qkv_dense_descale",
                                  "in_qkv_dense_bias", "in_attention_mask", "in_position_id",
                                  "in_cos_emb", "in_sin_emb", "in_seq_len", "in_place_holder",
                                  "in_past_key", "in_past_value"};
    } else {
        attentionInTensorNames = {"in_hidden_states", "in_input_norm_weight",
                                  "in_qkv_weight", "in_attention_out_weight", "in_mlp_gate_up_weight",
                                  "in_mlp_down_weight", "in_post_attention_norm_weight", "in_attention_mask",
                                  "in_position_id", "in_cos_emb", "in_sin_emb", "in_seq_len",
                                  "in_place_holder", "in_past_key", "in_past_value"};
    }
    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attentionInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out", \
        "out_present_key", "out_present_value"});
    return atb::NO_ERROR;
}

void SetMlpParam(atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam, const DecoderLayerParam &param)
{
    mlpParam.isBF16 = param.isBF16;
    mlpParam.layerLinearQuantType = param.linearQuantType;
    mlpParam.layerLinearTransposeType = param.linearTransposeType;
    mlpParam.packQuantType = param.packQuantType.at(1);
    mlpParam.quantGroupSize = param.quantGroupSize;
    mlpParam.isEdgeHardware = param.isEdgeHardware;
    // w2_w1(gate_up)
    mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_PACK;
    mlpParam.enableAddNorm = param.enableAddNorm;
    atb::infer::RmsNormParam mlpRmsNormParam;
    mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormParam.normParam.epsilon = param.rmsNormEps;
    mlpParam.normParamType = mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    if (param.enableAddNorm) {
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormQuantParam.preNormParam.epsilon = param.rmsNormEps;
        mlpRmsNormQuantParam.preNormParam.quantType = atb::infer::QUANT_INT8;
    } else {
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
        mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    }
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    mlpParam.supportLcoc = param.supportLcoc;
    if (param.supportSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
    } else {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    }
}

int64_t AddMlp(atb::Node &mlpParallelNode, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    SetMlpParam(mlpParam, param);
    if (param.supportSwiGLU) {
        CHECK_OPERATION_STATUS_RETURN(MlpSwiGLU(mlpParam, &mlpParallelNode.operation));
    } else {
        CHECK_OPERATION_STATUS_RETURN(Mlp(mlpParam, &mlpParallelNode.operation));
    }
  
    std::vector<std::string> mlpInTensorNames = {"intermediate_attention_out",
                                                 "in_post_attention_norm_weight",
                                                 "in_post_attn_norm_bias",
                                                 "in_post_attn_norm_new_weight",
                                                 "in_post_attn_norm_new_bias",
                                                 "in_mlp_gate_up_weight",
                                                 "in_mlp_scale_0",
                                                 "in_mlp_offset_0",
                                                 "in_mlp_descale_0",
                                                 "in_mlp_bias_0",
                                                 "in_mlp_compress_idx_0",
                                                 "in_mlp_weight_1",
                                                 "in_mlp_scale_1",
                                                 "in_mlp_offset_1",
                                                 "in_mlp_descale_1",
                                                 "in_mlp_bias_1",
                                                 "in_mlp_compress_idx_1",
                                                 "in_mlp_down_weight",
                                                 "in_mlp_down_scale",
                                                 "in_mlp_down_offset",
                                                 "in_mlp_down_descale",
                                                 "in_mlp_down_bias",
                                                 "in_mlp_down_compress_idx"};
 
    std::vector<std::string> mlpOutTensorName = {"intermediate_mlp_out"};
    mlpParallelNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpInTensorNames);
    mlpParallelNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpOutTensorName);
    return atb::NO_ERROR;
}

void SetInferShapeFunc(const DecoderLayerParam& param, atb::GraphParam& opGraph,
                       const std::map<std::string, uint32_t>& tensorMap)
{
    if (param.numAttentionHeadsPerRank == 0) {
        std::stringstream ss;
        ss << "Cannot be devided by zero. Param numAttentionHeadsPerRank is zero!" << std::endl;
        throw std::runtime_error(ss.str());
    }
    const uint64_t headDim = param.hiddenSize / param.numAttentionHeadsPerRank;
    const uint64_t RESULT_DIM_4 = 4;

    if (param.isPrefill) {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
            uint32_t inHiddenStatesIdx = atb_speed::common::GetTensorIdx(tensorMap, "in_hidden_states");
            outTensorDescs.at(0) = inTensorDescs.at(inHiddenStatesIdx);
            outTensorDescs.at(1) = inTensorDescs.at(0); // bs
            outTensorDescs.at(1).shape.dimNum = RESULT_DIM_4; // 四维
            outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // bs
            outTensorDescs.at(1).shape.dims[1] = param.numKeyValueHeadsPerRank; // KeyValueHeads
            outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // seq of outtensor 2
            outTensorDescs.at(1).shape.dims[3] = headDim; // dim in dims[3]

            outTensorDescs.at(2) = inTensorDescs.at(0); // outtensor 2
            outTensorDescs.at(2).shape.dimNum = RESULT_DIM_4; // bs of outtensor 2
            outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // bs dims[0] of outtensor 2
            outTensorDescs.at(2).shape.dims[1] = param.numKeyValueHeadsPerRank; // KeyValueHeads of outtensor 2
            outTensorDescs.at(2).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // seq of outtensor 2
            outTensorDescs.at(2).shape.dims[3] = headDim; // dim in dims[3] of outtensor 2
            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
            uint32_t inHiddenStatesIdx = atb_speed::common::GetTensorIdx(tensorMap, "in_hidden_states");
            outTensorDescs.at(0) = inTensorDescs.at(inHiddenStatesIdx);
            // in_past_key of outtensor 1
            outTensorDescs.at(1) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_past_key"));
            outTensorDescs.at(1).shape.dims[2] += 1; // dims[2] + 1 of outtensor 1
            // in_past_value of outtensor 2
            outTensorDescs.at(2) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_past_value"));
            outTensorDescs.at(2).shape.dims[2] += 1; // dim[2] + 1 of outtensor 2
            return atb::NO_ERROR;
        };
    }
}

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph internalTensorNum " << opGraph.internalTensorNum);

    atb::Node attentionNode;
    atb::Node selfResidualAddNode;
    atb::Node mlpParallelNode;
    atb::Node mlpResidualAddNode;

    CHECK_OPERATION_STATUS_RETURN(AddAttention(attentionNode, param, tensorMap));
    opGraph.nodes.push_back(attentionNode);

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    if (!param.enableAddNorm) {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
        selfResidualAddNode.inTensorIds = \
            atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states", "intermediate_attention_out"});
        selfResidualAddNode.outTensorIds =
            atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
        opGraph.nodes.push_back(selfResidualAddNode);
    }

    CHECK_OPERATION_STATUS_RETURN(AddMlp(mlpParallelNode, param, tensorMap));
    opGraph.nodes.push_back(mlpParallelNode);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    mlpResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out", "intermediate_mlp_out"});
    mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out_decoder_layer"});
    opGraph.nodes.push_back(mlpResidualAddNode);

    SetInferShapeFunc(param, opGraph, tensorMap);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}
} // namespace llama
} // namespace atb_speed