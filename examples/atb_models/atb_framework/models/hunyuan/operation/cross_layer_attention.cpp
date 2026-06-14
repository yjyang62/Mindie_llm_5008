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
#include "models/hunyuan/operation/cross_layer_attention.h"

#include "operations/fusion/utils.h"
#include "operations/aclnn/ops/repeat_operation.h"
#include "operations/fusion/infer_shape_functions.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetClaInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>>  claInTensorCandidates = {
        {"default", {
            "in_input",
            "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_weight_bias",
            "q_norm_weight", "q_norm_bias", "q_norm_new_weight", "q_norm_new_weight_bias",
            "k_norm_weight", "k_norm_bias", "k_norm_new_weight", "k_norm_new_weight_bias",
            "in_qkv_proj_weight", "in_qkv_proj_bias", "in_qkv_proj_descale", "in_qkv_proj_offset", "in_qkv_proj_scale",
            "in_qkv_proj_compress_idx",
            "in_attn_out_weight", "in_attn_out_bias", "in_attn_out_descale", "in_attn_out_offset",  "in_attn_out_scale",
            "in_attn_out_compress_idx",
            "in_cos_embed", "in_sin_embed", "in_seq_len", "in_k_cache", "in_v_cache", "in_attention_mask",
            "in_token_offset", "in_layer_id", "in_block_tables",
            "in_slots_in_pa_or_logn_in_fa"}
        },
        {"cross", {"in_k_cross", "in_v_cross"}}
    };
    return claInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetClaIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> claIntermediateTensorCandidates = {
        {"default",
            {
                "in_input_norm",
                "intermediate_q", "intermediate_k", "intermediate_self_attention"
            }
        },
        {"self", {"latent_qkv"}},
        {"cross", {"placeholder_k"}}
    };
    return claIntermediateTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetClaOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> claOutTensorCandidates = {
        {"default", {"out"}},
        {"self", {"out_k_cross", "out_v_cross"}}
    };
    return claOutTensorCandidates;
}

template <typename NormParamType>
std::map<std::string, uint32_t> ClaConstructTensorMap(const CrossLayerAttentionParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> outTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    auto claInTensorCandidates = GetClaInTensorCandidates();
    auto claIntermediateTensorCandidates = GetClaIntermediateTensorCandidates();
    auto claOutTensorCandidates = GetClaOutTensorCandidates();

    // 添加默认的Tensor
    AddTensorToList(claInTensorCandidates, "default", inTensorList);
    AddTensorToList(claIntermediateTensorCandidates, "default", intermediateTensorList);
    AddTensorToList(claOutTensorCandidates, "default", outTensorList);

    if (param.isCrossLayer) {
        AddTensorToList(claInTensorCandidates, "cross", inTensorList);
        AddTensorToList(claIntermediateTensorCandidates, "cross", intermediateTensorList);
    } else {
        AddTensorToList(claOutTensorCandidates, "self", outTensorList);
        AddTensorToList(claIntermediateTensorCandidates, "self", intermediateTensorList);
    }
   
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

template <typename NormParamType>
bool UseNormQuant(const CrossLayerAttentionParam<NormParamType> &param, uint64_t linearIndex)
{
    LinearQuantType quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[linearIndex], true);
    if (quantType == LinearQuantType::LINEAR_W8A8_DEQUANT ||
        quantType == LinearQuantType::LINEAR_W8A8_SC_DEQUANT) {
        return true;
    } else {
        return false;
    }
}

template <typename NormParamType>
atb::Status ClaAddPreNormNode(const CrossLayerAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node normNode;

    if (UseNormQuant(param, QKV_PROJ_LINEAR_INDEX)) {  // W8A8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &normNode.operation));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_input"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_norm_weight"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_norm_bias"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_qkv_proj_scale"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_qkv_proj_offset"));
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &normNode.operation));
        normNode.inTensorIds = {GetTensorIdx(tensorMap, "in_input"), GetTensorIdx(tensorMap, "in_norm_weight")};
    }
    normNode.outTensorIds = {GetTensorIdx(tensorMap, "in_input_norm")};
    opGraph.nodes.push_back(normNode);
    ATB_SPEED_LOG_DEBUG("Attention PreNorm calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status ClaAddQkvProjNode(const CrossLayerAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qkvProjNode;
    atb_speed::common::FusionLinearParam qkvProjNodeParam;
    qkvProjNodeParam.isBF16 = param.isBF16;
    qkvProjNodeParam.hasBias = param.selfAttnHasBias;
    qkvProjNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[QKV_PROJ_LINEAR_INDEX], true);
    qkvProjNodeParam.quantGroupSize = param.quantGroupSize;
    qkvProjNodeParam.transposeType = param.attnLinearTransposeType[QKV_PROJ_LINEAR_INDEX];
    qkvProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input_norm"),
        GetTensorIdx(tensorMap, "in_qkv_proj_weight"),
        GetTensorIdx(tensorMap, "in_qkv_proj_scale"),
        GetTensorIdx(tensorMap, "in_qkv_proj_offset"),
        GetTensorIdx(tensorMap, "in_qkv_proj_descale"),
        GetTensorIdx(tensorMap, "in_qkv_proj_bias"),
        GetTensorIdx(tensorMap, "in_qkv_proj_compress_idx"),
    };
    qkvProjNode.outTensorIds = {GetTensorIdx(tensorMap, param.isCrossLayer ? "intermediate_q" : "latent_qkv")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qkvProjNodeParam, &qkvProjNode.operation));
    opGraph.nodes.push_back(qkvProjNode);
    ATB_SPEED_LOG_DEBUG("Cla proj_Q(kv) calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSplitQkvNode(const CrossLayerAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitKNode;
    atb::infer::SplitParam splitKParam = {1, 3, // 1: split dim; 3: split num
        {param.headDim * param.selfAttentionParam.headNum,
         param.headDim * param.selfAttentionParam.kvHeadNum,
         param.headDim * param.selfAttentionParam.kvHeadNum}};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
    splitKNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_qkv")};
    splitKNode.outTensorIds = {GetTensorIdxList(tensorMap, {"intermediate_q", "out_k_cross", "out_v_cross"})};
    opGraph.nodes.push_back(splitKNode);
    ATB_SPEED_LOG_DEBUG("CLA AddSplitQkvNode calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status ClaAddRopeNode(const CrossLayerAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node ropeNode;
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.ropeParam.rotaryCoeff;
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, param.isCrossLayer ? "in_k_cross" : "out_k_cross"),
        GetTensorIdx(tensorMap, "in_cos_embed"),
        GetTensorIdx(tensorMap, "in_sin_embed"),
        GetTensorIdx(tensorMap, "in_seq_len")
    };
    ropeNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, param.isCrossLayer ? "placeholder_k" : "out_k_cross"),
    };
    opGraph.nodes.push_back(ropeNode);
    ATB_SPEED_LOG_DEBUG("CLA rope calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status ClaAddCrossLayerQKNormNode(const CrossLayerAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qNormNode;
    qNormNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "q_norm_weight")
    };
    qNormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &qNormNode.operation));
    qNormNode.inTensorReshapeFuncs.resize(qNormNode.inTensorIds.size());
    qNormNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnsqueezeHeadNumHeadDim(oldShape, newShape, param.selfAttentionParam.headNum, param.headDim);
    };
    opGraph.nodes.push_back(qNormNode);
    ATB_SPEED_LOG_DEBUG("CLA q norm calculation success");

    atb::Node kNormNode;
    kNormNode.inTensorIds = {
        GetTensorIdx(tensorMap, param.isCrossLayer ? "in_k_cross" : "out_k_cross"),
        GetTensorIdx(tensorMap, "k_norm_weight")
    };
    kNormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k")};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &kNormNode.operation));
    kNormNode.inTensorReshapeFuncs.resize(kNormNode.inTensorIds.size());
    kNormNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnsqueezeHeadNumHeadDim(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.headDim);
    };
    opGraph.nodes.push_back(kNormNode);
    ATB_SPEED_LOG_DEBUG("CLA k norm calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status ClaAddSelfOutLinearParallelNode(const CrossLayerAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfOutLinearParallelNode;
    atb_speed::common::LinearParallelParam selfOutLinearParam;
    selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
    selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    selfOutLinearParam.fusionLinearParam.hasBias = param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
    selfOutLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[O_PROJ_LINEAR_INDEX], false);
    selfOutLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
    selfOutLinearParam.fusionLinearParam.transposeType = param.attnLinearTransposeType[O_PROJ_LINEAR_INDEX];
    selfOutLinearParam.supportLcoc = param.supportLcoc;
    CHECK_OPERATION_STATUS_RETURN(LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation));
    selfOutLinearParallelNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_attention"),
        GetTensorIdx(tensorMap, "in_attn_out_weight"),
        GetTensorIdx(tensorMap, "in_attn_out_scale"),
        GetTensorIdx(tensorMap, "in_attn_out_offset"),
        GetTensorIdx(tensorMap, "in_attn_out_descale"),
        GetTensorIdx(tensorMap, "in_attn_out_bias"),
        GetTensorIdx(tensorMap, "in_attn_out_compress_idx"),
    };
    selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());
    if (!param.isFA) {
        selfOutLinearParallelNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            SqueezeHeadNumHeadDim(oldShape, newShape);
        };
    }
    selfOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(selfOutLinearParallelNode);
    ATB_SPEED_LOG_DEBUG("CLA o_proj calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status ClaAddCacheNode(const CrossLayerAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node reshapeAndCacheNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.reshapeCacheParm, &reshapeAndCacheNode.operation));
    if (param.isCrossLayer) {
        reshapeAndCacheNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_k"),
            GetTensorIdx(tensorMap, "in_k_cache"),
            GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
        };
        reshapeAndCacheNode.outTensorIds = {
            GetTensorIdx(tensorMap, "in_k_cache"),
        };
    } else {
        reshapeAndCacheNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_k"),
            GetTensorIdx(tensorMap, "out_v_cross"),
            GetTensorIdx(tensorMap, "in_k_cache"),
            GetTensorIdx(tensorMap, "in_v_cache"),
            GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
        };
        reshapeAndCacheNode.outTensorIds = {
            GetTensorIdx(tensorMap, "in_k_cache"),
            GetTensorIdx(tensorMap, "in_v_cache"),
        };
        reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
        reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnsqueezeHeadNumHeadDim(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.headDim);
        };
    }
    opGraph.nodes.push_back(reshapeAndCacheNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status CrossLayerAttention(const CrossLayerAttentionParam<NormParamType> &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchSizePtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = param.isCrossLayer ? "CrossLayerAttention" : "SelfLayerAttention";
    std::map<std::string, uint32_t> tensorMap = ClaConstructTensorMap(param,
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum " << opGraph.internalTensorNum);
    // PreNorm Node
    CHECK_OPERATION_STATUS_RETURN(ClaAddPreNormNode(param, opGraph, tensorMap));
    // KV_proj Node
    CHECK_OPERATION_STATUS_RETURN(ClaAddQkvProjNode(param, opGraph, tensorMap));
    if (!param.isCrossLayer) {
        CHECK_OPERATION_STATUS_RETURN(AddSplitQkvNode(param, opGraph, tensorMap));
    }
    // Rope Node
    if (param.rotaryType != RotaryType::NO_ROTARY) {
        CHECK_OPERATION_STATUS_RETURN(ClaAddRopeNode(param, opGraph, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(ClaAddCrossLayerQKNormNode(param, opGraph, tensorMap));
    // SelfAttention Node
    CHECK_OPERATION_STATUS_RETURN(AddCrossAttention(opGraph, param, tensorMap));
    // Dense Node
    CHECK_OPERATION_STATUS_RETURN(ClaAddSelfOutLinearParallelNode(param, opGraph, tensorMap));

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (!param.isCrossLayer) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
            outTensorDescs.at(1).shape.dims[inTensorDescs.at(0).shape.dimNum - 1] =
                param.selfAttentionParam.kvHeadNum * param.headDim;
            outTensorDescs.at(2) = outTensorDescs.at(1); // 2: out tensor index
        }
        return atb::NO_ERROR;
    };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t AddCrossAttention(
    atb::GraphParam &opGraph, const CrossLayerAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    // ReshapeAndCache Node
    CHECK_OPERATION_STATUS_RETURN(ClaAddCacheNode(param, opGraph, tensorMap));

    // CrossLayerAttentionNode
    atb::Node selfAttentionNode;
    if (param.isPrefill) {  // PA Prefill
        CHECK_OPERATION_STATUS_RETURN(ClaConstructPaEncoderNode(selfAttentionNode, param, tensorMap));
    } else {  // PA Decode
        CHECK_OPERATION_STATUS_RETURN(ClaConstructPaDecoderNode(selfAttentionNode, param, tensorMap));
    }

    selfAttentionNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention")};
    opGraph.nodes.push_back(selfAttentionNode);
    ATB_SPEED_LOG_DEBUG("CLA self-attention calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t ClaConstructPaEncoderNode(
    atb::Node &selfAttentionNode, const CrossLayerAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.selfAttentionParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, param.isCrossLayer ? "in_v_cross" : "out_v_cross")
    };
    if (param.selfAttentionParam.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
    }
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) { // 2:tensor idx
        UnsqueezeHeadNumHeadDim(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.headDim);
    };
    ATB_SPEED_LOG_DEBUG("CLA PA encoder calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
int64_t ClaConstructPaDecoderNode(
    atb::Node &selfAttentionNode, const CrossLayerAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    // 输出[num_tokens, N, D] [num_block,block_size,N,D]
    // 输出[num_tokens, num_head, head_size]
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.pageAttentionParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_v_cache"),
        GetTensorIdx(tensorMap, "in_block_tables"),
        GetTensorIdx(tensorMap, "in_seq_len"),
    };
    if (param.pageAttentionParam.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
    }
    ATB_SPEED_LOG_DEBUG("CLA PA decoder calculation success");
    return atb::NO_ERROR;
}


template int64_t AddCrossAttention(
    atb::GraphParam &opGraph, const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ClaConstructPaEncoderNode(
    atb::Node &selfAttentionNode, const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ClaConstructPaDecoderNode(
    atb::Node &selfAttentionNode, const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);

template std::map<std::string, uint32_t> ClaConstructTensorMap(
    const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template atb::Status ClaAddPreNormNode(const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status ClaAddQkvProjNode(const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status ClaAddRopeNode(const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status ClaAddCrossLayerQKNormNode(const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status ClaAddSelfOutLinearParallelNode(const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status ClaAddCacheNode(const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status CrossLayerAttention(
    const CrossLayerAttentionParam<atb::infer::RmsNormParam> &param,
    atb::Operation **operation);

} // namespace common
} // namespace atb_speed