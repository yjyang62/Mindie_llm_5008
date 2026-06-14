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
#include "models/bert/operation/attention.h"
#include "models/bert/operation/mlp.h"
#include "models/bert/layer/encoder_layer.h"


namespace atb_speed::bert {

    enum EncoderLayerTensorId : int {
        // input tensors
        IN_HIDDENSTATES = 0,
        IN_SELFATTENTIONQLINEAR_WEIGHT,
        IN_SELFATTENTIONQLINEAR_BIAS,
        IN_SELFATTENTIONKLINEAR_WEIGHT,
        IN_SELFATTENTIONKLINEAR_BIAS,
        IN_SELFATTENTIONVLINEAR_WEIGHT,
        IN_SELFATTENTIONVLINEAR_BIAS,
        IN_SELFOUTLINER_WEIGHT,
        IN_SELFOUTLINEAR_BIAS,
        IN_SELFOUTNORM_WEIGHT,
        IN_SELFOUTNORM_BIAS,
        IN_INTERLINEAR_WEIGHT,
        IN_INTERLINEAR_BIAS,
        IN_OUTLINEAR_WEIGHT,
        IN_OUTLINEAR_BIAS,
        IN_OUTNORM_WEIGHT,
        IN_OUTNORM_BIAS,
        // layer inputs
        IN_ATTENTIONMASK,
        IN_BLOCK_TABLES,
        IN_PASTKEY,
        IN_PASTVALUE,
        IN_TOKENOFFSET,
        IN_SEQLEN,
        IN_LAYERID,
        // output tensors
        OUT_LAYER_RESULT,
        // intermediate tensors
        INTERMEDIATE_ATTENTION_OUT
    };

    static const uint64_t IN_TENSOR_COUNT = 24;
    static const uint64_t OUT_TENSOR_COUNT = 1;
    static const uint64_t INTERNAL_TENSOR_COUNT = 1;

    int64_t Attention(atb::GraphParam &opGraph, const EncoderLayerParam &param)
    {
        atb::Node attentionNode;

        atb_speed::bert::AttentionParam attentionParam;
        attentionParam.dk = param.dk;
        attentionParam.headNum = param.headNum;
        attentionParam.layerNormEps = param.layerNormEps;
        attentionParam.layerNormImplMode = param.layerNormImplMode;
        attentionParam.enableAclNNAttn = param.enableAclNNAttn;
        attentionParam.enableAclNNMatmul = param.enableAclNNMatmul;
        atb_speed::bert::Attention(attentionParam, &attentionNode.operation);
        attentionNode.inTensorIds = {
            IN_HIDDENSTATES,
            IN_SELFATTENTIONQLINEAR_WEIGHT,
            IN_SELFATTENTIONQLINEAR_BIAS,
            IN_SELFATTENTIONKLINEAR_WEIGHT,
            IN_SELFATTENTIONKLINEAR_BIAS,
            IN_SELFATTENTIONVLINEAR_WEIGHT,
            IN_SELFATTENTIONVLINEAR_BIAS,
            IN_SELFOUTLINER_WEIGHT,
            IN_SELFOUTLINEAR_BIAS,
            IN_SELFOUTNORM_WEIGHT,
            IN_SELFOUTNORM_BIAS,
            IN_ATTENTIONMASK,
            IN_BLOCK_TABLES,
            IN_PASTKEY,
            IN_PASTVALUE,
            IN_TOKENOFFSET,
            IN_SEQLEN,
            IN_LAYERID,
        };
        attentionNode.outTensorIds = { INTERMEDIATE_ATTENTION_OUT };
        opGraph.nodes.push_back(attentionNode);

        return atb::NO_ERROR;
    }

    int64_t Mlp(atb::GraphParam &opGraph, const EncoderLayerParam &param)
    {
        atb::Node mlpNode;

        atb_speed::bert::MlpParam mlpParam;
        mlpParam.geluApproximate = param.geluApproximate;
        mlpParam.layerNormEps = param.layerNormEps;
        mlpParam.layerNormImplMode = param.layerNormImplMode;
        mlpParam.enableFasterGelu = param.enableFasterGelu;
        mlpParam.enableAclNNMatmul = param.enableAclNNMatmul;
        atb_speed::bert::Mlp(mlpParam, &mlpNode.operation);
        mlpNode.inTensorIds = {
            INTERMEDIATE_ATTENTION_OUT,
            IN_INTERLINEAR_WEIGHT,
            IN_INTERLINEAR_BIAS,
            IN_OUTLINEAR_WEIGHT,
            IN_OUTLINEAR_BIAS,
            IN_OUTNORM_WEIGHT,
            IN_OUTNORM_BIAS,
        };
        mlpNode.outTensorIds = { OUT_LAYER_RESULT };
        opGraph.nodes.push_back(mlpNode);

        return atb::NO_ERROR;
    }

    atb::Status EncoderLayer(const EncoderLayerParam &param, atb::Operation **operation)
    {
        ATB_SPEED_LOG_DEBUG(__func__ << " called, headNum: " << param.headNum);
        atb::GraphParam opGraph;
        opGraph.name = "EncoderLayer";
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.outTensorNum = OUT_TENSOR_COUNT;
        opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;

        CHECK_OPERATION_STATUS_RETURN(Attention(opGraph, param));
        CHECK_OPERATION_STATUS_RETURN(Mlp(opGraph, param));

        opGraph.inferShapeFunc = [=](
            const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs
        ) -> atb::ErrorType {
            outTensorDescs.at(0) = inTensorDescs.at(0);
            return atb::NO_ERROR;
        };

        CREATE_OPERATION(opGraph, operation);
        return atb::NO_ERROR;
    }

}  // namespace atb_speed::bert
