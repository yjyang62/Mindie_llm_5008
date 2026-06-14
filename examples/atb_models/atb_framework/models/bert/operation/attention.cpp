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
#include <cmath>
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/ops/matmul_operation.h"
#include "operations/aclnn/ops/attn_operation.h"
#include "operations/aclnn/ops/layer_norm_operation.h"
#include "models/bert/operation/attention.h"

namespace atb_speed::bert {

    enum AttentionTensorId : int {
        // input tensors
        IN_HIDDENSTATES = 0,
        IN_QLINEAR_WEIGHT,
        IN_QLINEAR_BIAS,
        IN_KLINEAR_WEIGHT,
        IN_KLINEAR_BIAS,
        IN_VLINEAR_WEIGHT,
        IN_VLINEAR_BIAS,
        IN_SELFOUTLINER_WEIGHT,
        IN_SELFOUTLINEAR_BIAS,
        IN_SELFOUTNORM_WEIGHT,
        IN_SELFOUTNORM_BIAS,
        // layer inputs
        IN_ATTENTIONMASK,
        IN_BLOCK_TABLES,
        IN_PASTKEY,
        IN_PASTVALUE,
        IN_TOKENOFFSET,
        IN_SEQLEN,
        IN_LAYERID,
        // output tensors
        OUT_SELF_RESULT,
        // intermediate tensors
        INTERMEDIATE_QLINER_OUT,
        INTERMEDIATE_KLINER_OUT,
        INTERMEDIATE_VLINER_OUT,
        INTERMEDIATE_SELFATTENTION_OUT,
        INTERMEDIATE_OUTLINEAR_OUT,
        INTERMEDIATE_OUTADD_OUT
    };

    static const uint64_t IN_TENSOR_COUNT = 18;
    static const uint64_t OUT_TENSOR_COUNT = 1;
    static const uint64_t INTERNAL_TENSOR_COUNT = 6;
    static const uint64_t SELF_ATTENTION_OUT_INDEX = 0;
    static const uint64_t SELF_ATTENTION_QKV_INPUT_SIZE = 3;
    static const uint64_t SELF_OUT_LINEAR_SIZE = 2;
    static const uint64_t SELF_OUT_ADD_SIZE = 3;
    static const uint64_t HEAD_DIM = 64;
    static const int NUM0 = 0;
    static const int NUM1 = 1;
    static const int NUM2 = 2;

    int64_t CacheTensorReshapePAEncoder(atb::Node &selfAttentionKVCacheNode, const AttentionParam &param)
    {
        if (param.headNum == NUM0) {
            return atb::ERROR_INVALID_PARAM;
        }

        selfAttentionKVCacheNode.inTensorReshapeFuncs.resize(selfAttentionKVCacheNode.inTensorIds.size());
        selfAttentionKVCacheNode.inTensorReshapeFuncs.at(NUM0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = SELF_ATTENTION_QKV_INPUT_SIZE;
            size_t newShapeDimIndex = NUM0;
            size_t oldShapeDimIndex = NUM1;
            newShape.dims[newShapeDimIndex++] =
                CheckIntMulOverFlow(oldShape.dims[oldShapeDimIndex - NUM1], oldShape.dims[oldShapeDimIndex]);
            newShape.dims[newShapeDimIndex++] = param.headNum;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[++oldShapeDimIndex] / param.headNum;
        };
        selfAttentionKVCacheNode.inTensorReshapeFuncs.at(NUM1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = SELF_ATTENTION_QKV_INPUT_SIZE;
            size_t newShapeDimIndex = NUM0;
            size_t oldShapeDimIndex = NUM1;
            newShape.dims[newShapeDimIndex++] =
                CheckIntMulOverFlow(oldShape.dims[oldShapeDimIndex - NUM1], oldShape.dims[oldShapeDimIndex]);
            newShape.dims[newShapeDimIndex++] = param.headNum;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[++oldShapeDimIndex] / param.headNum;
        };
        selfAttentionKVCacheNode.inTensorReshapeFuncs.at(NUM2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = SELF_ATTENTION_QKV_INPUT_SIZE;
            size_t newShapeDimIndex = NUM0;
            size_t oldShapeDimIndex = NUM1;
            newShape.dims[newShapeDimIndex++] =
                CheckIntMulOverFlow(oldShape.dims[oldShapeDimIndex - NUM1], oldShape.dims[oldShapeDimIndex]);
            newShape.dims[newShapeDimIndex++] = param.headNum;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[++oldShapeDimIndex] / param.headNum;
        };

        return atb::NO_ERROR;
    }

    int64_t CacheTensorReshapeSelfAdd(atb::Node &outAddNode)
    {
        int64_t batchSize = NUM0;

        outAddNode.inTensorReshapeFuncs.resize(outAddNode.inTensorIds.size());
        outAddNode.inTensorReshapeFuncs.at(NUM0) = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = SELF_OUT_ADD_SIZE;
            size_t newShapeDimIndex = NUM0;
            size_t oldShapeDimIndex = NUM0;
            batchSize = oldShape.dims[oldShapeDimIndex];
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        };
        outAddNode.inTensorReshapeFuncs.at(NUM1) = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
            if (batchSize == NUM0) {
                return atb::ERROR_INVALID_PARAM;
            }
            newShape.dimNum = SELF_OUT_ADD_SIZE;
            size_t newShapeDimIndex = NUM0;
            size_t oldShapeDimIndex = NUM0;
            newShape.dims[newShapeDimIndex++] = batchSize;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++] / batchSize;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
            return atb::NO_ERROR;
        };

        return atb::NO_ERROR;
    }

    int64_t QKVLinear(atb::GraphParam &opGraph, const AttentionParam &param)
    {
        atb::Node qLinerNode;
        atb::Node kLinerNode;
        atb::Node vLinerNode;
        // QKV Linear
        qLinerNode.inTensorIds = { IN_HIDDENSTATES, IN_QLINEAR_WEIGHT, IN_QLINEAR_BIAS };
        qLinerNode.outTensorIds = { INTERMEDIATE_QLINER_OUT };
        kLinerNode.inTensorIds = { IN_HIDDENSTATES, IN_KLINEAR_WEIGHT, IN_KLINEAR_BIAS };
        kLinerNode.outTensorIds = { INTERMEDIATE_KLINER_OUT };
        vLinerNode.inTensorIds = { IN_HIDDENSTATES, IN_VLINEAR_WEIGHT, IN_VLINEAR_BIAS };
        vLinerNode.outTensorIds = { INTERMEDIATE_VLINER_OUT };
        if (param.enableAclNNMatmul) {
            atb_speed::common::AclNNMatmulParam selfLinearParam;
            selfLinearParam.hasBias = true;
            qLinerNode.operation = new atb_speed::common::MatmulOperation("qLinerNode", selfLinearParam);
            kLinerNode.operation = new atb_speed::common::MatmulOperation("kLinerNode", selfLinearParam);
            vLinerNode.operation = new atb_speed::common::MatmulOperation("vLinerNode", selfLinearParam);
        } else {
            atb::infer::LinearParam selfLinearParam;
            selfLinearParam.hasBias = true;
            CREATE_OPERATION(selfLinearParam, &qLinerNode.operation);
            CREATE_OPERATION(selfLinearParam, &kLinerNode.operation);
            CREATE_OPERATION(selfLinearParam, &vLinerNode.operation);
        }
        opGraph.nodes.push_back(qLinerNode);
        opGraph.nodes.push_back(kLinerNode);
        opGraph.nodes.push_back(vLinerNode);

        return atb::NO_ERROR;
    }

    int64_t SelfAttention(atb::GraphParam &opGraph, const AttentionParam &param)
    {
        atb::Node selfAttentionKVCacheNode;
        // Attention Mask
        selfAttentionKVCacheNode.inTensorIds = {
            INTERMEDIATE_QLINER_OUT,
            INTERMEDIATE_KLINER_OUT,
            INTERMEDIATE_VLINER_OUT,
            IN_ATTENTIONMASK,
            IN_SEQLEN
        };
        selfAttentionKVCacheNode.outTensorIds = { INTERMEDIATE_SELFATTENTION_OUT };
        if (param.enableAclNNAttn) {
            selfAttentionKVCacheNode.inTensorIds.push_back(IN_BLOCK_TABLES);
            atb_speed::common::AclNNAttnParam aclnnIncreAttentionParam;
            aclnnIncreAttentionParam.headDim = HEAD_DIM;
            aclnnIncreAttentionParam.headNum = param.headNum;
            aclnnIncreAttentionParam.kvHeadNum = param.headNum;
            aclnnIncreAttentionParam.hasMask = false;
            aclnnIncreAttentionParam.isPrefill = true;
            aclnnIncreAttentionParam.isFA = true;
            aclnnIncreAttentionParam.hasKVQuant = false;
            aclnnIncreAttentionParam.hasQuantOffset = false;
            selfAttentionKVCacheNode.operation = new atb_speed::common::AttnOperation(
                "selfAttentionKVCacheNode",
                aclnnIncreAttentionParam
            );
        } else {
            atb::infer::SelfAttentionParam selfAttentionParam;
            selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
            selfAttentionParam.headNum = param.headNum;
            selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
            selfAttentionParam.qkScale = static_cast<float>(1.0 / sqrt(param.dk));
            CREATE_OPERATION(selfAttentionParam, &selfAttentionKVCacheNode.operation);
            CHECK_OPERATION_STATUS_RETURN(CacheTensorReshapePAEncoder(selfAttentionKVCacheNode, param));
        }
        opGraph.nodes.push_back(selfAttentionKVCacheNode);

        return atb::NO_ERROR;
    }

    int64_t SelfOutput(atb::GraphParam &opGraph, const AttentionParam &param)
    {
        atb::Node outLinearNode;
        atb::Node outAddNode;
        atb::Node outNormNode;

        // Linear
        if (param.enableAclNNMatmul) {
            atb_speed::common::AclNNMatmulParam outLinearParam;
            outLinearParam.hasBias = true;
            outLinearNode.operation = new atb_speed::common::MatmulOperation("outLinearNode", outLinearParam);
        } else {
            atb::infer::LinearParam outLinearParam;
            outLinearParam.hasBias = true;
            CREATE_OPERATION(outLinearParam, &outLinearNode.operation);
        }

        outLinearNode.inTensorIds = { INTERMEDIATE_SELFATTENTION_OUT, IN_SELFOUTLINER_WEIGHT, IN_SELFOUTLINEAR_BIAS };
        outLinearNode.outTensorIds = { INTERMEDIATE_OUTLINEAR_OUT };
        outLinearNode.inTensorReshapeFuncs.resize(outLinearNode.inTensorIds.size());
        outLinearNode.inTensorReshapeFuncs.at(SELF_ATTENTION_OUT_INDEX) = [=](
            const atb::Dims &oldShape,
            atb::Dims &newShape
        ) {
            // old: [64, 50, 1024], new: [64, 51200]
            if (oldShape.dimNum == NUM2) {
                newShape = oldShape;
            } else if (param.enableAclNNAttn) {
                newShape.dimNum = SELF_OUT_LINEAR_SIZE;
                newShape.dims[NUM0] = CheckIntMulOverFlow(oldShape.dims[NUM0], oldShape.dims[NUM1]);
                newShape.dims[NUM1] = oldShape.dims[NUM2];
            } else {
                newShape.dimNum = SELF_OUT_LINEAR_SIZE;
                newShape.dims[NUM0] = oldShape.dims[NUM0];
                newShape.dims[NUM1] = CheckIntMulOverFlow(oldShape.dims[NUM1], oldShape.dims[NUM2]);
            }
        };
        opGraph.nodes.push_back(outLinearNode);

        // Add
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CREATE_OPERATION(addParam, &outAddNode.operation);
        outAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMEDIATE_OUTLINEAR_OUT };
        outAddNode.outTensorIds = { INTERMEDIATE_OUTADD_OUT };
        CHECK_OPERATION_STATUS_RETURN(CacheTensorReshapeSelfAdd(outAddNode));
        opGraph.nodes.push_back(outAddNode);

        // Layer Norm
        atb_speed::common::AclNNLayerNormParam outNormParam;
        outNormParam.layerNormEps = param.layerNormEps;
        outNormParam.beginNormAxis = param.beginNormAxis;
        outNormParam.layerNormImplMode = param.layerNormImplMode;
        outNormNode.operation = new atb_speed::common::LayerNormOperation("outNormNode", outNormParam);
        outNormNode.inTensorIds = { INTERMEDIATE_OUTADD_OUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS };
        outNormNode.outTensorIds = { OUT_SELF_RESULT };
        opGraph.nodes.push_back(outNormNode);

        return atb::NO_ERROR;
    }

    atb::Status Attention(const AttentionParam &param, atb::Operation **operation)
    {
        ATB_SPEED_LOG_DEBUG(__func__ << " called, headNum: " << param.headNum);
        atb::GraphParam opGraph;
        opGraph.name = "Attention";
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.outTensorNum = OUT_TENSOR_COUNT;
        opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;

        CHECK_OPERATION_STATUS_RETURN(QKVLinear(opGraph, param));
        CHECK_OPERATION_STATUS_RETURN(SelfAttention(opGraph, param));
        CHECK_OPERATION_STATUS_RETURN(SelfOutput(opGraph, param));

        CREATE_OPERATION(opGraph, operation);
        return atb::NO_ERROR;
    }

}  // namespace atb_speed::bert
