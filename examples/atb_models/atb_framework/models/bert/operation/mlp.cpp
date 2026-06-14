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
#include "atb_speed/log.h"
#include "operations/aclnn/ops/gelu_operation.h"
#include "operations/aclnn/ops/matmul_operation.h"
#include "operations/aclnn/ops/layer_norm_operation.h"
#include "models/bert/operation/mlp.h"


namespace atb_speed::bert {

    enum MlpTensorId : int {
        // input tensors
        IN_HIDDENSTATES = 0,
        IN_INTERLINEAR_WEIGHT,
        IN_INTERLINEAR_BIAS,
        IN_OUTLINEAR_WEIGHT,
        IN_OUTLINEAR_BIAS,
        IN_NORM_WEIGHT,
        IN_NORM_BIAS,
        IN_TENSOR_END,
        // output tensors
        OUT_FEEDFORWARD_RESULT = IN_TENSOR_END,
        OUT_TENSOR_END,
        // intermediate tensors
        INTERMEDIATE_INTERLINEAR_OUT = OUT_TENSOR_END,
        INTERMEDIATE_ACT_OUT,
        INTERMEDIATE_OUTLINEAR_OUT,
        INTERMEDAITE_ADD_OUT,
        INTERMEDAITE_TENSOR_END,
    };

    static const uint64_t IN_TENSOR_COUNT = IN_TENSOR_END;
    static const uint64_t OUT_TENSOR_COUNT = OUT_TENSOR_END - IN_TENSOR_END;
    static const uint64_t INTERNAL_TENSOR_COUNT = INTERMEDAITE_TENSOR_END - OUT_TENSOR_END;
    static const uint64_t NODE_COUNT = 5;

    atb::Status AddInterLinear(atb::GraphParam &opGraph, const MlpParam &param, size_t nodeId)
    {
        // intermediate, dense Linear
        auto &interLinearNode = opGraph.nodes.at(nodeId);
        interLinearNode.inTensorIds = { IN_HIDDENSTATES, IN_INTERLINEAR_WEIGHT, IN_INTERLINEAR_BIAS };
        interLinearNode.outTensorIds = { INTERMEDIATE_INTERLINEAR_OUT };
        if (param.enableAclNNMatmul) {
            atb_speed::common::AclNNMatmulParam interLinearParam;
            interLinearParam.transposeB = true;
            interLinearParam.hasBias = true;
            interLinearNode.operation = new atb_speed::common::MatmulOperation("interLinearNode", interLinearParam);
        } else {
            atb::infer::LinearParam interLinearParam;
            interLinearParam.hasBias = true;
            interLinearParam.transposeA = false;
            interLinearParam.transposeB = true;
            CREATE_OPERATION(interLinearParam, &interLinearNode.operation);
        }
        return atb::NO_ERROR;
    }

    atb::Status AddGelu(atb::GraphParam &opGraph, const MlpParam &param, size_t nodeId)
    {
        // intermediate_act_fn, Gelu
        auto &interActNode = opGraph.nodes.at(nodeId);
        interActNode.inTensorIds = { INTERMEDIATE_INTERLINEAR_OUT };
        interActNode.outTensorIds = { INTERMEDIATE_ACT_OUT };
        if (param.enableFasterGelu) {
            atb::infer::ActivationParam interActParam;
            interActParam.activationType = atb::infer::ActivationType::ACTIVATION_FASTER_GELU_FORWARD;
            CREATE_OPERATION(interActParam, &interActNode.operation);
        } else {
            atb_speed::common::AclNNGeluParam interActParam;
            interActParam.geluApproximate = param.geluApproximate;
            interActNode.operation = new atb_speed::common::GeluOperation("interActNode", interActParam);
        }
        return atb::NO_ERROR;
    }

    atb::Status AddOutLinear(atb::GraphParam &opGraph, const MlpParam &param, size_t nodeId)
    {
        // output, dense Linear
        auto &outLinearNode = opGraph.nodes.at(nodeId);
        outLinearNode.inTensorIds = { INTERMEDIATE_ACT_OUT, IN_OUTLINEAR_WEIGHT, IN_OUTLINEAR_BIAS };
        outLinearNode.outTensorIds = { INTERMEDIATE_OUTLINEAR_OUT };
        if (param.enableAclNNMatmul) {
            atb_speed::common::AclNNMatmulParam outLinearParam;
            outLinearParam.transposeB = true;
            outLinearParam.hasBias = true;
            outLinearNode.operation = new atb_speed::common::MatmulOperation("outLinearNode", outLinearParam);
        } else {
            atb::infer::LinearParam outLinearParam;
            outLinearParam.hasBias = true;
            outLinearParam.transposeA = false;
            outLinearParam.transposeB = true;
            CREATE_OPERATION(outLinearParam, &outLinearNode.operation);
        }
        return atb::NO_ERROR;
    }

    atb::Status Mlp(const MlpParam &param, atb::Operation **operation)
    {
        ATB_SPEED_LOG_DEBUG(__func__ << " called");
        atb::GraphParam opGraph;
        opGraph.name = "Mlp";
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.outTensorNum = OUT_TENSOR_COUNT;
        opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
        opGraph.nodes.resize(NODE_COUNT);

        size_t nodeId = 0;

        CHECK_OPERATION_STATUS_RETURN(AddInterLinear(opGraph, param, nodeId++));
        CHECK_OPERATION_STATUS_RETURN(AddGelu(opGraph, param, nodeId++));
        CHECK_OPERATION_STATUS_RETURN(AddOutLinear(opGraph, param, nodeId++));

        // Add
        auto &outAddNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;
        CREATE_OPERATION(addParam, &outAddNode.operation);
        outAddNode.inTensorIds = { INTERMEDIATE_OUTLINEAR_OUT, IN_HIDDENSTATES };
        outAddNode.outTensorIds = { INTERMEDAITE_ADD_OUT };

        // LayerNorm
        auto &outNormNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::AclNNLayerNormParam outNormParam;
        outNormParam.layerNormEps = param.layerNormEps;
        outNormParam.beginNormAxis = param.beginNormAxis;
        outNormParam.layerNormImplMode = param.layerNormImplMode;
        outNormNode.operation = new atb_speed::common::LayerNormOperation("outNormNode", outNormParam);
        outNormNode.inTensorIds = { INTERMEDAITE_ADD_OUT, IN_NORM_WEIGHT, IN_NORM_BIAS };
        outNormNode.outTensorIds = { OUT_FEEDFORWARD_RESULT };

        CREATE_OPERATION(opGraph, operation);
        return atb::NO_ERROR;
    }

}  // namespace atb_speed::berts
