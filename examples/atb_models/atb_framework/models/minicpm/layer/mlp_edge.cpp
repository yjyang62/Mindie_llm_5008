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
#include "mlp_edge.h"

#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetQuantMlpInTensorCandidatesLite()
{
    std::map<std::string, std::vector<std::string>> mlpInTensorCandidates = {
        {"default", {
            "in_hiddenstates_id",
            "in_post_attention_norm_weight",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder",
            "in_mlp_gate_up_weight",
            "in_mlp_gate_up_weight_input_scale",
            "in_mlp_gate_up_weight_input_offset",
            "in_mlp_gate_up_weight_deq_scale",
            "in_mlp_gate_up_weight_quant_bias",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder",
            "in_mlp_down_weight",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder",
            "in_place_holder"}
        },
    };
    return mlpInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetMlpIntermediateTensorCandidatesLite()
{
    std::map<std::string, std::vector<std::string>> mlpIntermediateTensorCandidates = {
        {"default", {
            "intermediate_matmul_up_out_id", "intermediate_activation_out_id",
            "intermediate_sigmod_out_id", "intermediate_mul_out_id",
            "intermediate_matmul_out_id", "intermediate_split_out_id",
            "intermediate_hiddenstates_id_quant"}
        },
    };
    return mlpIntermediateTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetMlpOutTensorCandidatesLite()
{
    std::map<std::string, std::vector<std::string>> mlpOutTensorCandidates = {
        {"default", {
            "out_result_id"}
        },
    };
    return mlpOutTensorCandidates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto mlpInTensorCandidates = GetQuantMlpInTensorCandidatesLite();
    auto mlpIntermediateTensorCandidates = GetMlpIntermediateTensorCandidatesLite();
    auto mlpOutTensorCandidates = GetMlpOutTensorCandidatesLite();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {};

    AddTensorToList(mlpInTensorCandidates, "default", inTensorList);
    AddTensorToList(mlpIntermediateTensorCandidates, "default", intermediateTensorList);
    AddTensorToList(mlpOutTensorCandidates, "default", outTensorList);

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

atb::Status QuantGateUpInput(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node quantGateUpNode;
    atb::infer::ElewiseParam quantParam;
    quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CREATE_OPERATION(quantParam, &quantGateUpNode.operation);
    quantGateUpNode.inTensorIds = { GetTensorIdx(tensorMap, "in_hiddenstates_id"),
                                    GetTensorIdx(tensorMap, "in_mlp_gate_up_weight_input_scale"),
                                    GetTensorIdx(tensorMap, "in_mlp_gate_up_weight_input_offset")
    };
    quantGateUpNode.outTensorIds = { GetTensorIdx(tensorMap, "intermediate_hiddenstates_id_quant") };
    opGraph.nodes.push_back(quantGateUpNode);
    return atb::NO_ERROR;
}


atb::Status AddMatmulGateUpV3Node(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node matmulGateUpNode;
    atb::infer::LinearParam matmulGateUpParam;
    matmulGateUpParam.hasBias = true;
    matmulGateUpParam.transposeA = false;
    matmulGateUpParam.transposeB = true;
    matmulGateUpParam.outDataType = ACL_FLOAT16;
    CreateOperation(matmulGateUpParam, &matmulGateUpNode.operation);
    matmulGateUpNode.inTensorIds = { GetTensorIdx(tensorMap, "intermediate_hiddenstates_id_quant"),
                                     GetTensorIdx(tensorMap, "in_mlp_gate_up_weight"),
                                     GetTensorIdx(tensorMap, "in_mlp_gate_up_weight_quant_bias"),
                                     GetTensorIdx(tensorMap, "in_mlp_gate_up_weight_deq_scale")
    };
    matmulGateUpNode.outTensorIds = { GetTensorIdx(tensorMap, "intermediate_matmul_up_out_id") };
    opGraph.nodes.push_back(matmulGateUpNode);
    return atb::NO_ERROR;
}

atb::Status AddMatmulSplitNode(const MlpLiteParam &param, atb::GraphParam &opGraph,
                               std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitNode;
    if (!param.noGate) {
        if (param.isPack) {
            atb::infer::SplitParam splitParam;
            splitParam.splitDim = -1; // 2: [bs, seq, 2*hidden_size]
            splitParam.splitNum = 2;  // 2: 进行二等分
            CREATE_OPERATION(splitParam, &splitNode.operation);
            splitNode.inTensorIds = { GetTensorIdx(tensorMap, "intermediate_matmul_up_out_id") };
            splitNode.outTensorIds = { GetTensorIdx(tensorMap, "intermediate_matmul_out_id"),
                                       GetTensorIdx(tensorMap, "intermediate_split_out_id") };
            opGraph.nodes.push_back(splitNode);
        }
    }
    return atb::NO_ERROR;
}

atb::Status CreateActivationSigmoid(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node sigmoidNode;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SIGMOID;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &sigmoidNode.operation));
    sigmoidNode.inTensorIds = { GetTensorIdx(tensorMap, "intermediate_matmul_out_id") };
    sigmoidNode.outTensorIds = { GetTensorIdx(tensorMap, "intermediate_sigmod_out_id") };
    opGraph.nodes.push_back(sigmoidNode);
    return atb::NO_ERROR;
}

atb::Status CreateElseWiseMul(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node sigmoidMulNode;
    atb::infer::ElewiseParam elseWiseParam;
    elseWiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elseWiseParam, &sigmoidMulNode.operation));
    sigmoidMulNode.inTensorIds = { GetTensorIdx(tensorMap, "intermediate_matmul_out_id"),
                                   GetTensorIdx(tensorMap, "intermediate_sigmod_out_id")
    };
    sigmoidMulNode.outTensorIds = { GetTensorIdx(tensorMap, "intermediate_activation_out_id") };
    opGraph.nodes.push_back(sigmoidMulNode);
    return atb::NO_ERROR;
}

atb::Status AddActivateV3Node(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(CreateActivationSigmoid(opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(CreateElseWiseMul(opGraph, tensorMap));
    return atb::NO_ERROR;
}

atb::Status AddMulV3Node(const MlpLiteParam &param, atb::GraphParam &opGraph,
                         std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mulNode;
    if (!param.noGate) {
        atb::infer::ElewiseParam mulParam;
        mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
        CREATE_OPERATION(mulParam, &mulNode.operation);
        mulNode.inTensorIds = { GetTensorIdx(tensorMap, "intermediate_activation_out_id"),
                                GetTensorIdx(tensorMap, "intermediate_split_out_id") };
        mulNode.outTensorIds = { GetTensorIdx(tensorMap, "intermediate_mul_out_id") };
        opGraph.nodes.push_back(mulNode);
    }
    return atb::NO_ERROR;
}

atb::Status AddMatmulDownV3Node(const MlpLiteParam &param, atb::GraphParam &opGraph,
                                std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node matmulDownNode;
    atb::infer::LinearParam matmulDownParam;
    matmulDownParam.hasBias = false;
    matmulDownParam.transposeA = false;
    matmulDownParam.transposeB = param.transposeB;
    CreateOperation(matmulDownParam, &matmulDownNode.operation);
    matmulDownNode.inTensorIds = { GetTensorIdx(tensorMap, "intermediate_mul_out_id"),
                                   GetTensorIdx(tensorMap, "in_mlp_down_weight")
    };
    matmulDownNode.outTensorIds = { GetTensorIdx(tensorMap, "out_result_id")};
    opGraph.nodes.push_back(matmulDownNode);
    return atb::NO_ERROR;
}

atb::Status MlpLiteLayer(const MlpLiteParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "MlpLiteLayer";

    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
        
    CHECK_OPERATION_STATUS_RETURN(QuantGateUpInput(opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(AddMatmulGateUpV3Node(opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(AddMatmulSplitNode(param, opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(AddActivateV3Node(opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(AddMulV3Node(param, opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(AddMatmulDownV3Node(param, opGraph, tensorMap));

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (inTensorDescs.at(0).dtype == ACL_INT8) {
            outTensorDescs.at(0).dtype = ACL_FLOAT16;
        }
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace common
} // namespace atb_speed