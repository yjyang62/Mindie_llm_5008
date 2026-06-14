/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "operations/fusion/mlp/mlp.h"
#include "models/minicpm/layer/decoder_layer.h"


namespace atb_speed {
namespace minicpm {

void MiniCPMLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    std::stringstream ss;
    ss << "MiniCPM Layer Param: " << ", numHiddenLayers: " << this->numHiddenLayers
       << ", scaleDepth: " << this->scaleDepth;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

MiniCPMDecoderLayer::MiniCPMDecoderLayer(
    const MiniCPMLayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
    this->param.CheckParam();
}

atb::Status MiniCPMDecoderLayer::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttention());

    atb::Node attMulsNode;
    float scale = param.scaleDepth / sqrt(static_cast<float>(param.numHiddenLayers));
    atb::infer::ElewiseParam scaleParam;
    ATB_SPEED_LOG_DEBUG("redis scale " << scale);
    scaleParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    scaleParam.mulsParam.varAttr = scale;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(scaleParam, &attMulsNode.operation));
    attMulsNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_attn_out"});
    attMulsNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_attn_out"});
    this->graph.nodes.push_back(attMulsNode);

    atb::Node selfResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
    selfResidualAddNode.inTensorIds = \
            atb_speed::common::GetTensorIdxList(this->tensorMap, {"in_hidden_states", "intermediate_attn_out"});
    selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"in_hidden_states"});
    this->graph.nodes.push_back(selfResidualAddNode);

    CHECK_OPERATION_STATUS_RETURN(this->AddMlp());

    atb::Node mlpMulsNode;
    atb::infer::ElewiseParam scaleMlpParam;
    scaleMlpParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    scaleMlpParam.mulsParam.varAttr = scale;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(scaleMlpParam, &mlpMulsNode.operation));
    mlpMulsNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    mlpMulsNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    this->graph.nodes.push_back(mlpMulsNode);

    atb::Node mlpResidualAddNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    mlpResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {"in_hidden_states", "intermediate_mlp_out"});
    mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"out"});
    this->graph.nodes.push_back(mlpResidualAddNode);

    return atb::NO_ERROR;
}
} // namespace minicpm
} // namespace atb_speed