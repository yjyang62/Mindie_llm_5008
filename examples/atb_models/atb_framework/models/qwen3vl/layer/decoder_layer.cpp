/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "models/qwen3vl/layer/decoder_layer.h"

namespace atb_speed {
namespace qwen3vl {

const int DEEP_STACK_LAYER_NUM = 3;

DecoderLayer::DecoderLayer(const atb_speed::base::LayerParam &param)
    : base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->inTensorCandidates["deepstack_visual_embeds"] = {
        "in_deepstack_visual_embeds_0",
        "in_deepstack_visual_embeds_1",
        "in_deepstack_visual_embeds_2"};
};

void DecoderLayer::ConstructInTensorMap()
{
    base::DecoderLayer<atb::infer::RmsNormParam>::ConstructInTensorMap();
    if (this->param.isPrefill) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "deepstack_visual_embeds", this->inTensorList);
    }
}

atb::Status DecoderLayer::AddDeepStack(int layerId)
{
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Node deepStackNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &deepStackNode.operation));
    std::string deepStackTensorName = "in_deepstack_visual_embeds_" + std::to_string(layerId);
    deepStackNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {"out", deepStackTensorName});
    deepStackNode.outTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {"out"});
    this->graph.nodes.push_back(deepStackNode);
    return atb::NO_ERROR;
}

atb::Status DecoderLayer::BuildGraph(atb::Operation **operation)
{
    this->graph.name = this->param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    this->ConstructInTensorMap();
    this->ConstructInternalTensorMap();
    this->graph.inTensorNum = this->inTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.inTensorNum " << this->graph.inTensorNum);
    this->graph.internalTensorNum = this->intermediateTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.internalTensorNum " << this->graph.internalTensorNum);
    this->graph.outTensorNum = this->outTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.outTensorNum " << this->graph.outTensorNum);
    this->tensorMap = atb_speed::common::GetTensorMap(
        this->inTensorList, this->outTensorList, this->intermediateTensorList);
    std::stringstream ss;

    for (auto tensor = this->tensorMap.cbegin(); tensor != this->tensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("layer map tensor:\n" << ss.str());

    CHECK_OPERATION_STATUS_RETURN(this->AddOperationToGraph());

    // deepstack
    if (this->param.isPrefill && this->param.layerId < DEEP_STACK_LAYER_NUM) {
        CHECK_OPERATION_STATUS_RETURN(this->AddDeepStack(this->param.layerId));
    }

    uint32_t inHiddenStatesIdx = atb_speed::common::GetTensorIdx(this->tensorMap, "in_hidden_states");
    this->graph.inferShapeFunc = [inHiddenStatesIdx](
        const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(inHiddenStatesIdx);
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(this->graph, operation));
    return atb::NO_ERROR;
}

} // namespace qwen3vl
} // namespace atb_speed