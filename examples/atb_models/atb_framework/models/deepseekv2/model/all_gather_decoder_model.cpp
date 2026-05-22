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
#include "models/deepseekv2/model/all_gather_decoder_model.h"
#include <vector>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"

namespace atb_speed {
namespace deepseekV2 {

AllGatherDecoderModel::AllGatherDecoderModel(const std::string &param) : DecoderModel(param)
{
    modelName_ = "ExpertAllGather";
}

atb::Status AllGatherDecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    ATB_SPEED_LOG_DEBUG("Enter AllGatherDecoderModel InferShape");
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    uint8_t inTensorDimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0) = atb::TensorDesc{};
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = inTensorDimNum + 1;
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
    atb_speed::common::ParallelInfo epParallelInfo = param.mapping.Get(base::DYNAMIC_EPLB);
    outTensorDescs.at(0).shape.dims[0] = static_cast<int>(epParallelInfo.rankIds.size());
    for (uint8_t i = 0; i < inTensorDimNum; i++) {
        outTensorDescs.at(0).shape.dims[i + 1] = inTensorDescs.at(0).shape.dims[i];
    }
    return atb::NO_ERROR;
}

int64_t AllGatherDecoderModel::BuildGraph()
{
    isUsePlanPreExecuteAsync_ = false;
    uint8_t startInTensorIdx = 0;
    this->inTensorMap_["expert_router_map"] = startInTensorIdx;
    startInTensorIdx++;
    this->graph_.inTensors.resize(startInTensorIdx);
    ATB_SPEED_LOG_DEBUG("graph_.inTensorCount_ " << startInTensorIdx);
    this->graph_.outTensors.resize(1);

    ATB_SPEED_LOG_DEBUG("AllGatherDecoderModel build graph begin");

    CHECK_OPERATION_STATUS_RETURN(AddExpertRouterAllGather());
    ATB_SPEED_LOG_DEBUG("AllGatherDecoderModel build graph success");
    return atb::NO_ERROR;
}

atb::Status AllGatherDecoderModel::AddExpertRouterAllGather()
{
    atb::Operation *op = nullptr;

    auto expertRouterAllGatherNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::AllGatherParam allGatherParam;
    atb_speed::common::ParallelInfo epParallelInfo = param.mapping.Get(base::DYNAMIC_EPLB);
    allGatherParam.rank = static_cast<int>(epParallelInfo.rank);
    allGatherParam.rankSize = static_cast<int>(epParallelInfo.rankIds.size());
    allGatherParam.backend = epParallelInfo.defaultBackend;
    epParallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &op));
    expertRouterAllGatherNode->operation.reset(op);
    expertRouterAllGatherNode->inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap_, "expert_router_map"))
    };
    expertRouterAllGatherNode->outTensors = {
        &graph_.outTensors.at(0),
    };

    ATB_SPEED_LOG_DEBUG("AllGatherDecoderModel build graph : expertRouterAllGatherNode end");
    graph_.nodes.push_back(*expertRouterAllGatherNode);
    return atb::NO_ERROR;
}

} // namespace deepseekV2
} // namespace atb_speed
