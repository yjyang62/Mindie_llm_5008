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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_INIT_ROUTING_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_INIT_ROUTING_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct MoeInitRoutingParam {
    /// The number of experts selected for each token
    int32_t topkNum = 2;
    /// The non-deepseek models do not have the scaledTopk feature enabled by default
    int scaledTopk = -1;
    bool enableInitRoutingCutoff = false;
    /// The total number of experts utilized by the model
    int32_t expertNum = 8;
    int expertTokensCoutOrCumsumFlag = 1;
    int targetInputSize = 65536;
};

/// This class defines an operator that is used to gather and rearrange hidden states based
/// on the given list of selected experts of each token.
///
/// This class makes uses of `aclnnMoeInitRoutingV2GetWorkspaceSize` and `aclnnMoeInitRoutingV2`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name         | Dtype               | Shape |
/// -------------|---------------------|-------|
/// input        | float16 or bfloat16 | [m,h] |
/// expertIdx    | int32               | [m,k] |
///
/// Outputs of the operator:
/// Name                         | Dtype | Shape   |
/// -----------------------------|-------|---------|
/// expandedXOut                 | int32 | [m*k,h] |
/// expandedRowIdxOut            | int32 | [m*k]   |
/// expertTokensCountOrCumsumOut | int32 | [e]     |
/// Note: e is the total number of experts utilized by the model
/// k is the number of experts selected for each token
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_EXPERTIDX,
///     OUT_SORTED_HIDDENSTATES,
///     OUT_ROWIDX,
///     OUT_GROUP_LIST
/// };
///
/// atb::Node &initRoutingNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::MoeInitRoutingParam initRoutingParam;
/// initRoutingParam.topkNum = param.topk;
/// initRoutingParam.expertNum = param.numOfExperts;
/// initRoutingNode.operation = new atb_speed::common::MoeInitRoutingOperation("MoeInitRoutingOperation",
///                                                                            initRoutingParam);
/// initRoutingNode.inTensorIds = {IN_PUT, IN_EXPERTIDX};
/// initRoutingNode.outTensorIds = {OUT_SORTED_HIDDENSTATES,
///                                 OUT_ROWIDX,
///                                 OUT_GROUP_LIST};
/// \endcode

class MoeInitRoutingOperation : public AclNNOperation {
public:
    explicit MoeInitRoutingOperation(const std::string &name, MoeInitRoutingParam param);
    ~MoeInitRoutingOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    MoeInitRoutingParam param_;
};

template <typename ParamType>
int GetMoeInitRoutingOpOutputSize(int inputSize, const ParamType& param)
{
    int outputSize;
    if (param.enableInitRoutingCutoff) {
        // outputSize公式：min(未裁剪, max(裁剪后, 长上下文要求))
        outputSize = std::min(
            inputSize * param.topkNum,
            std::max(inputSize * param.scaledTopk,
                param.targetInputSize * param.scaledTopk));
    } else {
        outputSize = inputSize * param.topkNum;
    }
    return outputSize;
}

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H