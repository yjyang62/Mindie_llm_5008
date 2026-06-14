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

#ifndef ATB_SPEED_PLUGIN_ACLNN_FINALIZE_ROUTING_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_FINALIZE_ROUTING_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

/// This class defines an operator that performs scaling, sorting, and reducing by summation
/// on the input.
///
/// This class makes uses of `aclnnMoeFinalizeRoutingGetWorkspaceSize` and `aclnnMoeFinalizeRouting`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name               | Dtype               | Shape   |
/// -------------------|---------------------|---------|
/// input1             | float16 or bfloat16 | [m*k,h] |
/// input2             | the same as input1  | [m,h]   |
/// input3             | the same as input1  | [m,h]   |
/// bias               | the same as input1  | [e,h]   |
/// scales             | the same as input1  | [m,k]   |
/// expandedRowIdx     | int32               | [m*k]   |
/// expandedExpertIdx  | int32               | [m,k]   |
///
/// Outputs of the operator:
/// Name          | Dtype              | Shape   |
/// --------------|--------------------|---------|
/// output        | the same as input1 | [m,h]   |

/// Note: e is the total number of experts utilized by the model
/// k is the number of experts selected for each token
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_INPUT_TWO,
///     IN_INPUT_THREE
///     IN_BIAS,
///     IN_SCALES,
///     IN_EXPANDED_ROW_IDX,
///     IN_EXPANDED_EXPERT_IDX,
///     OUT
/// };
///
/// atb::Node &finalizeRoutingNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::MoefinalizeRoutingParam finalizeRoutingParam;
/// finalizeRoutingParam.topkNum = param.topk;
/// finalizeRoutingParam.expertNum = param.numOfExperts;
/// finalizeRoutingNode.operation = new atb_speed::common::FinalizeRoutingOperation("MoeFinalizeRoutingOperation",
///                                                                            initRoutingParam);
/// initRoutingNode.inTensorIds = {IN_INPUT = 0,
///                                IN_INPUT_TWO,
///                                IN_INPUT_THREE
///                                IN_BIAS,
///                                IN_SCALES,
///                                IN_EXPANDED_ROW_IDX,
///                                IN_EXPANDED_EXPERT_IDX};
/// initRoutingNode.outTensorIds = {OUT};
/// \endcode

class FinalizeRoutingOperation : public AclNNOperation {
public:
    explicit FinalizeRoutingOperation(const std::string &name);
    ~FinalizeRoutingOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
};

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H