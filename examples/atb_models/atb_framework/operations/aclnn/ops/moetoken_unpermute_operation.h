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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_TOKEN_UNPERMUTE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_TOKEN_UNPERMUTE_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"


/// This class defines an operator that is used to gather and reduce hidden states based on sortedIndices.
///
/// This class makes uses of `aclnnMoeTokenUnpermuteGetWorkspaceSize` and `aclnnMoeTokenUnpermute`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name           | Dtype               | Shape   |
/// ---------------|---------------------|---------|
/// permutedTokens | float16 or bfloat16 | [m*k,h] |
/// sortedIndices  | int32               | [m*k]   |
/// expertsWeights | float16 or bfloat16 | [m,k]   |
///
/// Outputs of the operator:
/// Name                        | Dtype | Shape   |
/// ----------------------------|-------|---------|
/// out                         | int32 | [m*k,h] |
/// Note:  k is the number of experts selected for each token
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_IDX,
///     IN_EXPERT_WEIGHT,
///     OUT_MOE_MLP_RESULT,
/// };
///
/// atb::Node &unpermuteNode = opGraph.nodes.at(nodeId++);
/// unpermuteNode.operation = new atb_speed::common::MoeTokenUnpermuteOperation("MoeTokenUnpermuteNode");
/// unpermuteNode.inTensorIds = {IN_INPUT,
///                              IN_IDX,
///                              IN_EXPERT_WEIGHT};
/// unpermuteNode.outTensorIds = {OUT_MOE_MLP_RESULT};
/// \endcode

namespace atb_speed::common {
    class MoeTokenUnpermuteOperation : public AclNNOperation {
    public:
        explicit MoeTokenUnpermuteOperation(const std::string &name);
        ~MoeTokenUnpermuteOperation() override;
        atb::Status InferShape(
            const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs
        ) const override;
        [[nodiscard]] uint32_t GetInputNum() const override;
        [[nodiscard]] uint32_t GetOutputNum() const override;

    protected:
        int SetAclNNWorkspaceExecutor() override;
        int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_TOKEN_UNPERMUTE_OPERATION_H