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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_COMPUTE_EXPERT_TOKENS_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_COMPUTE_EXPERT_TOKENS_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

struct MoeComputeExpertTokensParam {
    /// The total number of experts utilized by the model
    int32_t expertNum = 8;
};

/// This class defines an operator that computes the number of tokens that is processed by each expert
///
/// This class makes uses of `aclnnMoeComputeExpertTokensGetWorkspaceSize` and `aclnnMoeComputeExpertTokens`
/// form the AscendCL API.
///
/// Inputs to the operator
/// Name         | Dtype | Shape |
/// -------------|-------|-------|
/// input        | int32 | [m*k] |
///
/// Outputs of the operator:
/// Name         | Dtype | Shape |
/// -------------|-------|-------|
/// output       | int64 | [e]   |
/// Note: m is the length of input tokens, k is the number of experts selected for each token,
/// e is the total number of experts used by the model
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     INPUT = 0,
///     OUT,
/// };
///
/// atb::Node &expertTokenNode = opGraph.nodes.at(nodeId++);
/// expertTokenNode.operation = new atb_speed::common::MoeComputeExpertTokensOperation("ArgsortNode");
/// expertTokenNode.inTensorIds = {INPUT};
/// expertTokenNode.outTensorIds = {OUTPUT};
/// \endcode

class MoeComputeExpertTokensOperation : public AclNNOperation {
public:
    explicit MoeComputeExpertTokensOperation(const std::string &name, MoeComputeExpertTokensParam param);
    ~MoeComputeExpertTokensOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    MoeComputeExpertTokensParam param_;
};

}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MOE_TOPK_SOFTMAX_OPERATION_H