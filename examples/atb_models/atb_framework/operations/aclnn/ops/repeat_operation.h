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

#ifndef ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed::common {
struct AclNNRepeatParam {
    std::vector<int64_t> repeatsArray;
};

/// This class defines an repeat operator.
///
/// This class makes uses of `aclnnRepeatGetWorkspaceSize` and `aclnnRepeat`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name           | Dtype               | Shape   |
/// ---------------|---------------------|---------|
/// input          | float16 or bfloat16 | [m,h]   |
///
/// Outputs of the operator:
/// Name           | Dtype               | Shape   |
/// ---------------|---------------------|---------|
/// out            | float16 or bfloat16 |[m*k,h*n]|
/// Note: k, n are the repetition times.
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     OUT,
/// };
///
/// atb::Node &repeatNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::AclNNRepeatParam repeatParam;
/// repeatParam.repeatsArray = param.repeatsArray;
/// repeatNode.operation = new atb_speed::common::RepeatOperation("RepeatOperation", repeatParam);
/// repeatNode.inTensorIds = {IN_INPUT};
/// repeatNode.outTensorIds = {OUT};
/// \endcode

class RepeatOperation : public AclNNOperation {
public:
    explicit RepeatOperation(const std::string &name, AclNNRepeatParam param);
    ~RepeatOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const override;
    [[nodiscard]] uint32_t GetInputNum() const override;
    [[nodiscard]] uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

private:
    AclNNRepeatParam param_;
    std::string opName_;
    aclIntArray *repeats_ = nullptr;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_REPEAT_OPERATION_H