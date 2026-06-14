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

#ifndef ATB_SPEED_PLUGIN_ACLNN_DYNAMIC_QUANT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_DYNAMIC_QUANT_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

/// This class defines an dynamic quant operator.
///
/// This class makes uses of `aclnnDynamicQuantGetWorkspaceSize` and `aclnnDynamicQuant`
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
/// out            | int8                | [m,h]   |
/// tokenScales    | float16 or bfloat16 | [m]     |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     OUT,
///     OUT_SCALE,
/// };
///
/// atb::Node &dynamicQuantNode = opGraph.nodes.at(nodeId++);
/// dynamicQuantNode.operation = new atb_speed::common::DynamicQuantOperation("DynamicQuantOperation");
/// dynamicQuantNode.inTensorIds = {IN_INPUT};
/// dynamicQuantNode.outTensorIds = {OUT, OUT_SCALE};
/// \endcode

namespace atb_speed::common {

class DynamicQuantOperation : public AclNNOperation {
public:
    explicit DynamicQuantOperation(const std::string &name);
    ~DynamicQuantOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                            atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
};
}

#endif  // ATB_SPEED_PLUGIN_ACLNN_DYNAMIC_QUANT_OPERATION_H