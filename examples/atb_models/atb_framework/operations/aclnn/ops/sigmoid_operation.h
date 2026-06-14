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

#ifndef ATB_SPEED_PLUGIN_ACLNN_SIGMOID_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_SIGMOID_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"


/// This class defines an sigmoid operator.
///
/// This class makes uses of `aclnnSigmoidGetWorkspaceSize` and `aclnnSigmoid`
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
/// out            | float16 or bfloat16 | [m,h]   |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT = 0,
/// };
///
/// enum OutTensorIdx : uint32_t {
///     OUT = 0
/// };
///
/// atb::Node &sigmoidNode = opGraph.nodes.at(nodeId++);
/// sigmoidNode.operation = new atb_speed::common::SigmoidOperation("SigmoidOperation");
/// sigmoidNode.inTensorIds = {IN_INPUT};
/// sigmoidNode.outTensorIds = {OUT};
/// \endcode

namespace atb_speed {
namespace common {

class SigmoidOperation : public AclNNOperation {
public:
    explicit SigmoidOperation(const std::string &name);
    ~SigmoidOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
};

} // namespace common
} // namespace atb_speed
#endif // ATB_SPEED_PLUGIN_ACLNN_SIGMOID_OPERATION_H
