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

#ifndef ATB_SPEED_PLUGIN_ACLNN_ARGSORT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_ARGSORT_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"
namespace atb_speed {
namespace common {

/// This class defines an operator that returns the indices that would sort the input.
///
/// This class class makes uses of `aclnnArgsortGetWorkspaceSize` and `aclnnArgsort` from the AscendCL API.
///
/// Inputs to the operator:
/// Name         | Dtype                                       | Shape |
/// -------------|---------------------------------------------|-------|
/// input        | float16, float32, int8, int32, int64, uint8 | [m,n] |
///
/// Outputs of the operator:
/// Name         | Dtype | Shape |
/// -------------|-------|-------|
/// output       | int64 | [m,n] |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     INPUT = 0,
///     OUT,
/// };
///
/// atb::Node &argsortNode = opGraph.nodes.at(nodeId++);
/// argsortNode.operation = new atb_speed::common::ArgSortOperation("ArgsortNode");
/// argsortNode.inTensorIds = {INPUT};
/// argsortNode.outTensorIds = {OUTPUT};
/// \endcode

class ArgSortOperation : public AclNNOperation {
public:
    explicit ArgSortOperation(const std::string &name);

    ~ArgSortOperation() override;

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

}  // namespace common
}  // namespace atb_speed

#endif  // ATB_SPEED_PLUGIN_ACLNN_ARGSORT_OPERATION_H