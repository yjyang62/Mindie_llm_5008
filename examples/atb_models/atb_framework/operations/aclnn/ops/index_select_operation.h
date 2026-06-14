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

#ifndef ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"


namespace atb_speed::common {

/// A struct defines `IndexSelect`'s parameter.
struct IndexSelectParam {
    /// A flag indicating the specified dimension of the input tensor,
    /// the range is [-input.dim(), input.dim() - 1].
    int64_t dim = 0;
};

/// This class defines a matrix operation that supports
/// extract elements from the specified dimension dim of the input Tensor according to the index sequence numbers
/// and save them to the out Tensor.
///
/// This class makes use of `aclnnIndexSelectGetWorkspaceSize` and `aclnnIndexSelect` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype   | Shape |
/// ----------------|---------|-------|
/// input | float32, float16, bfloat16 | The dimension is not greater than 8 |
/// index | int32, int64               | [n] |
///
/// Operations's Outputs:
/// Name   | Dtype               | Shape |
/// -------|---------------------|-------|
/// output | same as input | The dimension is the same as input. The length of the dim dimension is equal to the index.|
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_INDEX,
///     OUT,
/// };
///
/// atb::Node indexSelectNode;
/// IndexSelectParam indexSelectParam;
/// indexSelectParam.dim = 0;
/// indexSelectNode.inTensorIds = {IN_INPUT, IN_INDEX};
/// indexSelectNode.outTensorIds = {OUT};
/// indexSelectNode.operation = new atb_speed::common::IndexSelectOperation("IndexSelectNode", IndexSelectParam);
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(indexSelectNode);
/// \endcode

class IndexSelectOperation : public AclNNOperation {
public:
    explicit IndexSelectOperation(const std::string &name, IndexSelectParam param);
    ~IndexSelectOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    IndexSelectParam param_;
    std::string opName_;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_INDEX_SELECT_OPERATION_H
