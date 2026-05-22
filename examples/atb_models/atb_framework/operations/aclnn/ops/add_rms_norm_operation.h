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

#ifndef ATB_SPEED_PLUGIN_ACLNN_ADDRMSNORM_OPERATION_V2_H
#define ATB_SPEED_PLUGIN_ACLNN_ADDRMSNORM_OPERATION_V2_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
/// This class defines a matrix operation combines the Add operator before the RmsNorm,
/// reducing the operations of moving data in and out.
///
/// This class makes use of `aclnnAddRmsNormGetWorkspaceSize` and `aclnnAddRmsNorm` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype                       | Shape   |
/// ----------------|-----------------------------|---------|
/// x1              | FLOAT, FLOAT16, BFLOAT16    | 1-8 dim |
/// x2              | FLOAT, FLOAT16, BFLOAT16    | 1-8 dim |
/// gamma           | FLOAT, FLOAT16, BFLOAT16    | 1-8 dim |
/// epsilon         | double                      | Scalar  |
///
/// Operations's Outputs:
/// Name    | Dtype                       | Shape   |
/// --------|-----------------------------|---------|
/// yOut    | FLOAT, FLOAT16, BFLOAT16    | 1-8 dim |
/// rstdOut | FLOAT                       | 1-8 dim |
/// xOut    | FLOAT, FLOAT16, BFLOAT16    | 1-8 dim |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT1 = 0,
///     IN_INPUT2,
///     IN_WEIGHT,
/// };
///
/// enum OutTensorIdx : uint32_t {
///     OUT1 = 0,
///     OUT2,
///     OUT3,
/// };
///
/// atb::Node addNormNode;
/// addNormNode.operation = new atb_speed::common::AddRmsNormOperation("AddRmsNormNode",  param.rmsNormEps);
/// addNormNode.outTensorIds = {OUT1, OUT2, OUT3};
/// addNormNode.inTensorIds = {IN_INPUT1, IN_INPUT2, IN_WEIGHT};
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(addNormNode);
/// \endcode
class AddRmsNormOperation : public AclNNOperation {
public:
    explicit AddRmsNormOperation(const std::string &name, float epsilon);
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    float epsilon = 1e-5;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
};
} // namespace common
} // namespace atb_speed
#endif