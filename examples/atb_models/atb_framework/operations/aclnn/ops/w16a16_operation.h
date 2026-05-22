/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_PLUGIN_ACLNN_W16A16_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_W16A16_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "aclnnop/aclnn_addmm.h"

namespace atb_speed {
namespace common {

/// A struct defines `W16A16Operation`'s parameter.
struct AclNNMatmulParam {
    /// A flag indicating whether the second matrix in the matmul operation is transposed.
    bool transposeB = false;
    /// A flag indicating whether the matmul operation includes a bias tensor.
    bool hasBias = false;
};

/// This class defines a matrix operation combines the matmul and add bias operation.
///
/// This class makes use of `aclnnAddmmGetWorkspaceSize` and `aclnnAddmm` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype                       | Shape | Description |
/// ----------------|-----------------------------|-------|-------------|
/// input           | FLOAT, FLOAT16, BFLOAT16    | [m,k] | |
/// weight          | FLOAT, FLOAT16, BFLOAT16    | [n,k] if `transposeB` is true; otherwise, [k,n] | |
/// bias            | FLOAT, FLOAT16, BFLOAT16    | [m,n] or can be broadcasted to [m,n] | Optional. Required if `hasBias` is true. |
///
/// Operations's Outputs:
/// Name   | Dtype                              | Shape |
/// -------|------------------------------------|-------|
/// out    | the same dtype as the input tensor | [m,n] |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_BIAS,
///     OUT,
/// };
///
/// atb::Node linearNode;
/// AclNNMatmulParam aclNNMatmulParam;
/// aclNNMatmulParam.hasBias = false;
/// aclNNMatmulParam.transposeB = true;
/// linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
/// linearNode.outTensorIds = {OUT};
/// linearNode.operation = new atb_speed::common::W16A16Operation("W16A16LinearNode", aclNNMatmulParam);
///
/// atb::Node linearWithBiasNode;
/// AclNNMatmulParam aclNNMatmulParam;
/// aclNNMatmulParam.hasBias = true;
/// aclNNMatmulParam.transposeB = true;
/// linearWithBiasNode.inTensorIds = {
///     IN_INPUT, IN_WEIGHT, IN_BIAS};
/// linearWithBiasNode.outTensorIds = {OUT};
/// linearWithBiasNode.operation = new atb_speed::common::W16A16Operation(
///     "W16A16LinearWithBiasNode", aclNNMatmulParam);
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(linearNode);
/// opGraph.nodes.push_back(linearWithBiasNode);
/// \endcode

class W16A16Operation : public AclNNOperation {
public:
    explicit W16A16Operation(const std::string &name, AclNNMatmulParam param);
    ~W16A16Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

    atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const;

    AclNNMatmulParam param_;
    aclScalar* betaZero = nullptr;
    aclScalar* betaOne = nullptr;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PUBLIC_ACLNN_W8A8_OPERATION_H