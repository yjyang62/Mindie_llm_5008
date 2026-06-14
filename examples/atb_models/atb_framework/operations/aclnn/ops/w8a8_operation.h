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

#ifndef ATB_SPEED_PLUGIN_ACLNN_W8A8_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_W8A8_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/fusion/utils.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

/// A struct defines `W8A8Operation`'s parameter.
struct AclNNQuantMatmulParam {
    /// A flag indicating whether the matmul operation includes a bias tensor.
    bool hasBias = false;
    /// A flag indicating whether the second matrix in the matmul operation is transposed.
    bool transposeB = true;
    /// A flag indicating whether the matmul operation includes a perTokenScaleOptional tensor.
    bool hasPerTokenScale = false;
    /// A flag indicating whether the tensor type is bfloat16.
    bool isBF16 = true;
    /// A flag indicating whether the matmul operation includes an offset tensor.
    bool hasOffset = false;
};

/// This class defines a matrix operation that supports
/// dynamic per-token activation quantization and weight per-channel quantization.
///
/// This class makes use of `aclnnQuantMatmulV4GetWorkspaceSize` and `aclnnQuantMatmulV4` from the AscendCL API.
///
/// Operation's Inputs:
/// Name            | Dtype   | Shape |
/// ----------------|---------|-------|
/// input           | int8    | [m,k] |
/// weight          | int8    | [n,k] if `transposeB` is true; otherwise, [k,n] |
/// weight scale    | float32 if the output tensor's dtype is float16; bfloat16 if the output tensor's dtype is bfloat16 | [n] |
/// per token scale | float32 | [m]   |
/// bias            | int32   | [n]   |
///
/// Operations's Outputs:
/// Name   | Dtype               | Shape |
/// -------|---------------------|-------|
/// output | float16 or bfloat16 | [m,n] |
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_WEIGHT_SCALE,
///     IN_PER_TOKEN_SCALE,
///     IN_BIAS,
///     OUT,
/// };
///
/// atb::Node linearNode;
/// AclNNQuantMatmulParam aclnnQuantMatmulParam;
/// aclnnQuantMatmulParam.hasBias = false;
/// aclnnQuantMatmulParam.transposeB = true;
/// linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_WEIGHT_SCALE, IN_PER_TOKEN_SCALE};
/// linearNode.outTensorIds = {OUT};
/// linearNode.operation = new atb_speed::common::W8A8Operation("W8A8LinearNode", aclnnQuantMatmulParam);
///
/// atb::Node linearWithBiasNode;
/// AclNNQuantMatmulParam aclnnQuantMatmulWithBiasParam;
/// aclnnQuantMatmulWithBiasParam.hasBias = true;
/// aclnnQuantMatmulWithBiasParam.transposeB = true;
/// linearWithBiasNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_WEIGHT_SCALE, IN_PER_TOKEN_SCALE, IN_BIAS};
/// linearWithBiasNode.outTensorIds = {OUT};
/// linearWithBiasNode.operation = new atb_speed::common::W8A8Operation(
///     "W8A8LinearWithBiasNode", aclnnQuantMatmulWithBiasParam);
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(linearNode);
/// opGraph.nodes.push_back(linearWithBiasNode);
/// \endcode
class W8A8Operation : public AclNNOperation {
public:
    explicit W8A8Operation(const std::string &name, AclNNQuantMatmulParam param);
    ~W8A8Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const;

private:
    AclNNQuantMatmulParam param_;
};
} // namespace common
} // namespace atb_speed
#endif // ATB_SPEED_PUBLIC_ACLNN_W8A8_OPERATION_H