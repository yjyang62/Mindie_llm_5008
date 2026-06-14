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

#ifndef ATB_SPEED_PLUGIN_ACLNN_W8A16_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_W8A16_OPERATION_H
#include "quant_batch_matmul_operation.h"

namespace atb_speed {
namespace common {

/// This class defines a matrix operation that supports 8-bit weight quantization
/// while keeping activations in floating-point format.
///
/// It inherits from the `QuantBatchMatmulOperation` class.
///
/// Operation's Inputs (per channel):
/// Name            | Dtype   | Shape |
/// ----------------|---------|-------|
/// input           | int8    | [m,k] |
/// weight          | int8    | [n,k] if `transposeB` is true; otherwise, [k,n] |
/// antiquant scale | the same dtype as the output tensor | [n,1] if `transposeB` is true; otherwise, [1,n] |
/// antiquant offset| the same dtype as the output tensor | [n,1] if `transposeB` is true; otherwise, [1,n] |
/// bias            | int32 if the output tensor's dtype is float16; bfloat16 if the output tensor's dtype is bfloat16   | [n]   |
///
/// Operation's Inputs (per group):
/// Name            | Dtype   | Shape |
/// ----------------|---------|-------|
/// input           | int8    | [m,k] |
/// weight          | int8    | [n,k] if `transposeB` is true; otherwise, [k,n] |
/// antiquant scale | the same dtype as the output tensor | [n,ceil(k, group_size)] if `transposeB` is true; otherwise, [ceil(k, group_size),n] |
/// antiquant offset| the same dtype as the output tensor | [n,ceil(k, group_size)] if `transposeB` is true; otherwise, [ceil(k, group_size),n] |
/// bias            | int32 if the output tensor's dtype is float16; bfloat16 if the output tensor's dtype is bfloat16 | [n]   |
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
///     IN_WEIGHT_ANTI_QUANT_SCALE,
///     IN_WEIGHT_ANTI_QUANT_OFFSET,
///     IN_BIAS,
///     OUT,
/// };
///
/// atb::Node linearNode;
/// AclNNWeightQuantBatchMatmulParam aclnnQuantBatchMatmulParam;
/// aclnnQuantBatchMatmulParam.hasBias = false;
/// aclnnQuantBatchMatmulParam.transposeB = true;
/// aclnnQuantBatchMatmulParam.quantGroupSize = 0;  // 0: per channel; otherwise, per group
/// linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_WEIGHT_ANTI_QUANT_SCALE, IN_WEIGHT_ANTI_QUANT_OFFSET};
/// linearNode.outTensorIds = {OUT};
/// linearNode.operation = new atb_speed::common::W8A16Operation("W8A16LinearNode", aclnnQuantBatchMatmulParam);
///
/// atb::Node linearWithBiasNode;
/// AclNNWeightQuantBatchMatmulParam aclnnQuantBatchMatmulWithBiasParam;
/// aclnnQuantBatchMatmulWithBiasParam.hasBias = true;
/// aclnnQuantBatchMatmulWithBiasParam.transposeB = true;
/// linearWithBiasNode.inTensorIds = {
///     IN_INPUT, IN_WEIGHT, IN_WEIGHT_ANTI_QUANT_SCALE, IN_WEIGHT_ANTI_QUANT_OFFSET, IN_BIAS};
/// linearWithBiasNode.outTensorIds = {OUT};
/// linearWithBiasNode.operation = new atb_speed::common::W8A16Operation(
///     "W8A16LinearWithBiasNode", aclnnQuantBatchMatmulWithBiasParam);
///
/// // Add the operation node to the graph as required
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(linearNode);
/// opGraph.nodes.push_back(linearWithBiasNode);
/// \endcode
class W8A16Operation : public QuantBatchMatmulOperation {
public:
    explicit W8A16Operation(const std::string &name, AclNNWeightQuantBatchMatmulParam param);

protected:
    atb::Tensor PreprocessATBInTensor(atb::Tensor atbTensor, int index) override;

private:
    AclNNWeightQuantBatchMatmulParam param_;
};
} // namespace common
} // namespace atb_speed
#endif