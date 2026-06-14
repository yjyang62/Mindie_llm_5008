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

#ifndef ATB_SPEED_PLUGIN_ACLNN_DEQUANT_BIAS_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_DEQUANT_BIAS_OPERATION_H


#include "operations/aclnn/core/acl_nn_operation.h"


namespace atb_speed::common {

    /// A struct defines `aclnnDequantBias` operation parameter.
    struct AclNNDequantBiasParam {
        /// Defines the data type of the output tensor.
        aclDataType outputDtype = ACL_DT_UNDEFINED;
        /// Indicates whether the dequant operation includes an activate scale tensor
        bool hasActivateScale = false;
        /// Indicates whether the dequant operation includes a bias tensor.
        bool hasBias = false;
    };

    /// This class defines a matrix operation that supports dequantization.
    ///
    /// This class makes use of `aclnnDequantBias` form AscendCL API.
    ///
    /// Operation's Inputs:
    /// | Name            | Dtype               | Shape  |
    /// |-----------------|---------------------|--------|
    /// | input           | int32               | [m, n] |
    /// | weight scale    | float32 or bfloat16 | [n]    |
    /// | per token scale | float32 or bfloat16 | [m]    |
    /// | bias            | int32               | [n]    |
    ///
    /// Operation's Outputs:
    /// | Name            | Dtype               | Shape  |
    /// |-----------------|---------------------|--------|
    /// | output          | float16 or bfloat16 | [m, n] |
    ///
    /// Example:
    /// \code
    /// enum InTensorIdx : uint32_t {
    ///     IN_INPUT = 0,
    ///     IN_WEIGHT_SCALE,
    ///     IN_PER_TOKEN_SCALE,
    ///     IN_BIAS,
    ///     OUT,
    /// };
    ///
    /// atb::Node dequantBiasNode;
    /// AclNNDequantBiasParam aclnnDequantBiasParam;
    /// aclnnDequantBiasParam.isBF16 = true;
    /// aclnnDequantBiasParam.hasActivateScale = false;
    /// aclnnDequantBiasParam.hasBias = true;
    /// dequantBiasNode.inTensorIds = { IN_INPUT, IN_WEIGHT_SCALE, IN_BIAS };
    /// dequantBiasNode.outTensorIds = { OUT };
    /// dequantBiasNode.operation = new atb_speed::common::DequantBiasOperation(
    ///     "dequantBiasNode", aclnnDequantBiasParam
    /// );
    ///
    /// atb::GraphParam opGraph;
    /// opGraph.nodes.push_back(dequantBiasNode);
    /// \endcode
    class DequantBiasOperation : public AclNNOperation {
    public:
        explicit DequantBiasOperation(const std::string &name, AclNNDequantBiasParam param);
        ~DequantBiasOperation() override;
        atb::Status InferShape(
            const atb::SVector<atb::TensorDesc> &inTensorDesc,
            atb::SVector<atb::TensorDesc> &outTensorDesc
        ) const override;
        [[nodiscard]] uint32_t GetInputNum() const override;
        [[nodiscard]] uint32_t GetOutputNum() const override;

    protected:
        int SetAclNNWorkspaceExecutor() override;
        int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
        atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;

    private:
        AclNNDequantBiasParam param_;
        std::string opName_;
    };

}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_DEQUANT_BIAS_OPERATION_H
