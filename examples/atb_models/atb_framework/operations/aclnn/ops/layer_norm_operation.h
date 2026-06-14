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

#ifndef ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"


namespace atb_speed::common {

    /// A struct defines `aclnnLayerNormWithImplModeGetWorkspaceSize` operation parameter.
    struct AclNNLayerNormParam {
        /// Indicates a value added to the denominator for numerical stability.
        float layerNormEps = 0;
        /// Indicates the start of normalization axis.
        int beginNormAxis = 0;
        /// Indicates the number of normalization axes.
        int normAxes = 1;
        /// Indicates the accuracy implementation mode in execution.
        ///
        /// 0: high accuracy mode.
        /// 1: high performance mode.
        /// 2: keep dtype of `float16` in execution.
        int64_t layerNormImplMode = 0;
        /// Indicates whether inputs include a bias tensor.
        bool hasBias = true;
    };

    /// This class defines a matrix operation that applies Layer Normalization over a mini-batch of inputs.
    ///
    /// This class makes use of `aclnnLayerNormGetWorkspaceSize` and `aclnnLayerNormWithImplModeGetWorkspaceSize`
    /// from the AscendCL API.
    ///
    /// Operation's Inputs: \n
    /// | Name   | Dtype                    | Shape                   | \n
    /// |--------|--------------------------|-------------------------| \n
    /// | input  | float32/float16/bfloat16 | [-1,…,-1]               | \n
    /// | weight | float32/float16/bfloat16 | [beginNormAxis:]/[1:-1] | \n
    /// | bias   | float32/float16/bfloat16 | [beginNormAxis:]/[1:-1] | \n
    ///
    /// Operation's Outputs: \n
    /// | Name   | Dtype                    | Shape                   | \n
    /// |--------|--------------------------|-------------------------| \n
    /// | output | float32/float16/bfloat16 | [-1,…,-1]               | \n
    ///
    /// Example:
    /// \code
    /// enum TensorIdx : uint32_t {
    ///     IN_INPUT = 0,
    ///     IN_WEIGHT,
    ///     IN_BIAS,
    ///     OUT,
    /// };
    ///
    /// atb::Node layerNormNode;
    /// AclNNLayerNormParam layerNormParam;
    /// layerNormParam.layerNormEps = 1e-5;
    /// layerNormParam.beginNormAxis = -1;
    /// layerNormParam.normAxes = 1;
    /// layerNormParam.hasBias = true;
    /// layerNormNode.inTensorIds = { IN_INPUT, IN_WEIGHT, IN_BIAS };
    /// layerNormNode.outTensorIds = { OUT };
    /// layerNormNode.operation = new atb_speed::common::LayerNormOperation("layerNormNode", layerNormParam);
    ///
    /// atb::GraphParam opGraph;
    /// opGraph.nodes.push_back(layerNormNode);
    /// \endcode
    class LayerNormOperation : public AclNNOperation {
    public:
        explicit LayerNormOperation(const std::string &name, AclNNLayerNormParam param);
        ~LayerNormOperation() override;
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
        atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

    private:
        AclNNLayerNormParam param_;
        std::string opName_;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_LAYER_NORM_OPERATION_H
