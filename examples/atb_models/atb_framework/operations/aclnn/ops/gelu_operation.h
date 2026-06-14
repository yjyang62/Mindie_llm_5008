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

#ifndef ATB_SPEED_PLUGIN_ACLNN_GELU_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_GELU_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"


namespace atb_speed::common {

    /// A struct defines `aclnnGelu` and `aclnnGelvV2` operation parameter.
    struct AclNNGeluParam {
        /// Indicates the gelu approximation algorithm to use.
        ///
        /// -1: use `aclnnGelu` operation, and use Tanh approximation approach to calculate Gelu.
        /// 0: use `aclnnGelvV2` operation, and use Cumulative Distribution Function for Gaussian Distribution.
        /// 1: use `aclnnGelvV2` operation, and use Tanh approximation approach to calculate Gelu.
        int64_t geluApproximate = -1;
    };

    /// This class defines a matrix operation that applies the Gaussian Error Linear Units function.
    ///
    /// This class makes use of `aclnnGeluGetWorkspaceSize` and `aclnnGeluV2GetWorkspaceSize` from AscendCL API.
    ///
    /// Operation's Inputs: \n
    /// | Name   | Dtype                    | Shape     | \n
    /// |--------|--------------------------|-----------| \n
    /// | x      | float32/float16/bfloat16 | [-1,…,-1] | \n
    ///
    /// Operation's Outputs: \n
    /// | Name   | Dtype                    | Shape     | \n
    /// |--------|--------------------------|-----------| \n
    /// | output | float32/float16/bfloat16 | [-1,…,-1] | \n
    ///
    /// Example:
    /// \code
    /// enum TensorIdx : uint32_t {
    ///     IN_INPUT = 0,
    ///     OUT,
    /// };
    ///
    /// atb::Node geluNode;
    /// AclNNGeluParam geluParam;
    /// geluParam.geluApproximate = 1;
    /// geluNode.inTensorIds = { IN_INPUT };
    /// geluNode.outTensorIds = { OUT };
    /// geluNode.operation = new atb_speed::common::GeluOperation("geluNode", geluParam);
    ///
    /// atb::GraphParam opGraph;
    /// opGraph.nodes.push_back(geluNode);
    /// \endcode
    class GeluOperation : public AclNNOperation {
    public:
        explicit GeluOperation(const std::string &name, AclNNGeluParam param);
        ~GeluOperation() override;
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
        AclNNGeluParam param_;
        std::string opName_;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_GELU_OPERATION_H
