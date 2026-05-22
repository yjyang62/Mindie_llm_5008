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

#ifndef ATB_SPEED_PLUGIN_ACLNN_INPLACE_NAN_TO_NUM_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_INPLACE_NAN_TO_NUM_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"


namespace atb_speed::common {


struct AclNNNanToNumParam {
    /// nanValue: Input parameter that replaces NaN values in tensor elements. The data type supports FLOAT.
    /// posInfValue: Input parameter that replaces positive infinity values in tensor elements.
    ///              The data type supports FLOAT.
    /// negInfValue: Input parameter that replaces negative infinity values in tensor elements.
    ///              The data type supports FLOAT.
    float nanValue = 0.0;
    float posInfValue = 65504.0;
    float negInfValue = -65504.0;
};

/// Replace NaN, positive infinity, and negative infinity values in the input with the
/// values specified by nan, posinf, and neginf, respectively.
///
/// Operation's Inputs: \n
/// | Name   | Dtype                    | Shape     | \n
/// |--------|--------------------------|-----------| \n
/// | x      | FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16 | [-1,…,-1] | \n
///
/// Operation's Outputs: it is inplace replace.\n
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
/// };
///
/// atb::GraphParam opGraph;
/// atb::Node nanToNumNode;
/// atb_speed::common::AclNNNanToNumParam NanToNumParam;
/// NanToNumParam.posInfValue = 50000.0;
/// NanToNumParam.negInfValue = -50000.0;
/// nanToNumNode.operation = new atb_speed::common::InplaceNanToNumOperation("nanToNumNode", NanToNumParam);
/// nanToNumNode.inTensorIds = { IN_INPUT };
/// nanToNumNode.outTensorIds = { IN_INPUT };
/// opGraph.nodes.push_back(nanToNumNode);
///
/// \endcode
class InplaceNanToNumOperation : public AclNNOperation {
public:
    explicit InplaceNanToNumOperation(const std::string &name, AclNNNanToNumParam param);
    ~InplaceNanToNumOperation() override;
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
    AclNNNanToNumParam param_;
    std::string opName_;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_INPLACE_NAN_TO_NUM_OPERATION_H

