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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MASKEDFILL_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MASKEDFILL_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed::common {
struct InplaceMaskedFillTensorParam {
    float value = 0;
    aclDataType outDataType = ACL_FLOAT16;
};

/// This class defines an operator that replaces the value in the tensor with another specified value.
///
/// This class makes uses of `aclnnInplaceMaskedFillScalarGetWorkspaceSize` and `aclnnInplaceMaskedFillScalar`
/// form the AscendCL API.
///
/// Inputs to the operator
/// Name         | Dtype               | Shape |
/// -------------|---------------------|-------|
/// input        | float16 or bfloat16 | [m]   |
///
/// Outputs of the operator:
/// Name         | Dtype               | Shape |
/// -------------|---------------------|-------|
/// output       | float16 or bfloat16 | [m]   |
/// Note: The output is a placeholder that wouldn't be written during executing.
///
/// Example:
/// \code
/// enum InTensorIdx : uint32_t {INPUT = 0};
///
/// enum OutTensorIdx : uint32_t {OUT = 0};
///
/// atb::Node &maskedFillNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::InplaceMaskedFillTensorParam fillParam;
/// fillParam.value = param.fillValue;
/// fillParam.outDataType = param.outDataType;
/// maskedFillNode.operation = new atb_speed::common::InplaceMaskedFillTensorOperation("MaskedFill", fillParam);
/// maskedFillNode.inTensorIds = {INPUT};
/// maskedFillNode.outTensorIds = {OUTPUT};
/// \endcode

class InplaceMaskedFillTensorOperation : public AclNNOperation {
public:
    explicit InplaceMaskedFillTensorOperation(const std::string &name, InplaceMaskedFillTensorParam param);
    ~InplaceMaskedFillTensorOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs
    ) const override;
    [[nodiscard]] uint32_t GetInputNum() const override;
    [[nodiscard]] uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    InplaceMaskedFillTensorParam param_;
    std::string opName_;
    aclScalar* value_ = nullptr;
};
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_MASKEDFILL_OPERATION_H