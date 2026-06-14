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

#ifndef ATB_SPEED_PLUGIN_ACLNN_ROTARY_POS_EMB_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_ROTARY_POS_EMB_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

/// This class defines an operator that calculates the standard deviation of the input.
///
/// This class makes uses of `aclnnStdGetWorkspaceSize` and `aclnnStd` from AscendCL Api.
///
/// Inputs to the operator:
/// Name         | Dtype               | Shape |
/// -------------|---------------------|-------|
/// queryRef        | float16 or bfloat16 | [b,s,n,d] |
/// keyRef        | float16 or bfloat16 | [b,s,n,d] |
/// cos        | float16 or bfloat16 | [b,s,n,d] |
/// sin        | float16 or bfloat16 | [b,s,n,d] |
///

///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     QUERY = 0,
///     KEY,
///     COS,
///     SIN
/// };
/// atb::Node &RotaryPosEmbNode = opGraph.nodes.at(nodeId++);
/// RotaryPosEmbNode.operation = new atb_speed::common::RotaryPosEmbOperation("RotaryPosEmbNode");
/// RotaryPosEmbNode.inTensorIds = {QUERY, KEY, COS, SIN};
/// \endcode

class RotaryPosEmbOperation : public AclNNOperation {
public:
    explicit RotaryPosEmbOperation(const std::string &name);
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    ~RotaryPosEmbOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Dims GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
};
} // namespace common
} // namespace atb_speed
#endif