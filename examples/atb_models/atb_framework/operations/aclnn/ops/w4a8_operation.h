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

#ifndef ATB_SPEED_PLUGIN_ACLNN_W4A8_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_W4A8_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/fusion/utils.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {

/// A struct defines `W4A8Operation`'s parameter.
struct AclNNW4A8Param {
    /// A flag indicating whether the tensor type is bfloat16.
    aclDataType outDataType = ACL_FLOAT16;
    /// A flag indicating whether the matmul operation includes an offset tensor.
    bool hasBias = false;
    /// Group size of per group quantization.
    int groupSize = 256;
};

class W4A8Operation : public AclNNOperation {
public:
    explicit W4A8Operation(const std::string &name, AclNNW4A8Param param);
    ~W4A8Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    AclNNW4A8Param param_;
};
} // namespace common
} // namespace atb_speed
#endif // ATB_SPEED_PUBLIC_ACLNN_W4A8_OPERATION_H