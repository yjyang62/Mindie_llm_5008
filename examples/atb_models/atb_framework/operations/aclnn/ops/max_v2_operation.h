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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MAX_V2_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MAX_V2_OPERATION_H

#include <vector>
#include "aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {
struct AclNNMaxV2Param {
    std::vector<int64_t> dims = {-1};
    bool keepdim = false;
};
class MaxV2Operation : public AclNNOperation {
public:
    explicit MaxV2Operation(const std::string &name);
    explicit MaxV2Operation(const std::string &name, AclNNMaxV2Param param);
    ~MaxV2Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                           atb::SVector<atb::TensorDesc> &outTensorDesc) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
private:
    AclNNMaxV2Param param_;
    aclIntArray* dims = nullptr;
};
} // namespace common
} // namespace atb_speed

#endif // ATB_SPEED_PLUGIN_ACLNN_MAX_V2_OPERATION_H
