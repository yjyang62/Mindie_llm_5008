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

#ifndef ATB_SPEED_PLUGIN_ACLNN_CAST_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_CAST_OPERATION_H

#include <string>
#include "acl/acl.h"
#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

struct AclNNCastParam {
    aclDataType dtype;
};

class CastOperation : public AclNNOperation {
public:
    explicit CastOperation(const std::string &name, AclNNCastParam param);
    ~CastOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    AclNNCastParam param_;
};

} // namespace common
} // namespace atb_speed

#endif // ATB_SPEED_PLUGIN_ACLNN_CAST_OPERATION_H