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

#ifndef ATB_SPEED_PLUGIN_ACLNN_FUSED_INFER_ATTENTION_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_FUSED_INFER_ATTENTION_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "attn_operation.h"

namespace atb_speed {
namespace common {

class FusedInferAttentionOperation : public AclNNOperation {
public:
    explicit FusedInferAttentionOperation(const std::string &name, AclNNAttnParam param);
    ~FusedInferAttentionOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

private:
    AclNNAttnParam param_;
    aclTensor *tensorsOfValue[1]{nullptr};
    aclTensor *tensorsOfKey[1]{nullptr};
    const int aclnnTensorIndex[7] = {0, 0, 0, 4, 5, 6, 14};
    const int aclnnTensorListIndex[7] = {-1, 1, 2, -1, -1, -1, -1};
};
} // namespace common
} // namespace atb_speed
#endif