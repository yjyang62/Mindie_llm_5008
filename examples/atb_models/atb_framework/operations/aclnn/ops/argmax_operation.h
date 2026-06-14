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

#ifndef MINDIE_LLM_PLUGIN_ACLNN_ARGMAX_OPERATION_H
#define MINDIE_LLM_PLUGIN_ACLNN_ARGMAX_OPERATION_H
#include "operations/aclnn/utils/utils.h"
#include "aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

struct AclNNArgMaxParam {
    int64_t dim = -1;
    bool keepdim = false;
};

class ArgMaxOperation : public AclNNOperation {
public:
    explicit ArgMaxOperation(const std::string &name);
    explicit ArgMaxOperation(const std::string &name, AclNNArgMaxParam param);
    ~ArgMaxOperation() override;
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
    AclNNArgMaxParam param_;
};
} // namespace common
} // namespace atb_speed

#endif // MINDIE_LLM_PLUGIN_ACLNN_ARGMAX_OPERATION_H
