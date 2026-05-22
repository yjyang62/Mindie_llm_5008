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

#ifndef ATB_SPEED_PLUGIN_ACLNN_SPLIT_WITH_SIZE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_SPLIT_WITH_SIZE_OPERATION_H
#include "aclnnop/aclnn_split_with_size.h"
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "operations/aclnn/core/acl_nn_tensor.h"


namespace atb_speed {
namespace common {

struct AclNNSplitWithSizeParam {
    int64_t dim = 0;
    uint64_t num = 1;
};

class SplitWithSizeOperation : public AclNNOperation {
public:
    explicit SplitWithSizeOperation(const std::string &name, AclNNSplitWithSizeParam param);
    ~SplitWithSizeOperation() override;
    uint32_t GetInputNum() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    AclNNSplitWithSizeParam param_;
    std::vector<aclTensor *> outputTensorVector;
    aclIntArray* splitSizeIntArray = nullptr;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_SPLIT_WITH_SIZE_OPERATION_Hs