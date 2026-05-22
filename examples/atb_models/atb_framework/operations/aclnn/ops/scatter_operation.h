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

#ifndef ATB_SPEED_PLUGIN_ACLNN_SCATTER_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_SCATTER_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

enum class ReduceType {
    REPLACE = 0,
    ADD = 1,
    MULTIPLY = 2
};

struct AclNNScatterParam {
    int64_t dim = 0;
    ReduceType reduce = ReduceType::REPLACE;
};

class ScatterOperation : public AclNNOperation {
public:
    explicit ScatterOperation(const std::string &name, AclNNScatterParam param, bool isInplace);
    ~ScatterOperation() override;
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
    AclNNScatterParam param_;
    bool isInplace_;
};
} // namespace common
} // namespace atb_speed
#endif // ATB_SPEED_PLUGIN_ACLNN_SCATTER_OPERATION_H