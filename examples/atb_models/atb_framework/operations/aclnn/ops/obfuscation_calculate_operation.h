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

#ifndef ATB_SPEED_PLUGIN_ACLNN_OBFUSCATION_CALCULATE_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_OBFUSCATION_CALCULATE_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

struct ObfuscationCalculateParam {
    int32_t fd = 0;
    int32_t cmd = 1;
    uint32_t hiddenSizePerRank = 0;
    float obfCoefficient = 1.0; // 混淆因子
};

class ObfuscationCalculateOperation : public AclNNOperation {
public:
    explicit ObfuscationCalculateOperation(const std::string &name, ObfuscationCalculateParam param);
    ~ObfuscationCalculateOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    ObfuscationCalculateParam param_;
};
} // namespace common
} // namespace atb_speed
#endif