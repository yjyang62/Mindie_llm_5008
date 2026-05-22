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

#ifndef ATB_SPEED_PLUGIN_ACLNN_OBFUSCATION_SETUP_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_OBFUSCATION_SETUP_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed {
namespace common {

struct ObfuscationSetupParam {
    int32_t fdtoClose = 0;
    int32_t dataType = 1; // 0: float32; 1: float16; 27: bfloat16
    int32_t hiddenSizePerRank = 1;
    int32_t tpRank = 0;
    int32_t cmd = 1; // 1: Normal mode; 3: Exit mode
    int32_t threadNum = 6; // thread num in aicpu
    float obfCoefficient = 1.0; // 混淆因子
};

class ObfuscationSetupOperation : public AclNNOperation {
public:
    explicit ObfuscationSetupOperation(const std::string &name, ObfuscationSetupParam param);
    ~ObfuscationSetupOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    ObfuscationSetupParam param_;
};
} // namespace common
} // namespace atb_speed
#endif