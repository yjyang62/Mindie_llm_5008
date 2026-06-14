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
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v2.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"
#include "moe_init_routing_operation.h"

namespace atb_speed {
namespace common {

MoeInitRoutingOperation::MoeInitRoutingOperation(
    const std::string &name, MoeInitRoutingParam param) : AclNNOperation(name), param_(param) {}
MoeInitRoutingOperation::~MoeInitRoutingOperation() {}

atb::Status MoeInitRoutingOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeInitRoutingOperation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    outTensorDescs.at(DIM1).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM1).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM1).shape.dimNum = DIM1;

    outTensorDescs.at(DIM2).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM2).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM2).shape.dimNum = DIM1;

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeInitRoutingOperation infer shape origin inTensorDescs.at(DIM0).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM0]);
    int inputSize = inTensorDescs.at(DIM0).shape.dims[DIM0]; // 输入 token 数
    int outputSize = GetMoeInitRoutingOpOutputSize(inputSize, param_);
    outTensorDescs.at(DIM0).shape.dims[DIM0] = outputSize;
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
    outTensorDescs.at(DIM1).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0] * param_.topkNum;
    outTensorDescs.at(DIM2).shape.dims[DIM0] = param_.expertNum;

    ATB_SPEED_LOG_DEBUG(opName_ << "MoeInitRoutingOperation infer shape end");
    return 0;
}
uint32_t MoeInitRoutingOperation::GetInputNum() const
{
    return DIM2;
}

uint32_t MoeInitRoutingOperation::GetOutputNum() const
{
    return DIM3;
}

int MoeInitRoutingOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeInitRoutingOperation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int inputSize =  aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc.shape.dims[DIM0]; // 输入 token 数
    int outputSize = GetMoeInitRoutingOpOutputSize(inputSize, param_);
    int ret = aclnnMoeInitRoutingV2GetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
        outputSize,
        0, param_.expertNum, 0, param_.expertTokensCoutOrCumsumFlag, false,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM2)->tensor,
        nullptr,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeInitRoutingOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeInitRoutingOperation start");

    int ret = aclnnMoeInitRoutingV2(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeInitRoutingOperation end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed