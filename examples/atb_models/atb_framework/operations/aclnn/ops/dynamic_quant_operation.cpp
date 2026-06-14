/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "dynamic_quant_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_quant.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

DynamicQuantOperation::DynamicQuantOperation(const std::string &name) : AclNNOperation(name) {}
DynamicQuantOperation::~DynamicQuantOperation() {}

atb::Status DynamicQuantOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "DynamicQuantOperation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = aclDataType::ACL_INT8;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    outTensorDescs.at(DIM1).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM1).dtype = aclDataType::ACL_FLOAT;
    outTensorDescs.at(DIM1).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum - 1;

    ATB_SPEED_LOG_DEBUG(opName_ << "DynamicQuantOperation infer shape origin inTensorDescs.at(DIM0).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM0]);

    for (uint64_t i = 0; i < inTensorDescs.at(DIM0).shape.dimNum; i++) {
        outTensorDescs.at(DIM0).shape.dims[i] = inTensorDescs.at(DIM0).shape.dims[i];
    }
    for (uint64_t i = 0; i < inTensorDescs.at(DIM0).shape.dimNum - 1; i++) {
        outTensorDescs.at(DIM1).shape.dims[i] = inTensorDescs.at(DIM0).shape.dims[i];
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "DynamicQuantOperation infer shape end");
    return 0;
}

uint32_t DynamicQuantOperation::GetInputNum() const
{
    return DIM1;
}

uint32_t DynamicQuantOperation::GetOutputNum() const
{
    return DIM2;
}

int DynamicQuantOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " DynamicQuantOperation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnDynamicQuantGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        nullptr,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int DynamicQuantOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " DynamicQuantOperation start");

    int ret = aclnnDynamicQuant(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " DynamicQuantOperation end, ret:" << ret);
    return ret;
}

}
}