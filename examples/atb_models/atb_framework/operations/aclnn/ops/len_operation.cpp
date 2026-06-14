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
#include "len_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_range.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

LenOperation::LenOperation(const std::string &name) : AclNNOperation(name) {}

LenOperation::~LenOperation()
{
    ATB_SPEED_LOG_DEBUG("LenOperation deconstruct");
    this->DestroyOperation();
    if (this->start_ != nullptr) {
        aclDestroyScalar(this->start_);
        this->start_ = nullptr;
    }
    if (this->end_ != nullptr) {
        aclDestroyScalar(this->end_);
        this->end_ = nullptr;
    }
    if (this->step_ != nullptr) {
        aclDestroyScalar(this->step_);
        this->step_ = nullptr;
    }
}

uint32_t LenOperation::GetInputNum() const { return NUM1; }

uint32_t LenOperation::GetOutputNum() const { return NUM1; }

atb::Status LenOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDesc,
    atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " LenOperation infer shape start");
    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = aclDataType::ACL_INT32;
    outTensorDesc.at(0).shape.dimNum = 1;
    outTensorDesc.at(0).shape.dims[0] = 1;
    ATB_SPEED_LOG_DEBUG(opName_ << "LenOperation InferShape end");

    return atb::NO_ERROR;
}

atb::Status LenOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        if (CreateTensor(variantPack.inTensors.at(i), i, aclnnVariantPack.aclInTensors[i]) != atb::NO_ERROR) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}


atb::Status LenOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        if (CreateTensor(variantPack.outTensors.at(i), i, aclnnVariantPack.aclOutTensors[i]) != atb::NO_ERROR) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " outTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

int LenOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    auto start = aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc.shape.dims[DIM0];
    auto end = start + 1;
    auto step = 1;
    if (this->start_ != nullptr) {
        aclDestroyScalar(this->start_);
        this->start_ = nullptr;
    }
    if (this->end_ != nullptr) {
        aclDestroyScalar(this->end_);
        this->end_ = nullptr;
    }
    if (this->step_ != nullptr) {
        aclDestroyScalar(this->step_);
        this->step_ = nullptr;
    }
    this->start_ = aclCreateScalar(&start, aclDataType::ACL_INT32);
    this->end_ = aclCreateScalar(&end, aclDataType::ACL_INT32);
    this->step_ = aclCreateScalar(&step, aclDataType::ACL_INT32);
    int ret = aclnnRangeGetWorkspaceSize(
        this->start_,
        this->end_,
        this->step_,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(
        opName_ << " SetAclNNWorkspaceExecutor end"
                << ", ret: " << ret
                << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor
    );
    return ret;
}

int LenOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnRange(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed