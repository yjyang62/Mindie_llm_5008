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
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/aclnn_repeat.h"
#include "repeat_operation.h"

namespace atb_speed::common {

RepeatOperation::RepeatOperation(
    const std::string &name,
    atb_speed::common::AclNNRepeatParam param
) : AclNNOperation(name), param_(param)
{
    this->opName_ = name;
    this->param_ = param;
}

RepeatOperation::~RepeatOperation()
{
    ATB_SPEED_LOG_DEBUG("RepeatOperation deconstruct");
    this->DestroyOperation();
    if (this->repeats_ != nullptr) {
        aclDestroyIntArray(this->repeats_);
        this->repeats_ = nullptr;
    }
}

/**
    *
    * @param[in] inTensorDesc: dimNum <= 8,
    * @param[in] outTensorDesc: dimNum <= 8
    * @return atb::Status
    */
atb::Status RepeatOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "RepeatOperation infer shape start");
    if (param_.repeatsArray.size() < inTensorDescs.at(0).shape.dimNum) {
        ATB_SPEED_LOG_ERROR(opName_ << "RepeatOperation infer shape failed: \
            repeatsArray size should be equal or greater than intensor[0] size.");
        return -1;
    }
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    for (uint64_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i] * param_.repeatsArray[i];
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "RepeatOperation infer shape end"
                << " format: " << inTensorDescs.at(0).format << " dimNum: " << inTensorDescs.at(0).shape.dimNum
                << " dims: " << inTensorDescs.at(0).shape.dims[0]);
    return 0;
}


uint32_t RepeatOperation::GetInputNum() const
{
    return DIM1;
}

uint32_t RepeatOperation::GetOutputNum() const
{
    return DIM1;
}

int RepeatOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    if (repeats_ != nullptr) {
        aclDestroyIntArray(this->repeats_);
        repeats_ = nullptr;
    }
    repeats_ = aclCreateIntArray(param_.repeatsArray.data(), param_.repeatsArray.size());
    int ret = aclnnRepeatGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,     // input
        repeats_,                                         // repeatShape
        aclnnVariantPack.aclOutTensors.at(0)->tensor,    // out
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                    << ", ret: " << ret
                    << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                    << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int RepeatOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnRepeat(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end"
                    << ", ret: " << ret);
    return ret;
}
}  // namespace atb_speed::common