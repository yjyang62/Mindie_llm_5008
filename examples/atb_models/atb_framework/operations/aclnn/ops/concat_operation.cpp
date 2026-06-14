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
#include "concat_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_cat.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"

namespace atb_speed {
namespace common {

ConcatOperation::ConcatOperation(const std::string &name, atb_speed::common::AclNNConcatParam param)
    : AclNNOperation(name), param_(param)
{
    CheckParamRange(param_.dim, 0, 8); // 8: max dim
}

ConcatOperation::~ConcatOperation()
{
    ATB_SPEED_LOG_DEBUG("ConcatOperation deconstruct");
    this->DestroyOperation();
}

uint32_t ConcatOperation::GetInputNum() const { return NUM2; }

uint32_t ConcatOperation::GetOutputNum() const { return NUM1; }

atb::Status ConcatOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                        atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ConcatOperation infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
    outTensorDescs.at(0).shape.dims[this->param_.dim] = inTensorDescs.at(0).shape.dims[this->param_.dim] + \
        inTensorDescs.at(1).shape.dims[this->param_.dim];
    ATB_SPEED_LOG_DEBUG(opName_ << "ConcatOperation InferShape end");
    return atb::NO_ERROR;
}

atb::Status ConcatOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
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

atb::Status ConcatOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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

int ConcatOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    std::vector<aclTensor *> tmp{aclnnVariantPack.aclInTensors.at(0)->tensor, \
                                    aclnnVariantPack.aclInTensors.at(1)->tensor};
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    int ret = aclnnCatGetWorkspaceSize(tensorList, this->param_.dim,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret: " << ret
        << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
        << ", aclExecutor address: " << &this->aclnnOpCache_->aclExecutor
        << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int ConcatOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnCat(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}

}
}