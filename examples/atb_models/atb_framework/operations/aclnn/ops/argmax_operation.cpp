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
#include "argmax_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_argmax.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace common {

ArgMaxOperation::ArgMaxOperation(const std::string &name) : AclNNOperation(name) {}

ArgMaxOperation::ArgMaxOperation(const std::string &name, atb_speed::common::AclNNArgMaxParam param)
    : AclNNOperation(name), param_(param)
{
}

ArgMaxOperation::~ArgMaxOperation()
{
    ATB_SPEED_LOG_DEBUG("ArgMaxOperation deconstruct");
    this->DestroyOperation();
}

uint32_t ArgMaxOperation::GetInputNum() const { return NUM1; }

uint32_t ArgMaxOperation::GetOutputNum() const { return NUM1; }

atb::Status ArgMaxOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                        atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ArgMaxOperation infer shape start");
    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(0).dtype = ACL_INT32;
    uint32_t inputDimNum = inTensorDesc.at(0).shape.dimNum;
    uint32_t outputDimNum = inputDimNum;
    uint32_t realDim = this->param_.dim < 0 ? this->param_.dim + inputDimNum : this->param_.dim;

    if (!param_.keepdim) {
        outputDimNum -= 1;
    }
    outTensorDesc.at(0).shape.dimNum = outputDimNum;

    uint32_t j = 0;
    for (uint32_t i = 0; i < outputDimNum; ++i) {
        if (i == realDim && param_.keepdim) {
            outTensorDesc.at(0).shape.dims[i] = 1;
            j++;
        } else {
            outTensorDesc.at(0).shape.dims[j++] = inTensorDesc.at(0).shape.dims[i];
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "ArgMaxOperation InferShape end");

    return atb::NO_ERROR;
}

atb::Status ArgMaxOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
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


atb::Status ArgMaxOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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

int ArgMaxOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnArgMaxGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor, this->param_.dim,
                                          this->param_.keepdim, aclnnVariantPack.aclOutTensors.at(0)->tensor,
                                          &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                   << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                   << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int ArgMaxOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnArgMax(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed