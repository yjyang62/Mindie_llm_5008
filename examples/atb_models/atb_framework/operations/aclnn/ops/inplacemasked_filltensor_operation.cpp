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
#include "inplacemasked_filltensor_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/aclnn_masked_fill_scalar.h"

namespace atb_speed::common {

InplaceMaskedFillTensorOperation::InplaceMaskedFillTensorOperation(
    const std::string &name,
    atb_speed::common::InplaceMaskedFillTensorParam param
) : AclNNOperation(name), param_(param)
{
    this->opName_ = name;
    this->param_ = param;
}

InplaceMaskedFillTensorOperation::~InplaceMaskedFillTensorOperation()
{
    ATB_SPEED_LOG_DEBUG("InplaceMaskedFillTensorOperation deconstruct");
    this->DestroyOperation();
    if (this->value_ != nullptr) {
        aclDestroyScalar(this->value_);
        this->value_ = nullptr;
    }
}

atb::Status InplaceMaskedFillTensorOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "InplaceMaskedFillTensorOperation infer shape start");
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    for (uint64_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[i];
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "InplaceMaskedFillTensorOperation infer shape end"
                << " format: " << inTensorDescs.at(0).format << " dimNum: " << inTensorDescs.at(0).shape.dimNum
                << " dims: " << inTensorDescs.at(0).shape.dims[0]);
    return atb::NO_ERROR;
}


uint32_t InplaceMaskedFillTensorOperation::GetInputNum() const
{
    return DIM2;
}

uint32_t InplaceMaskedFillTensorOperation::GetOutputNum() const
{
    return DIM1;
}

atb::Status InplaceMaskedFillTensorOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());

    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = variantPack.inTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        if (i == 1) {
            squeezedAtbTensor.desc.dtype = aclDataType::ACL_BOOL;
        }
        CallAclCreateTensor(squeezedAtbTensor.desc.shape, squeezedAtbTensor.desc.shape,
            squeezedAtbTensor, aclnnTensor);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int InplaceMaskedFillTensorOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    if (this->value_ != nullptr) {
        aclDestroyScalar(this->value_);
        this->value_ = nullptr;
    }
    this->value_ = aclCreateScalar(&param_.value, param_.outDataType);
    int ret = aclnnInplaceMaskedFillScalarGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,     // input
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,   // input
        this->value_,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                    << ", ret: " << ret
                    << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                    << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int InplaceMaskedFillTensorOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnInplaceMaskedFillScalar(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end"
                    << ", ret: " << ret);
    return ret;
}
}  // namespace atb_speed::common