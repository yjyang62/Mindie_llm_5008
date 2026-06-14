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
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "operations/aclnn/utils/utils.h"
#include "split_with_size_operation.h"

namespace atb_speed {
namespace common {

SplitWithSizeOperation::SplitWithSizeOperation(
    const std::string &name,
    AclNNSplitWithSizeParam param) : AclNNOperation(name), param_(param)
{
    outputTensorVector.resize(param.num);
}

SplitWithSizeOperation::~SplitWithSizeOperation()
{
    int ret = aclDestroyIntArray(splitSizeIntArray);
    if (ret > 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " call aclDestroyIntArray failed.");
    }
}

atb::Status SplitWithSizeOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "SplitWithSizeOperation infer shape start");
    if (param_.num <= 0) {
        ATB_SPEED_LOG_ERROR(opName_ << "SplitWithSizeOperation infer shape failed, param.num must be greater than 0");
        return atb::ERROR_INVALID_PARAM;
    }
    if (param_.dim < 0 || param_.dim >= static_cast<int64_t>(inTensorDescs.at(DIM0).shape.dimNum)) {
        ATB_SPEED_LOG_ERROR(opName_ << "SplitWithSizeOperation infer shape failed, " <<
            "param.dim must be greater than or equal to 0 and less than " << inTensorDescs.at(DIM0).shape.dimNum);
        return atb::ERROR_INVALID_PARAM;
    }
    int splitSize = inTensorDescs.at(DIM0).shape.dims[param_.dim] / param_.num;
    int remainSize = inTensorDescs.at(DIM0).shape.dims[param_.dim] % param_.num;
    for (size_t i = 0; i < GetOutputNum(); i++) {
        outTensorDescs.at(i) = inTensorDescs.at(DIM0);
        if (i < static_cast<size_t>(remainSize)) {
            outTensorDescs.at(i).shape.dims[param_.dim] = splitSize + 1;
        } else {
            outTensorDescs.at(i).shape.dims[param_.dim] = splitSize;
        }
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "SplitWithSizeOperation infer shape end");
    return 0;
}

uint32_t SplitWithSizeOperation::GetInputNum() const
{
    // 1: aclInTensors size
    return 1;
}

uint32_t SplitWithSizeOperation::GetOutputNum() const
{
    return param_.num;
}

atb::Status SplitWithSizeOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.inTensors.at(i));
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            squeezedAtbTensor.desc.shape.dims, squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
            squeezedAtbTensor.desc.format, squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum, squeezedAtbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status SplitWithSizeOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor squeezedAtbTensor = SqueezeBatchSeq(variantPack.outTensors.at(i));
        aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            squeezedAtbTensor.desc.shape.dims, squeezedAtbTensor.desc.shape.dimNum,
            squeezedAtbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
            squeezedAtbTensor.desc.format, squeezedAtbTensor.desc.shape.dims,
            squeezedAtbTensor.desc.shape.dimNum, squeezedAtbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); i++) {
        outputTensorVector[i] = aclnnVariantPack.aclOutTensors.at(i)->tensor;
    }
    return atb::NO_ERROR;
}

int SplitWithSizeOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclTensorList *out = aclCreateTensorList(outputTensorVector.data(), outputTensorVector.size());

    std::vector<int64_t> splitSizeVec;
    int splitSize = aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc.shape.dims[param_.dim] / param_.num;
    int remainSize = aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc.shape.dims[param_.dim] % param_.num;
    for (size_t i = 0; i < GetOutputNum(); i++) {
        if (i < static_cast<size_t>(remainSize)) {
            splitSizeVec.emplace_back(splitSize + 1);
        } else {
            splitSizeVec.emplace_back(splitSize);
        }
    }
    if (splitSizeIntArray) {
        int destroyRet = aclDestroyIntArray(splitSizeIntArray);
        if (destroyRet > 0) {
            ATB_SPEED_LOG_ERROR(opName_ << " call aclDestroyIntArray failed.");
            return destroyRet;
        }
    }
    splitSizeIntArray = aclCreateIntArray(splitSizeVec.data(), splitSizeVec.size());
    int ret = aclnnSplitWithSizeGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor, splitSizeIntArray, param_.dim, out,
        &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int SplitWithSizeOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnSplitWithSize start");
    int ret = aclnnSplitWithSize(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnSplitWithSize end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed