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
#include "acl/acl.h"
#include "aclnnop/aclnn_obfuscation_calculate_v2.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "obfuscation_calculate_operation.h"

namespace atb_speed {
namespace common {

ObfuscationCalculateOperation::ObfuscationCalculateOperation(
    const std::string &name, ObfuscationCalculateParam param) : AclNNOperation(name), param_(param) {}

ObfuscationCalculateOperation:: ~ObfuscationCalculateOperation()
{
    ATB_SPEED_LOG_DEBUG("ObfuscationCalculateOperation deconstructor");
    this->DestroyOperation();
}

atb::Status ObfuscationCalculateOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG("ObfuscationCalculateOperation infer shape start");
    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;
    for (uint32_t i = 0; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(DIM0).shape.dims[i] = inTensorDescs.at(DIM0).shape.dims[i];
    }
    ATB_SPEED_LOG_DEBUG("ObfuscationCalculateOperation infer shape end");
    return 0;
}

uint32_t ObfuscationCalculateOperation::GetInputNum() const { return NUM1; }

uint32_t ObfuscationCalculateOperation::GetOutputNum() const { return NUM1; }

atb::Status ObfuscationCalculateOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    uint32_t inputNum = GetInputNum();
    aclnnVariantPack.aclInTensors.resize(inputNum);
    atb::Tensor atbTensor = variantPack.inTensors.at(0);
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorIdx = 1;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    aclnnTensor->tensor = aclCreateTensor(
        atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum,
        atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    aclnnVariantPack.aclInTensors.at(0) = aclnnTensor;
    return atb::NO_ERROR;
}

int ObfuscationCalculateOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationCalculateGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnObfuscationCalculateV2GetWorkspaceSize(
        param_.fd,
        aclnnVariantPack.aclInTensors.at(0)->tensor,
        param_.hiddenSizePerRank,
        param_.cmd,
        param_.obfCoefficient,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationCalculateGetWorkspaceSize end, ret:" <<
        ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize <<
        ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int ObfuscationCalculateOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationCalculate start");
    int ret = aclnnObfuscationCalculateV2(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationCalculate end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed