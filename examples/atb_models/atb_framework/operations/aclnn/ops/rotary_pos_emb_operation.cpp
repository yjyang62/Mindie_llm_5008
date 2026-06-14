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
#include <unistd.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "rotary_pos_emb_operation.h"

namespace atb_speed {
namespace common {


RotaryPosEmbOperation::RotaryPosEmbOperation(const std::string &name) : AclNNOperation(name)
{
    this->opName_ = name;
}

RotaryPosEmbOperation::~RotaryPosEmbOperation()
{
}

uint32_t RotaryPosEmbOperation::GetInputNum() const
{
    return NUM4;
}

uint32_t RotaryPosEmbOperation::GetOutputNum() const
{
    return 0;
}

atb::Status RotaryPosEmbOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "RotaryPosEmbOperation infer shape start");
    for (uint64_t i = 0; i < outTensorDescs.size(); ++i) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        if (inTensorDescs.at(0).shape.dimNum == NUM4 && inTensorDescs.at(0).shape.dims[2] == 1) { // 2: [B,S,N,D]
            outTensorDescs.at(i).shape.dimNum = 3; // 3: [BS, N, D]
            outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
            outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
            outTensorDescs.at(i).shape.dims[DIM2] = inTensorDescs.at(0).shape.dims[DIM3];
        } else {
            outTensorDescs.at(i).shape = inTensorDescs.at(0).shape;
        }
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "RotaryPosEmbOperation infer shape end"
                  << " format: " << inTensorDescs.at(0).format << " dimNum: " << inTensorDescs.at(0).shape.dimNum
                  << " dims: " << inTensorDescs.at(0).shape.dims[0]);
    return 0;
}

atb::Dims RotaryPosEmbOperation::GetWeightStorageShape(const atb::TensorDesc atbTensorDesc) const
{
    atb::Dims storageTensorDims = atbTensorDesc.shape;  // ND格式下，storageShape和originalShape一致
    if (atbTensorDesc.shape.dimNum == 3) { // 3: [bs*seq, head_num, head_dim]
        storageTensorDims.dimNum = NUM4;  // 4维
        storageTensorDims.dims[DIM0] = atbTensorDesc.shape.dims[DIM0];
        storageTensorDims.dims[DIM1] = 1;
        storageTensorDims.dims[DIM2] = atbTensorDesc.shape.dims[DIM1];
        storageTensorDims.dims[DIM3] = atbTensorDesc.shape.dims[DIM2];
    } else if (atbTensorDesc.shape.dimNum == 2) { // 2: [bs*seq, head_dim]
        storageTensorDims.dimNum = NUM4;  // 4维
        storageTensorDims.dims[DIM0] = atbTensorDesc.shape.dims[DIM0];
        storageTensorDims.dims[DIM1] = 1;
        storageTensorDims.dims[DIM2] = 1;
        storageTensorDims.dims[DIM3] = atbTensorDesc.shape.dims[DIM1];
    }
    return storageTensorDims;
}

atb::Status RotaryPosEmbOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor atbTensor = variantPack.inTensors.at(i);
        atb::Dims storageTensorDims = GetWeightStorageShape(atbTensor.desc);
        aclnnTensor->strides = GetCopyTensorStride(storageTensorDims);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(storageTensorDims, storageTensorDims,
            atbTensor, aclnnTensor));
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status RotaryPosEmbOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->tensorListidx = AclNNTensor::notInTensorList;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.outTensors.at(i);
        atb::Tensor atbTensor = variantPack.outTensors.at(i);
        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
        CHECK_OPERATION_STATUS_RETURN(CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape,
            atbTensor, aclnnTensor));
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}


int RotaryPosEmbOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnApplyRotaryPosEmbGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor, // 0: query
        aclnnVariantPack.aclInTensors.at(1)->tensor, // 1: key
        aclnnVariantPack.aclInTensors.at(2)->tensor, // 2: cos table
        aclnnVariantPack.aclInTensors.at(3)->tensor, // 3: sin table
        1, // 1: 目前只支持1，代表格式为BSND的4维Tensor
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}


int RotaryPosEmbOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " RotaryPosEmbOperation start");
    int ret = aclnnApplyRotaryPosEmb(workspace, this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " RotaryPosEmbOperation end, ret:" << ret);
    return ret;
}

} // namespace common
} // namespace atb_speed