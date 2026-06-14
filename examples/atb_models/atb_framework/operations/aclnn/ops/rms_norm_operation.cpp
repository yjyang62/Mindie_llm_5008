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
#include "aclnnop/aclnn_rms_norm.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "rms_norm_operation.h"

namespace atb_speed {
namespace common {

RmsNormOperation::RmsNormOperation(const std::string &name, float epsilon) : AclNNOperation(name)
{
    this->opName_ = name;
    this->epsilon = epsilon;
}

atb::Status RmsNormOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                         atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        if (i == NUM1) {
            outTensorDescs.at(i).dtype = aclDataType::ACL_FLOAT;
        } else {
            outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        }

        outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

        if (inTensorDescs.at(0).shape.dimNum == DIM3) {
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK aclnn rmsnorm inputs shape: [input0]"
                           << inTensorDescs.at(0).shape.dims[DIM0] << ", " << inTensorDescs.at(0).shape.dims[DIM1]
                           << ", " << inTensorDescs.at(0).shape.dims[DIM2]);
            outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
            outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
            outTensorDescs.at(i).shape.dims[DIM2] = inTensorDescs.at(0).shape.dims[DIM2];
        } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK aclnn rmsnorm inputs shape: [input0]"
                           << inTensorDescs.at(0).shape.dims[DIM0] << ", "
                           << inTensorDescs.at(0).shape.dims[DIM1]);
            outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
            outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
        } else {
            ATB_SPEED_LOG_ERROR(opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum);
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return 0;
}

uint32_t RmsNormOperation::GetInputNum() const { return NUM2; }

uint32_t RmsNormOperation::GetOutputNum() const { return NUM2; }

int RmsNormOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnRmsNormGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnRmsNormGetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,
        aclnnVariantPack.aclInTensors.at(1)->tensor,
        this->epsilon,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        aclnnVariantPack.aclOutTensors.at(1)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnRmsNormGetWorkspaceSize end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize << ", aclExecutor:"
                  << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int RmsNormOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnRmsNorm start");
    int ret = aclnnRmsNorm(workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnRmsNorm end, ret:" << ret);
    return ret;
}

} // namespace common
} // namespace atb_speed