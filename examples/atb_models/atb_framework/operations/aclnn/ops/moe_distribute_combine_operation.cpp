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
#include "moe_distribute_combine_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <atb/types.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_distribute_combine.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

MoeDistributeCombineOperation::MoeDistributeCombineOperation(
    const std::string &name, MoeDistributeCombineParam param) : AclNNOperation(name), param_(param) {}
MoeDistributeCombineOperation::~MoeDistributeCombineOperation() {}

atb::Status MoeDistributeCombineOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeDistributeCombineOperation infer shape start");

    outTensorDescs.at(DIM0).format = inTensorDescs.at(DIM0).format;
    outTensorDescs.at(DIM0).dtype = inTensorDescs.at(DIM0).dtype;
    outTensorDescs.at(DIM0).shape.dimNum = inTensorDescs.at(DIM0).shape.dimNum;

    ATB_SPEED_LOG_DEBUG(opName_
                  << "MoeDistributeCombineOperation infer shape origin inTensorDescs.at(DIM1).shape.dims[DIM0]"
                  << inTensorDescs.at(DIM1).shape.dims[DIM0]);
    ATB_SPEED_LOG_DEBUG(opName_
                  << "MoeDistributeCombineOperation infer shape origin inTensorDescs.at(DIM0).shape.dims[DIM1]"
                  << inTensorDescs.at(DIM0).shape.dims[DIM1]);
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM1).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
    ATB_SPEED_LOG_DEBUG(opName_ << "MoeDistributeCombineOperation infer shape end");
    return 0;
}
uint32_t MoeDistributeCombineOperation::GetInputNum() const
{
    return NUM7;
}

uint32_t MoeDistributeCombineOperation::GetOutputNum() const
{
    return DIM1;
}

int32_t MoeDistributeCombineOperation::GetGlobalBS(const atb::TensorDesc &inTensorDesc) const
{
    int32_t worldSize = param_.epRankSize * std::max(param_.tpRankSize, 1);
    if (param_.globalBS > 0) {
        return param_.globalBS;
    }
    int32_t maxDecodeDpTokenSize = param_.maxDecodeDpTokenSize;
    // if param_.maxDecodeDpTokenSize is not available，use in_padding_idx's DIM0
    if (maxDecodeDpTokenSize == 0) {
        maxDecodeDpTokenSize = inTensorDesc.shape.dims[DIM0] / \
            CheckPositive(std::min(param_.localMoeExpertNum, param_.topk)) / CheckPositive(worldSize);
    }
    return maxDecodeDpTokenSize * worldSize;
}

int MoeDistributeCombineOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeCombineOperation start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    
    aclnnVariantPack.aclInTensors.at(NUM6)->tensorIdx = NUM10;
    int64_t globalBS = GetGlobalBS(aclnnVariantPack.aclInTensors.at(DIM0)->atbTensor.desc);
    int ret = aclnnMoeDistributeCombineGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
        aclnnVariantPack.aclInTensors.at(DIM2)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM3)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM4)->tensor,
        aclnnVariantPack.aclInTensors.at(NUM5)->tensor,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        aclnnVariantPack.aclInTensors.at(NUM6)->tensor,
        param_.epCommName.data(),
        param_.epRankSize,
        param_.epRankId,
        param_.moeExpertNum,
        param_.tpCommName.data(),
        param_.tpRankSize,
        param_.tpRankId,
        param_.expertSharedType,
        1,
        param_.sharedExpertRankNum,
        globalBS,
        0,
        param_.commQuantMode,
        0,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:" << ret
                  << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                  << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int MoeDistributeCombineOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeCombineOperation start");

    int ret = aclnnMoeDistributeCombine(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " MoeDistributeCombineOperation end, ret:" << ret);
    return ret;
}

}  // namespace common
}  // namespace atb_speed