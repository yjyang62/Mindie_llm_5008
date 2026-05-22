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
#include "acl/acl.h"
#include "aclnnop/aclnn_obfuscation_setup_v2.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "obfuscation_setup_operation.h"

namespace atb_speed {
namespace common {

ObfuscationSetupOperation::ObfuscationSetupOperation(const std::string &name,
    ObfuscationSetupParam param) : AclNNOperation(name), param_(param) {}

ObfuscationSetupOperation:: ~ObfuscationSetupOperation()
{
    ATB_SPEED_LOG_DEBUG("ObfuscationSetupOperation deconstructor");
    this->DestroyOperation();
}

atb::Status ObfuscationSetupOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG("ObfuscationSetupOperation infer shape start");
    if (inTensorDescs.size() != 0) {
        ATB_SPEED_LOG_ERROR("ObfuscationSetupOperation intensors should be 0, but get " <<
           inTensorDescs.size());
        return atb::ERROR_INVALID_TENSOR_SIZE;
    }
    outTensorDescs.at(DIM0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(DIM0).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM0).shape.dimNum = NUM1;
    outTensorDescs.at(DIM0).shape.dims[DIM0] = NUM1;
    ATB_SPEED_LOG_DEBUG("ObfuscationSetupOperation infer shape end");
    return 0;
}

uint32_t ObfuscationSetupOperation::GetInputNum() const { return DIM0; }

uint32_t ObfuscationSetupOperation::GetOutputNum() const { return NUM1; }

int ObfuscationSetupOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationSetupGetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    int ret = aclnnObfuscationSetupV2GetWorkspaceSize(
        param_.fdtoClose,
        param_.dataType,
        param_.hiddenSizePerRank,
        param_.tpRank,
        0,
        0,
        param_.cmd,
        param_.threadNum,
        param_.obfCoefficient,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationSetupGetWorkspaceSize end, ret:" <<
        ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize <<
        ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int ObfuscationSetupOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationSetup start");
    int ret = aclnnObfuscationSetupV2(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG("aclnnObfuscationSetup end, ret:" << ret);
    return ret;
}

} // namespace common
} // namespace atb_speed