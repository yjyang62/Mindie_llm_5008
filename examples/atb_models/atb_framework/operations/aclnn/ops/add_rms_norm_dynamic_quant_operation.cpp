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
#include "aclnnop/aclnn_add_rms_norm_dynamic_quant_v2.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "add_rms_norm_dynamic_quant_operation.h"

namespace atb_speed {
namespace common {

const double EPSILON_THRESHOLD = 1e-9; // 定义一个很小的阈值

AddRmsNormDynamicQuantOperation::AddRmsNormDynamicQuantOperation(
    const std::string &name, AclNNAddNormDynamicQuantMatmulParam param) : AclNNOperation(name), param_(param)
{
    opName_ = name;
    if (std::abs(param.epsilon) > EPSILON_THRESHOLD) {
        epsilon_ = param.epsilon;
    }
}

AddRmsNormDynamicQuantOperation::~AddRmsNormDynamicQuantOperation()
{
    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormDynamicQuantOperation deconstructor");
    this->DestroyOperation();
}

atb::Status AddRmsNormDynamicQuantOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormDynamicQuantOperation infer shape start");
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        if (i == 0 || i == NUM1) {  // y1Out、y2Out输出dtype固定为INT8
            outTensorDescs.at(i).dtype = aclDataType::ACL_INT8;
        } else if (i == NUM3 || i == NUM4) {  // scale1Out、scale2Out：FLOAT32
            outTensorDescs.at(i).dtype = aclDataType::ACL_FLOAT;
        } else {  // xOut同x1输入的dtype
            outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        }
        // 不输入任何 smoothScale场景
        // y2Out、scale2Out 搞个1维即可, 占位, 内容无所谓
        if (i == NUM1 || i == NUM4) {
            outTensorDescs.at(i).shape.dimNum = NUM1;
            outTensorDescs.at(i).shape.dims[0] = 1;
        } else if (i < NUM3) {
            // y1Out、xOut输出支持2-8维, shape 同x1, x2
            outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(1).shape.dimNum;
            for (size_t j = 0; j < outTensorDescs.at(i).shape.dimNum; j++) {
                outTensorDescs.at(i).shape.dims[j] = inTensorDescs.at(1).shape.dims[j];
            }
        } else {
            // scale1Out：shape维度为x的shape剔除最后一维
            outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(1).shape.dimNum - 1;
            for (size_t j = 0; j < outTensorDescs.at(i).shape.dimNum; j++) {
                outTensorDescs.at(i).shape.dims[j] = inTensorDescs.at(1).shape.dims[j];
            }
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormDynamicQuantOperation infer shape end");
    return 0;
}

uint32_t AddRmsNormDynamicQuantOperation::GetInputNum() const
{
    uint32_t inputNum = 3;
    ATB_SPEED_LOG_DEBUG("initial inputNum: " << inputNum);
    if (param_.hasBias) {
        ATB_SPEED_LOG_DEBUG("AddRmsNormQuant & hasbias");
        ++inputNum;
    }
    return inputNum;
}

uint32_t AddRmsNormDynamicQuantOperation::GetOutputNum() const { return NUM5; }

int AddRmsNormDynamicQuantOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantV2GetWorkspaceSize start");
    uint32_t inputIdx = 3;
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclTensor* biasTensor = param_.hasBias ? aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;
    // 不输入任何 smoothScale场景
    int ret = aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,  // x1
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // x2
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // gamma(weight)
        nullptr,  // smoothScale1Optional
        nullptr,  // smoothScale2Optional
        biasTensor,
        epsilon_,  // epsilonOptional
        // 算子之后会将两个divmod改成列表，后续同步更新
        nullptr,
        aclnnVariantPack.aclOutTensors.at(0)->tensor,  // y1Out
        aclnnVariantPack.aclOutTensors.at(1)->tensor,  // y2Out, shape为1, 占位, 内容无所谓
        aclnnVariantPack.aclOutTensors.at(2)->tensor,  // xOut
        aclnnVariantPack.aclOutTensors.at(3)->tensor,  // scale1Out
        aclnnVariantPack.aclOutTensors.at(4)->tensor,  // scale2Out, shape为1, 占位, 内容无所谓
        &this->aclnnOpCache_->workspaceSize,  // workspaceSize
        &this->aclnnOpCache_->aclExecutor);   // executor
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantV2GetWorkspaceSize end, ret:" << ret
        << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize << ", aclExecutor:"
        << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int AddRmsNormDynamicQuantOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormDynamicQuantV2 start");
    int ret = aclnnAddRmsNormDynamicQuantV2(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormDynamicQuantV2 end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed
