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
#include "aclnnop/aclnn_add_rms_norm_quant_v2.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "add_rms_norm_quant_operation.h"

namespace atb_speed {
namespace common {

const double EPSILON_THRESHOLD = 1e-9; // 定义一个很小的阈值

AddRmsNormQuantOperation::AddRmsNormQuantOperation(
    const std::string &name,
    AclNNAddNormQuantMatmulParam param) : AclNNOperation(name), param_(param)
{
    opName_ = name;
    if (std::abs(param.epsilon) > EPSILON_THRESHOLD) {
        epsilon_ = param.epsilon;
    }
}

AddRmsNormQuantOperation::~AddRmsNormQuantOperation()
{
    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormQuantOperation deconstructor");
    this->DestroyOperation();
}

atb::Status AddRmsNormQuantOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormQuantOperation infer shape start");
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        if (i == 0 || i == NUM1) {  // y1Out、y2Out输出dtype固定为INT8
            outTensorDescs.at(i).dtype = aclDataType::ACL_INT8;
        } else {  // xOut同x1输入的dtype
            outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        }

        // 输出支持1-8维, shape 同x1, x2
        outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(1).shape.dimNum;
        for (size_t j = 0; j < outTensorDescs.at(i).shape.dimNum; j++) {
            outTensorDescs.at(i).shape.dims[j] = inTensorDescs.at(1).shape.dims[j];
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << "AddRmsNormQuantOperation infer shape end");
    return 0;
}

uint32_t AddRmsNormQuantOperation::GetInputNum() const
{
    uint32_t inputNum = 5;
    ATB_SPEED_LOG_DEBUG("initial inputNum: " << inputNum);
    if (param_.hasBias) {
        ATB_SPEED_LOG_DEBUG("AddRmsNormQuant & hasbias");
        ++inputNum;
    }
    return inputNum;
}

uint32_t AddRmsNormQuantOperation::GetOutputNum() const { return NUM3; }

atb::Status AddRmsNormQuantOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclTensor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    // Create aclInTensor
    aclnnVariantPack.aclInTensors.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = i;
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        atb::Tensor atbTensor = variantPack.inTensors.at(i);

        aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);

        atb::Dims viewDims = atbTensor.desc.shape;
        if (i == NUM4) {  // zeroPoints1Optional fp16为:DT_INT32, bf16为:DT_BFLOAT16
            // tensorIdx与算子14个入参的idx一一对应, i只与外部输入的inTensors(5个)一致;
            // 如果inTensors前有nullptr入参, 则要注意idx值与i值的匹配关系(不能tensorIdx有值,但算子入参给的是nullptr)
            aclnnTensor->tensorIdx = NUM5;
        }
        aclnnTensor->tensor = aclCreateTensor(
            viewDims.dims, atbTensor.desc.shape.dimNum, atbTensor.desc.dtype,
            aclnnTensor->strides.data(), 0, atbTensor.desc.format, viewDims.dims,
            atbTensor.desc.shape.dimNum, atbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " Create aclInTensor end");

    return atb::NO_ERROR;
}

int AddRmsNormQuantOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantGetWorkspaceSize start");
    uint32_t inputIdx = 5;
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclTensor* biasTensor = param_.hasBias ? aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor : nullptr;
    int ret = aclnnAddRmsNormQuantV2GetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,  // x1
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // x2
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // gamma(weight)
        aclnnVariantPack.aclInTensors.at(3)->tensor,  // scales1
        nullptr,  // scales2Optional  -> 实际未使用
        aclnnVariantPack.aclInTensors.at(4)->tensor,  // zeroPoints1Optional
        nullptr,  // zeroPoints2Optional  -> 实际未使用
        biasTensor,
        -1,
        epsilon_,  // epsilonOptional
        true,  // divMode
        aclnnVariantPack.aclOutTensors.at(0)->tensor,  // y1Out
        aclnnVariantPack.aclOutTensors.at(1)->tensor,  // y2Out, shape为1, 内容无所谓
        aclnnVariantPack.aclOutTensors.at(2)->tensor,  // xOut
        nullptr,
        &this->aclnnOpCache_->workspaceSize,  // workspaceSize
        &this->aclnnOpCache_->aclExecutor);   // executor
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantV2GetWorkspaceSize end, ret:" << ret
        << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize << ", aclExecutor:"
        << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int AddRmsNormQuantOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantV2 start");
    int ret = aclnnAddRmsNormQuantV2(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnAddRmsNormQuantV2 end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed