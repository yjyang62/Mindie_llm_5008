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

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "attn_v3_operation.h"

namespace atb_speed {
namespace common {

FusedInferAttentionOperation::FusedInferAttentionOperation(
    const std::string &name, AclNNAttnParam param) : AclNNOperation(name), param_(param)
{}

FusedInferAttentionOperation::~FusedInferAttentionOperation()
{
    ATB_SPEED_LOG_DEBUG(opName_ << "FusedInferAttentionOperation deconstructor");
    this->DestroyOperation();
}

atb::Status FusedInferAttentionOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << "FusedInferAttentionOperation infer shape start");
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (param_.inputLayoutPA == "TND") {
        outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0]; // T
        outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1]; // N
        outTensorDescs.at(0).shape.dims[DIM2] = inTensorDescs.at(DIM0).shape.dims[DIM2]; // D
        outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    } else {
        outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0]; // B
        outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM2]; // N
        outTensorDescs.at(0).shape.dims[DIM2] = inTensorDescs.at(DIM0).shape.dims[DIM3]; // D
        outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum - 1;
    }
    ATB_SPEED_LOG_DEBUG(opName_ << "FusedInferAttentionOperation infer shape end");
    return 0;
}

uint32_t FusedInferAttentionOperation::GetInputNum() const { return 7; }

uint32_t FusedInferAttentionOperation::GetOutputNum() const { return 1; }

atb::Status FusedInferAttentionOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        aclnnTensor->tensorIdx = this->aclnnTensorIndex[i];
        aclnnTensor->tensorListidx = this->aclnnTensorListIndex[i];
        if (i == 4 || i == 5) { // 4: qlen; 5: seqlen
            aclnnTensor->needUpdateTensorDataPtr = false;
            aclnnTensor->intArrayHostData.dataSize = aclnnTensor->atbTensor.dataSize / NUM4; // int32 has 4 bytes
            aclnnTensor->intArrayHostData.data.resize(aclnnTensor->intArrayHostData.dataSize);
            aclnnTensor->intArrayHostData.dataOri.resize(aclnnTensor->intArrayHostData.dataSize);
            std::transform(
                static_cast<int32_t *>(aclnnTensor->atbTensor.hostData),
                static_cast<int32_t *>(aclnnTensor->atbTensor.hostData) + aclnnTensor->atbTensor.dataSize / NUM4,
                aclnnTensor->intArrayHostData.data.data(), [](int32_t value) {
                    return static_cast<int64_t>(value);
            });
            std::copy(static_cast<int32_t *>(aclnnTensor->atbTensor.hostData),
                static_cast<int32_t *>(aclnnTensor->atbTensor.hostData) +
                    aclnnTensor->atbTensor.dataSize / sizeof(int32_t),
                aclnnTensor->intArrayHostData.dataOri.data());
            aclnnTensor->intArrayHostData.intArray = aclCreateIntArray(
                static_cast<int64_t *>(aclnnTensor->intArrayHostData.data.data()),
                aclnnTensor->intArrayHostData.dataSize);
        } else {
            aclnnTensor->needUpdateTensorDataPtr = true;
            atb::Tensor atbTensor = variantPack.inTensors.at(i);
            aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
            CHECK_OPERATION_STATUS_RETURN(
                CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensor));
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    tensorsOfKey[0] = aclnnVariantPack.aclInTensors.at(1)->tensor;   // 1: key tensor index
    tensorsOfValue[0] = aclnnVariantPack.aclInTensors.at(2)->tensor; // 2: value tensor index
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, 1);
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, 1);
    aclnnVariantPack.aclInTensorList.clear();
    aclnnVariantPack.aclInTensorList.push_back(nullptr);
    aclnnVariantPack.aclInTensorList.push_back(tensorKeyList);
    aclnnVariantPack.aclInTensorList.push_back(tensorValueList);
    return atb::NO_ERROR;
}


int FusedInferAttentionOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnFusedInferAttentionScoreV3GetWorkspaceSize start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    double scaleValue = 1 / sqrt(param_.headDim);
    int ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(aclnnVariantPack.aclInTensors.at(0)->tensor,  // q
        aclnnVariantPack.aclInTensorList.at(1),  // k
        aclnnVariantPack.aclInTensorList.at(2),  // v
        nullptr,  // preShiftOptional
        aclnnVariantPack.aclInTensors.at(3)->tensor,  // attenMaskOptional
        aclnnVariantPack.aclInTensors.at(4)->intArrayHostData.intArray,  // actualSeqLengthsOptional
        aclnnVariantPack.aclInTensors.at(5)->intArrayHostData.intArray,  // actualSeqLengthsKvOptional
        nullptr,  // deqScale1Optional
        nullptr,  // quantScale1Optional
        nullptr,  // deqScale2Optional
        nullptr,  // quantScale2Optional
        nullptr,  // quantOffset2Optional
        nullptr,  // antiquantScaleOptional
        nullptr,  // antiquantOffsetOptional
        aclnnVariantPack.aclInTensors.at(6)->tensor,  // blockTableOptional
        nullptr,  // queryPaddingSizeOptional
        nullptr,  // kvPaddingSizeOptional
        nullptr,  // keyAntiquantScaleOptional
        nullptr,  // keyAntiquantOffsetOptional
        nullptr,  // valueAntiquantScaleOptional
        nullptr,  // valueAntiquantOffsetOptional
        nullptr,  // keySharedPrefixOptional
        nullptr,  // valueSharedPrefixOptional
        nullptr,  // actualSharedPrefixLenOptional
        nullptr,  // queryRopeOptional mla q rope
        nullptr,  // keyRopeOptional mla k rope
        nullptr,  // keyRopeAntiquantScaleOptional
        param_.headNum,  // numHeads
        scaleValue,  // scaleValue
        2147483647,  // preTokens
        2147483647,  // nextTokens
        param_.inputLayoutPA.data(),  // inputLayout
        param_.kvHeadNum,  // numKeyValueHeads
        3,  // sparseMode
        param_.innerPrecise,  // innerPrecise
        param_.blockSize,  // blockSize
        0,  // antiquantMode
        false,  // softmaxLseFlag
        0,  // keyAntiquantMode
        0,  // valueAntiquantMode
        aclnnVariantPack.aclOutTensors.at(0)->tensor,  // attentionOut
        nullptr,  // softmaxLse
        &this->aclnnOpCache_->workspaceSize,  // workspaceSize
        &this->aclnnOpCache_->aclExecutor);   // executor
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnFusedInferAttentionScoreV3GetWorkspaceSize end, ret:" << ret
        << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize << ", aclExecutor:"
        << this->aclnnOpCache_->aclExecutor);

    return ret;
}

int FusedInferAttentionOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnFusedInferAttentionScoreV3 start");
    int ret = aclnnFusedInferAttentionScoreV3(
        workspace,
        this->aclnnOpCache_->workspaceSize,
        this->aclnnOpCache_->aclExecutor,
        stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " aclnnFusedInferAttentionScoreV3 end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed
