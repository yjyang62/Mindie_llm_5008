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
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/aclnn_dequant_bias.h"
#include "dequant_bias_operation.h"


namespace atb_speed::common {

    DequantBiasOperation::DequantBiasOperation(
        const std::string &name,
        AclNNDequantBiasParam param
    ) : AclNNOperation(name), param_(param)
    {
        this->opName_ = name;
        this->param_ = param;
    }

    DequantBiasOperation::~DequantBiasOperation()
    {
        ATB_SPEED_LOG_DEBUG("DequantBiasOperation deconstruct");
        this->DestroyOperation();
    }

    /**
     *
     * @param[in] inTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @param[in] outTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @return atb::Status
     */
    atb::Status DequantBiasOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc
    ) const
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape start");
        // check input tensors
        for (int i = 0; i < NUM2; ++i) {
            std::string inputShape;
            for (uint64_t dim = 0; dim < inTensorDesc.at(i).shape.dimNum; ++dim) {
                inputShape.append(std::to_string(inTensorDesc.at(i).shape.dims[dim]));
                inputShape.append(", ");
            }
            ATB_SPEED_LOG_DEBUG(
                opName_ << " input" << i << " dimNum = " << inTensorDesc.at(i).shape.dimNum <<
                ", inputShape = [" << inputShape << "]" <<
                ", dtype = " << inTensorDesc.at(i).dtype
            );
        }
        // check output tensors
        outTensorDesc.at(DIM0).format = inTensorDesc.at(DIM0).format;
        outTensorDesc.at(DIM0).shape.dimNum = inTensorDesc.at(DIM0).shape.dimNum;
        outTensorDesc.at(DIM0).dtype = param_.outputDtype;
        for (uint64_t dim = 0; dim < inTensorDesc.at(DIM0).shape.dimNum; ++dim) {
            outTensorDesc.at(DIM0).shape.dims[dim] = inTensorDesc.at(DIM0).shape.dims[dim];
        }
        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape end");

        return atb::NO_ERROR;
    }

    uint32_t DequantBiasOperation::GetInputNum() const
    {
        std::string inputTensors = "IN_INPUT, IN_DESCALE";
        int inputNum = NUM2;  // IN_INPUT, IN_DESCALE
        if (param_.hasActivateScale) {
            inputNum++;  // IN_PER_TOKEN_SCALE
            inputTensors.append(", IN_PER_TOKEN_SCALE");
        }
        if (param_.hasBias) {
            inputNum++;  // IN_BIAS
            inputTensors.append(", IN_BIAS");
        }
        ATB_SPEED_LOG_DEBUG(opName_ << " inputNum: " << inputNum << ", inputTensors: " << inputTensors);
        return inputNum;
    }

    uint32_t DequantBiasOperation::GetOutputNum() const
    {
        return NUM1;
    }

    atb::Status DequantBiasOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(GetInputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
            atb::Tensor atbTensor = variantPack.inTensors.at(i);
            aclnnTensor->needUpdateTensorDataPtr = true;
            aclnnTensor->atbTensor = atbTensor;
            aclnnTensor->tensorIdx = i == NUM2 && !param_.hasActivateScale ? i + 1 : i;
            aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
            CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensor);
            if (aclnnTensor->tensor == nullptr) {
                return atb::ERROR_INTERNAL_ERROR;
            }
            aclnnVariantPack.aclInTensors[i] = aclnnTensor;
        }
        ATB_SPEED_LOG_DEBUG(opName_ << " " << GetInputNum() << " AclNNInTensor created");
        return atb::NO_ERROR;
    }

    int DequantBiasOperation::SetAclNNWorkspaceExecutor()
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        int inputIdx = DIM2;
        aclTensor* activateScaleTensor = param_.hasActivateScale
            ? aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor
            : nullptr;
        aclTensor* biasTensor = param_.hasBias
            ? aclnnVariantPack.aclInTensors.at(inputIdx++)->tensor
            : nullptr;
        int ret = aclnnDequantBiasGetWorkspaceSize(
            aclnnVariantPack.aclInTensors.at(DIM0)->tensor,
            aclnnVariantPack.aclInTensors.at(DIM1)->tensor,
            activateScaleTensor,
            biasTensor,
            param_.outputDtype,
            aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        if (const char *errMsg = aclGetRecentErrMsg(); errMsg != nullptr) {
            std::stringstream ss;
            ss << this->opName_ << " SetAclNNWorkspaceExecutor error: " << errMsg;
            ATB_SPEED_LOG_ERROR(ss.str());
            throw std::runtime_error(ss.str());
        }
        ATB_SPEED_LOG_DEBUG(
            opName_ << " SetAclNNWorkspaceExecutor end"
            << ", ret: " << ret
            << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
            << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor
        );
        return ret;
    }

    int DequantBiasOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
        int ret = aclnnDequantBias(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end");
        return ret;
    }

}  // namespace atb_speed::common
