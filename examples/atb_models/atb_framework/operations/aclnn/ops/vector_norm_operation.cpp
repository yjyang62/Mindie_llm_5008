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
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/aclnn_linalg_vector_norm.h"
#include "vector_norm_operation.h"


namespace atb_speed::common {

    VectorNormOperation::VectorNormOperation(
        const std::string &name,
        atb_speed::common::AclNNVectorNormParam param
    ) : AclNNOperation(name), param_(param)
    {
        this->opName_ = name;
        this->param_ = param;
    }

    VectorNormOperation::~VectorNormOperation()
    {
        ATB_SPEED_LOG_DEBUG("VectorNormOperation deconstruct");
        if (dims != nullptr) {
            aclDestroyIntArray(dims);
        }
        if (param_.ord != nullptr) {
            aclDestroyScalar(param_.ord);
        }

        this->DestroyOperation();
    }

    /**
     *
     * @param[in] inTensorDesc: dimNum = 3, [batch_size, seq_len, hidden_size]
     * @param[in] outTensorDesc: dimNum = 3, [batch_size, seq_len, hidden_size]
     * @return atb::Status
     */
    atb::Status VectorNormOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc
    ) const
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape start");
        outTensorDesc.at(0).format = inTensorDesc.at(0).format;
        outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
        outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;

        if (inTensorDesc.at(0).shape.dimNum == DIM3) {
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 3] CHECK " << opName_ << " input shape: [input0] "
                          << inTensorDesc.at(0).shape.dims[DIM0] << ", "
                          << inTensorDesc.at(0).shape.dims[DIM1] << ", "
                          << inTensorDesc.at(0).shape.dims[DIM2]);
            outTensorDesc.at(0).shape.dims[DIM0] = inTensorDesc.at(0).shape.dims[DIM0];
            outTensorDesc.at(0).shape.dims[DIM1] = inTensorDesc.at(0).shape.dims[DIM1];
            outTensorDesc.at(0).shape.dims[DIM2] = inTensorDesc.at(0).shape.dims[DIM2];
        } else if (inTensorDesc.at(0).shape.dimNum == DIM2) {
            ATB_SPEED_LOG_DEBUG("[input0 dimNum = 2] CHECK " << opName_ << " input shape: [input0] "
                          << inTensorDesc.at(0).shape.dims[DIM0] << ", "
                          << inTensorDesc.at(0).shape.dims[DIM1]);
            outTensorDesc.at(0).shape.dims[DIM0] = inTensorDesc.at(0).shape.dims[DIM0];
            outTensorDesc.at(0).shape.dims[DIM1] = 1;
        }  else {
            ATB_SPEED_LOG_ERROR(opName_ << " invalid dimNum = " << inTensorDesc.at(0).shape.dimNum);
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape end");
        return atb::NO_ERROR;
    }

    uint32_t VectorNormOperation::GetInputNum() const
    {
        return NUM1;  // inputTensorNum = 1
    }

    uint32_t VectorNormOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    atb::Status VectorNormOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(GetInputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            atb::Tensor (*ReshapeTensorDecsFuncPtr)(atb::Tensor) = &SqueezeBatchSeq;
            if (CreateTensor(variantPack.inTensors.at(i), i, aclnnVariantPack.aclInTensors[i],
                             ReshapeTensorDecsFuncPtr) != atb::NO_ERROR) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
            ATB_SPEED_LOG_DEBUG(opName_ << " aclnnTensor = " << aclnnVariantPack.aclInTensors[i]);
        }
        return atb::NO_ERROR;
    }

    atb::Status VectorNormOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
            atb::Tensor (*ReshapeTensorDecsFuncPtr)(atb::Tensor) = &SqueezeBatchSeq;
            if (CreateTensor(variantPack.outTensors.at(i), i, aclnnVariantPack.aclOutTensors[i],
                             ReshapeTensorDecsFuncPtr) != atb::NO_ERROR) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
        }
        return atb::NO_ERROR;
    }

    int VectorNormOperation::SetAclNNWorkspaceExecutor()
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        float ord = 1.0;
        param_.ord = aclCreateScalar(&ord, aclDataType::ACL_FLOAT);
        std::vector<int64_t> dimData = { -1 };
        if (dims == nullptr) {
            dims = aclCreateIntArray(dimData.data(), 1);
        }

        int ret = aclnnLinalgVectorNormGetWorkspaceSize(
            aclnnVariantPack.aclInTensors.at(0)->tensor,
            param_.ord,
            dims,
            true,
            aclDataType::ACL_FLOAT16,
            aclnnVariantPack.aclOutTensors.at(0)->tensor,
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end"
                      << ", ret: " << ret
                      << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                      << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor);
        return ret;
    }

    int VectorNormOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
        int ret = aclnnLinalgVectorNorm(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end"
                      << ", ret: " << ret);
        return ret;
    }

}  // namespace atb_speed::common
