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
#include "aclnnop/aclnn_layer_norm.h"
#include "layer_norm_operation.h"


namespace atb_speed::common {

    LayerNormOperation::LayerNormOperation(
        const std::string &name,
        atb_speed::common::AclNNLayerNormParam param
    ) : AclNNOperation(name), param_(param)
    {
        this->opName_ = name;
        this->param_ = param;
    }

    LayerNormOperation::~LayerNormOperation()
    {
        ATB_SPEED_LOG_DEBUG("LayerNormOperation deconstruct");
        this->DestroyOperation();
    }

    /**
     *
     * @param[in] inTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @param[in] outTensorDesc: FA: [batchSize, seqLen, hiddenSize]; PA: [seqLen, hiddenSize]
     * @return atb::Status
     */
    atb::Status LayerNormOperation::InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc
    ) const
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape start");
        outTensorDesc.at(0).format = inTensorDesc.at(0).format;
        outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
        outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;

        ATB_SPEED_LOG_DEBUG("Check " << opName_ << " input dimNum=" << inTensorDesc.at(0).shape.dimNum);
        for (uint64_t dim = 0; dim < inTensorDesc.at(0).shape.dimNum; ++dim) {
            ATB_SPEED_LOG_DEBUG("input dim" << dim << " shape=" << inTensorDesc.at(0).shape.dims[dim]);
            outTensorDesc.at(0).shape.dims[dim] = inTensorDesc.at(0).shape.dims[dim];
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " InferShape end");
        return atb::NO_ERROR;
    }

    uint32_t LayerNormOperation::GetInputNum() const
    {
        return NUM3;  // inputTensorNum = 3
    }

    uint32_t LayerNormOperation::GetOutputNum() const
    {
        return NUM1;  // outputTensorNum = 1
    }

    atb::Status LayerNormOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
    {
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        aclnnVariantPack.aclInTensors.resize(GetInputNum());
        for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
            ATB_SPEED_LOG_DEBUG(opName_ << " CreateTensor start");
            atb::Tensor (*ReshapeTensorDecsFuncPtr)(atb::Tensor) = &SqueezeBatchSeq;
            if (CreateTensor(variantPack.inTensors.at(i), i, aclnnVariantPack.aclInTensors[i],
                             ReshapeTensorDecsFuncPtr) != atb::NO_ERROR) {
                ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
                return atb::ERROR_INTERNAL_ERROR;
            }
            ATB_SPEED_LOG_DEBUG(opName_ << " CreateTensor end");
        }
        return atb::NO_ERROR;
    }

    atb::Status LayerNormOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
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

    int LayerNormOperation::SetAclNNWorkspaceExecutor()
    {
        ATB_SPEED_LOG_DEBUG(
            opName_ << " SetAclNNWorkspaceExecutor start"
                    << ", layerNormEps: " << param_.layerNormEps
                    << ", beginNormAxis: " << param_.beginNormAxis
                    << ", layerNormImplMode: " << param_.layerNormImplMode
        );
        AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
        int beginNormAxis =
            param_.beginNormAxis < 0 ?
            aclnnVariantPack.aclInTensors.at(0)->atbTensor.desc.shape.dimNum + param_.beginNormAxis :
            param_.beginNormAxis;
        uint64_t normalizedMaxDimNum = static_cast<uint64_t>(beginNormAxis + param_.normAxes);
        uint64_t inTensorDimNum = aclnnVariantPack.aclInTensors.at(0)->atbTensor.desc.shape.dimNum;
        if (normalizedMaxDimNum > inTensorDimNum) {
            std::stringstream ss;
            ss << this->opName_ << " normalized max dimNum " << normalizedMaxDimNum
                                << " > inTensor dimNum " << inTensorDimNum;
            ATB_SPEED_LOG_ERROR(
                this->opName_ << " normalized max dimNum " << normalizedMaxDimNum
                              << " > inTensor dimNum " << inTensorDimNum;
            );
            throw std::runtime_error(ss.str());
        }
        int64_t normalizedShapeValue[param_.normAxes];
        for (int i = 0; i < param_.normAxes; ++i) {
            normalizedShapeValue[i] = aclnnVariantPack.aclInTensors.at(0)->atbTensor.desc.shape.dims[
                beginNormAxis + i
            ];
        }
        aclIntArray *normalizedShape = aclCreateIntArray(normalizedShapeValue, param_.normAxes);
        int ret = aclnnLayerNormWithImplModeGetWorkspaceSize(
            aclnnVariantPack.aclInTensors.at(0)->tensor,                                // input
            normalizedShape,                                                            // normalizedShape
            aclnnVariantPack.aclInTensors.at(1)->tensor,                                // weight
            param_.hasBias ? aclnnVariantPack.aclInTensors.at(2)->tensor : nullptr,     // bias
            param_.layerNormEps,                                                        // eps
            aclnnVariantPack.aclOutTensors.at(0)->tensor,                               // out
            nullptr,                                                                    // meanOut
            nullptr,                                                                    // rstdOut
            static_cast<int32_t>(param_.layerNormImplMode),                             // implMode
            &this->aclnnOpCache_->workspaceSize,
            &this->aclnnOpCache_->aclExecutor);
        aclDestroyIntArray(normalizedShape);
        ATB_SPEED_LOG_DEBUG(
            opName_ << " SetAclNNWorkspaceExecutor end"
                    << ", ret: " << ret
                    << ", workspaceSize: " << this->aclnnOpCache_->workspaceSize
                    << ", aclExecutor: " << this->aclnnOpCache_->aclExecutor
        );
        return ret;
    }

    int LayerNormOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
    {
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
        int ret = aclnnLayerNormWithImplMode(
            workspace,
            this->aclnnOpCache_->workspaceSize,
            this->aclnnOpCache_->aclExecutor,
            stream);
        ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end" << ", ret: " << ret);
        return ret;
    }

}  // namespace atb_speed::common
