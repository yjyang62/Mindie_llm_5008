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
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "acl/acl.h"
#include "aclrt_cmo_async.h"

namespace atb_speed {
namespace common {
constexpr size_t BYTES_PER_KB = 1024;
constexpr size_t BYTES_PER_MB = BYTES_PER_KB * BYTES_PER_KB;
AclrtCmoAsyncOperation::AclrtCmoAsyncOperation(const std::string &opName,
    size_t dataSize): opName_(opName), dataSize_(dataSize * BYTES_PER_MB) {}

AclrtCmoAsyncOperation::~AclrtCmoAsyncOperation()
{
    ATB_SPEED_LOG_DEBUG("AclrtCmoAsyncOperation deconstructor");
}

std::string AclrtCmoAsyncOperation::GetName() const
{
    return this->opName_;
}


uint32_t AclrtCmoAsyncOperation::GetInputNum() const
{
    return NUM1;
}

uint32_t AclrtCmoAsyncOperation::GetOutputNum() const
{
    return 0;
}

atb::Status AclrtCmoAsyncOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                               atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    ATB_SPEED_LOG_DEBUG("inTensorDesc size: " << inTensorDesc.size() << ", outTensorDesc size: "
                        << outTensorDesc.size());
    return atb::NO_ERROR;
}

atb::Status AclrtCmoAsyncOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize,
                                          atb::Context *context)
{
    ATB_SPEED_LOG_DEBUG("variantPack outTensors size: "
                        << variantPack.outTensors.size()
                        << ", workspaceSize: "
                        << workspaceSize);
    ATB_SPEED_LOG_DEBUG(this->opName_ << " setup start");

    if (context == nullptr) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " setup context is null");
        return atb::ERROR_INVALID_PARAM;
    }

    workspaceSize = 0;

    ATB_SPEED_LOG_DEBUG("setup end");
    return atb::NO_ERROR;
}

atb::Status AclrtCmoAsyncOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace,
                                            uint64_t workspaceSize, atb::Context *context)
{
    ATB_SPEED_LOG_DEBUG(this->opName_ << " execute start: ");

    if (!context) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " execute fail, context param is null. Enable log: "
            << "export ASCEND_GLOBAL_LOG_LEVEL=3, export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl, ATB_MODELS_EXECUTION_FAILURE);
        return atb::ERROR_INVALID_PARAM;
    }

    std::vector<aclrtStream> streams = context->GetExecuteStreams();

    if (!streams[1]) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " execute fail, execute stream in context is null. "
            << "Enable log: export ASCEND_GLOBAL_LOG_LEVEL=3, export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl, ATB_MODELS_EXECUTION_FAILURE);
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtCmoType cmoType = ACL_RT_CMO_TYPE_PREFETCH;

    ATB_SPEED_LOG_DEBUG("variantPack deviceData: " << variantPack.inTensors.at(0).deviceData
                        << " ,variantPack dataSize: " << variantPack.inTensors.at(0).dataSize
                        << " ,stream: " << streams[1]);

    if (dataSize_ > variantPack.inTensors.at(0).dataSize || dataSize_ == 0) {
        dataSize_ = variantPack.inTensors.at(0).dataSize;
    }

    CheckAcl(aclrtCmoAsync(variantPack.inTensors.at(0).deviceData,
                           dataSize_,
                           cmoType,
                           streams[1]));

    ATB_SPEED_LOG_DEBUG("aclrtCmoAsync create success.");

    if (workspaceSize != 0 || workspace != nullptr) {
        ATB_SPEED_LOG_DEBUG("execute workspace: " << workspaceSize);
    }

    return atb::NO_ERROR;
}

aclError AclrtCmoAsyncOperation::CheckAcl(aclError ret) const
{
    if (ret != ACL_ERROR_NONE) {
        ATB_SPEED_LOG_ERROR(__FILE__ << ":" << __LINE__ << " aclError:" << ret);
    }
    return ret;
}

} // namespace common
} // namespace atb_speed
