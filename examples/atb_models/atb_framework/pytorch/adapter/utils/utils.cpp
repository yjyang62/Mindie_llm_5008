/**
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

#include "utils.h"
#include <iostream>
#pragma GCC diagnostic push
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma GCC diagnostic pop
#include <acl/acl.h>
#include <atb/utils.h>
#include <atb_speed/utils/file_system.h>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#endif

#include "atb_speed/log.h"
#include "atb_speed/utils/tensor_util.h"

namespace atb_speed {
void *Utils::GetCurrentStream()
{
    int32_t devId = 0;
    int ret = aclrtGetDevice(&devId);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR("aclrtGetDevice failed, error: " << ret);
    }
    void *stream = c10_npu::getCurrentNPUStream(devId).stream();
    if (stream == nullptr) {
        ATB_SPEED_LOG_ERROR("Get current stream fail");
    }
    return stream;
}

int64_t Utils::GetTensorNpuFormat(const at::Tensor &tensor)
{
#ifdef TORCH_HIGHER_THAN_PTA6
    return at_npu::native::get_npu_format(tensor);
#else
    return at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
}

at::Tensor Utils::NpuFormatCast(const at::Tensor &tensor)
{
#ifdef TORCH_HIGHER_THAN_PTA6
    return at_npu::native::npu_format_cast(tensor, GetTensorNpuFormat(tensor));
#else
    return at_npu::native::NPUNativeFunctions::npu_format_cast(tensor, GetTensorNpuFormat(tensor));
#endif
}

void Utils::BuildVariantPack(const std::vector<torch::Tensor> &inTensors, const std::vector<torch::Tensor> &outTensors,
                             atb::VariantPack &variantPack)
{
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2Tensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2Tensor(outTensors.at(i)));
    }
}

atb::Tensor Utils::AtTensor2Tensor(const at::Tensor &atTensor)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},    {at::ScalarType::Byte, ACL_UINT8},  {at::ScalarType::Char, ACL_INT8},
        {at::ScalarType::Half, ACL_FLOAT16}, {at::ScalarType::Float, ACL_FLOAT}, {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64},   {at::ScalarType::BFloat16, ACL_BF16},
    };

    if (!atTensor.is_contiguous()) {
        ATB_SPEED_LOG_ERROR("atTensor is not contiguous");
    }
    
    atb::Tensor tensor;
    tensor.desc.format = static_cast<aclFormat>(GetTensorNpuFormat(atTensor));
    tensor.deviceData = atTensor.data_ptr();

    tensor.desc.shape.dimNum = atTensor.sizes().size();
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        tensor.desc.shape.dims[i] = atTensor.sizes()[i];
    }

    if (tensor.desc.shape.dimNum == 1 && tensor.desc.shape.dims[0] == 0) {
        tensor.desc.shape.dimNum = 0;
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    if (it != dtypeMap.end()) {
        tensor.desc.dtype = it->second;
    } else {
        throw std::runtime_error("AtTensor2Tensor: not support dtype");
    }

    tensor.dataSize = atb::Utils::GetTensorSize(tensor);

    return tensor;
}

at::Tensor Utils::CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc)
{
    static std::map<aclDataType, at::ScalarType> dtypeMap = {
        {ACL_BOOL, at::ScalarType::Bool},    {ACL_UINT8, at::ScalarType::Byte},  {ACL_INT8, at::ScalarType::Char},
        {ACL_FLOAT16, at::ScalarType::Half}, {ACL_FLOAT, at::ScalarType::Float}, {ACL_INT32, at::ScalarType::Int},
        {ACL_INT64, at::ScalarType::Long},   {ACL_BF16, at::ScalarType::BFloat16},
    };
    at::TensorOptions options = at::TensorOptions();
    auto it = dtypeMap.find(tensorDesc.dtype);
    if (it != dtypeMap.end()) {
        options = options.dtype(it->second);
    } else {
        throw std::runtime_error("CreateAtTensorFromTensorDesc: not support dtype");
    }

    options = options.layout(torch::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());

    ATB_SPEED_LOG_DEBUG("tensor_with_format stat, " << atb_speed::TensorUtil::TensorDescToString(tensorDesc));

#ifdef TORCH_HIGHER_THAN_PTA6
    at::Tensor newTensor = at_npu::native::empty_with_format(
        at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options, tensorDesc.format);
#else
    at::Tensor newTensor = at_npu::native::NPUNativeFunctions::tensor_with_format(
        at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options, tensorDesc.format);
#endif

    ATB_SPEED_LOG_DEBUG("tensor_with_format end, newTensor.format:" << GetTensorNpuFormat(newTensor)
                  << ", is_contiguous:" << newTensor.is_contiguous());
    if (GetTensorNpuFormat(newTensor) != tensorDesc.format) {
        ATB_SPEED_LOG_WARN("tensor_with_format newTensor.format:" << GetTensorNpuFormat(newTensor)
                      << " != " << tensorDesc.format);
    }
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }

    ATB_SPEED_LOG_DEBUG("tensor_with_format success, newTensor.options:" << newTensor.options()
                  << ", format:" << GetTensorNpuFormat(newTensor) << ", is_contiguous:" << newTensor.is_contiguous());

    return newTensor;
}

void Utils::ContiguousAtTensor(std::vector<torch::Tensor> &atTensors)
{
    for (size_t i = 0; i < atTensors.size(); ++i) {
        if (!atTensors.at(i).is_contiguous()) {
            atTensors.at(i) = atTensors.at(i).contiguous();
        }
    }
}

void Utils::ContiguousAtTensor(torch::Tensor &atTensor)
{
    if (!atTensor.is_contiguous()) {
        atTensor = atTensor.contiguous();
    }
}
}
