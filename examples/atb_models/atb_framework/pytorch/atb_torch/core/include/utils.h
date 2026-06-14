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

#ifndef EXAMPLE_UTIL_H
#define EXAMPLE_UTIL_H
#include <vector>
#include <string>
#include <atb/types.h>
#include <torch/torch.h>
#include "atb/operation.h"

namespace atb_torch {
class Utils {
public:
    static void *GetCurrentStream();
    static int64_t GetTensorNpuFormat(const at::Tensor &tensor);
    static at::Tensor NpuFormatCast(const at::Tensor &tensor);
    static void BuildVariantPack(const std::vector<torch::Tensor> &inTensors,
                                 const std::vector<torch::Tensor> &outTensors, atb::VariantPack &variantPack);
    static atb::Tensor AtTensor2Tensor(const at::Tensor &atTensor);
    static at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc);
    static void ContiguousAtTensor(std::vector<torch::Tensor> &atTensors);
    static void ContiguousAtTensor(torch::Tensor &atTensor);
    static std::string AtTensor2String(const at::Tensor &atTensor);
    static std::string TensorToString(const atb::Tensor &tensor);
    static std::string TensorDescToString(const atb::TensorDesc &tensorDesc);
    static void *CreateWorkspae(uint64_t workspaceSize);
};
}

#endif