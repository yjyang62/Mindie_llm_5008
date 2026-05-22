/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace mie_ops {
using namespace at_npu::native;

std::tuple<at::Tensor, at::Tensor> npu_dispatch_gmm_combine_decode_npu(
    const at::Tensor &x,
    const at::Tensor &expert_ids,
    const at::TensorList &gmm1_permuted_weight,
    const at::TensorList &gmm1_permuted_weight_scale,
    const at::TensorList &gmm2_weight,
    const at::TensorList &gmm2_weight_scale,
    const at::Tensor &expert_scales,
    const c10::optional<at::Tensor> &expert_smooth_scales,
    const c10::optional<at::Tensor> &x_active_mask,
    c10::string_view group_ep,
    int64_t ep_rank_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs)
{
    auto x_shape = x.sizes();
    int batch_size = x_shape[0];
    int hidden_size = x_shape[1];

    at::Tensor output = at::empty({batch_size, hidden_size}, x.options());

    bool is_shared_expert = (ep_rank_id < shared_expert_rank_num);
    int64_t num_local_experts = is_shared_expert ? 1 : moe_expert_num / (ep_rank_size - shared_expert_rank_num);
    auto opts = expert_ids.options().dtype(at::kLong);
    at::Tensor expert_token_nums = at::empty({num_local_experts}, opts);

    vector<char> group_ep_chars(group_ep.begin(), group_ep.end());
    group_ep_chars.push_back('\0');
    char *group_ep_ptr = &group_ep_chars[0];
    EXEC_NPU_CMD_V1(
        // op api
        aclnnDispatchGmmCombineDecode,
        // input tensors
        x,
        expert_ids,
        gmm1_permuted_weight,
        gmm1_permuted_weight_scale,
        gmm2_weight,
        gmm2_weight_scale,
        expert_scales,
        expert_smooth_scales,
        x_active_mask,
        // input attrs
        group_ep_ptr,
        ep_rank_size,
        ep_rank_id,
        moe_expert_num,
        shared_expert_num,
        shared_expert_rank_num,
        quant_mode,
        global_bs,
        // output tensors
        output,
        expert_token_nums);
    return {output, expert_token_nums};
}

std::tuple<at::Tensor, at::Tensor> npu_dispatch_gmm_combine_decode_meta(
    const at::Tensor &x,
    const at::Tensor &expert_ids,
    const at::TensorList &gmm1_permuted_weight,
    const at::TensorList &gmm1_permuted_weight_scale,
    const at::TensorList &gmm2_weight,
    const at::TensorList &gmm2_weight_scale,
    const at::Tensor &expert_scales,
    const c10::optional<at::Tensor> &expert_smooth_scales,
    const c10::optional<at::Tensor> &x_active_mask,
    c10::string_view group_ep,
    int64_t ep_rank_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs)
{
    auto x_shape = x.sizes();
    int batch_size = x_shape[0];
    int hidden_size = x_shape[1];

    at::Tensor output = at::empty({batch_size, hidden_size}, x.options().device(at::kMeta));

    bool is_shared_expert = (ep_rank_id < shared_expert_rank_num);
    int64_t num_local_experts = is_shared_expert ? 1 : moe_expert_num / (ep_rank_size - shared_expert_rank_num);
    auto opts = expert_ids.options().dtype(at::kLong);
    at::Tensor expert_token_nums = at::empty({num_local_experts}, opts.device(at::kMeta)); 

    return {output, expert_token_nums};
}

}

TORCH_LIBRARY_IMPL(mie_ops, PrivateUse1, m) {
    m.impl("npu_dispatch_gmm_combine_decode", &mie_ops::npu_dispatch_gmm_combine_decode_npu);
}

TORCH_LIBRARY_IMPL(mie_ops, Meta, m) {
    m.impl("npu_dispatch_gmm_combine_decode", &mie_ops::npu_dispatch_gmm_combine_decode_meta);
}
