#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import torch.nn.functional as F
import torch_npu # noqa
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.quantize.quant_type import QuantType


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, ((0, r_pad, 0, c_pad)))
    nz_mat = torch.permute(
        torch.reshape(nd_mat, (r // block_size[0], block_size[0], c // block_size[1], block_size[1])), [2, 0, 1, 3]
    )
    nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def transdata_3d(nd_mat, block_size: tuple = (16, 16)):
    if nd_mat.ndim != 3:
        raise ValueError("Expected a 3-dimensional input array.")
    b, _, _ = nd_mat.shape
    processed_slices = []

    for batch_index in range(b):
        current_slice = nd_mat[batch_index]
        nz_mat = transdata(current_slice, block_size)
        processed_slices.append(nz_mat)

    result = torch.stack(processed_slices, axis=0)
    return result


def preprocess_kv_weights(weight, config, proj_name):
    if isinstance(weight, tuple):
        new_weight = list(weight)
    else:
        new_weight = [weight]
    kv_b_proj_weight = new_weight[0].reshape(config.tp_num_key_value_heads,
                                                config.qk_nope_head_dim + config.v_head_dim,
                                                config.kv_lora_rank)
    if proj_name == "projk":
        new_weight[0] = kv_b_proj_weight[:, :config.qk_nope_head_dim, :].contiguous()
    elif proj_name == "projv":
        new_weight[0] = kv_b_proj_weight[:, config.qk_nope_head_dim:, :].transpose(1, 2).contiguous()
    else:
        msg = f"`proj_name`'s type field must be one of ['projk', 'projv'], " \
                f"got {proj_name}"
        raise ValueError(msg)
        
    if len(new_weight) == 1:
        new_weight = new_weight[0]
    
    return new_weight


def trans_rope_weight(weight, rope_dim):
    weight_1 = weight[..., -rope_dim:: 2, :].contiguous()
    weight_2 = weight[..., -rope_dim + 1:: 2, :].contiguous()
    weight[..., -rope_dim:, :] = torch.cat([weight_1, weight_2], dim=-2)

    return weight.contiguous()


def view_tenor(weight, config, proj_name, pre_view=True):
    k_head_dim = config.kv_lora_rank + config.qk_rope_head_dim
    if proj_name == "projq":
        if pre_view:
            return weight.view(config.tp_num_attention_heads, config.q_head_dim_before, -1).contiguous()
        else:
            return weight.view(config.tp_num_attention_heads * config.q_head_dim_before, -1).contiguous()
    elif proj_name == "projk":
        return weight.view((k_head_dim), -1).contiguous()
    else:
        msg = f"`proj_name`'s type field must be one of ['projq', 'projk'], " \
                f"got {proj_name}"
        logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
        raise ValueError(msg)


def preprocess_linear_for_rope(linear_weight, config, proj_name):
    if isinstance(linear_weight, torch.Tensor):
        linear_weight = [linear_weight]
    new_weight = list(linear_weight)
    weight = linear_weight[0]
    weight = view_tenor(weight, config, proj_name=proj_name, pre_view=True)
    weight = trans_rope_weight(weight, config.qk_rope_head_dim)
    new_weight[0] = view_tenor(weight, config, proj_name=proj_name, pre_view=False)

    supported_quantize_type = [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]

    if weight.dtype not in [torch.float16, torch.bfloat16] and config.quantize in supported_quantize_type:
        scale = linear_weight[1]
        scale = view_tenor(scale, config, proj_name=proj_name, pre_view=True)
        scale = trans_rope_weight(scale, config.qk_rope_head_dim)
        new_weight[1] = view_tenor(scale, config, proj_name=proj_name, pre_view=False)
        if config.quantize in [QuantType.W8A8_DYNAMIC]:
            new_weight[1] = new_weight[1].flatten()

        offset = linear_weight[2]
        offset = view_tenor(offset, config, proj_name=proj_name, pre_view=True)
        offset = trans_rope_weight(offset, config.qk_rope_head_dim)
        new_weight[2] = view_tenor(offset, config, proj_name=proj_name, pre_view=False)
        if config.quantize in [QuantType.W8A8_DYNAMIC]:
            new_weight[2] = new_weight[2].flatten()
    
    elif weight.dtype not in [torch.float16, torch.bfloat16] and config.quantize in [QuantType.W8A8]:
        deq_scale = linear_weight[1]
        deq_scale = view_tenor(deq_scale, config, proj_name=proj_name, pre_view=True)
        deq_scale = trans_rope_weight(deq_scale, config.qk_rope_head_dim)
        new_weight[1] = view_tenor(deq_scale, config,
                                                proj_name=proj_name, pre_view=False).flatten()

        quant_bias = linear_weight[2]
        quant_bias = view_tenor(quant_bias, config, proj_name=proj_name, pre_view=True)
        quant_bias = trans_rope_weight(quant_bias, config.qk_rope_head_dim)
        new_weight[2] = view_tenor(quant_bias, config,
                                                    proj_name=proj_name, pre_view=False).flatten()
        
    if len(linear_weight) == 1:
        new_weight = new_weight[0]
    
    return new_weight