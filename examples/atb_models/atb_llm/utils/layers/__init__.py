#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# MIT License
# Copyright (c) 2025 inclusionAI
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Implement part of this file based on inclusionAI/Ling
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List

import torch
import torch_npu
import torch.distributed
from torch.nn import functional as F

from atb_llm.utils.log import logger
from .attention import AttentionMask, KvCache, FA3
from .embedding.position_rotary_embedding import PositionRotaryEmbedding
from .embedding.tensor_embedding import TensorEmbedding, TensorParallelEmbedding
from .linear import (	
    support_load_sharded_weight,	
    get_linear,	
    TensorParallelRowLinear,	
    TensorParallelColumnLinear,	
    TensorReplicatedLinear,	
    TensorParallelHead,
    TensorHead
)	
from .linear.reduce_quant import ReduceQuant
from .norm.fast_layer_norm import RMSNorm, RMSNormBias, RMSNormWrapper, RMSNormAntiOutlierWrapper


@support_load_sharded_weight(is_classmethod=False)
def load_column_multi(
        config, prefixes: List[str], weights, head_size, lm_head: bool = False, \
        norm: bool = False, bias: bool = False, dim: int = 0, is_nzcasted=False):
    soc_version = torch_npu._C._npu_get_soc_version()
    quantize = None if lm_head else config.quantize
    expert_nzcasted = is_nzcasted and (not lm_head)
    if expert_nzcasted and ('layers.61' not in prefixes[0] or hasattr(config, "mtp_quantize")):
        weight = weights.get_nzcasted_weights(config, prefixes)
    else:
        weight = weights.get_multi_weights_col(prefixes, quantize=quantize, dim=0, gqa_size=head_size)
    if bias:
        b = [weights.get_sharded(f"{p}.bias", dim=0, gqa_size=head_size) for p in prefixes]
        bias = torch.cat(b, dim=dim)
    else:
        bias = None
    if lm_head:
        weight_type = weight.dtype
        weight = weight.float()
        weight = weight if not norm else torch.nan_to_num(F.normalize(weight))
        if soc_version == 240:
            weight = weight.to(dtype=weight_type)
            weight = weight.npu()
        else:
            weight = weight.to(dtype=weight_type).npu()
    linear = get_linear(weight, bias, quantize, inter_type=config.torch_dtype, prefixes=prefixes,
        num_linear_before_pack=len(prefixes), tensor_parallel_dim=0, align_size=head_size, nd_weight=is_nzcasted)

    process_group = weights.process_group
    should_gather = weights.process_group.size() != 1
    if lm_head:
        return TensorParallelHead(linear, process_group=process_group, should_gather=should_gather)
    else:
        return TensorParallelColumnLinear(linear)


def load_row(config, prefix: str, weights, head_size):
    weight = weights.get_sharded(f"{prefix}.weight", dim=1, gqa_size=head_size)
    linear = get_linear(weight, None, quantize=config.quantize, inter_type=config.torch_dtype, prefixes=[prefix],
        tensor_parallel_dim=1, align_size=head_size)
    return TensorParallelRowLinear(linear, process_group=weights.process_group)
