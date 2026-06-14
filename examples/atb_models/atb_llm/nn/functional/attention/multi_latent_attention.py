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

from typing import Optional
from enum import Enum
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.node import Node
from atb_llm.nn.tensor import Tensor


class MaskType(str, Enum):
    UNDEFINED = "UNDEFINED"
    MASK_TYPE_SPEC = "MASK_TYPE_SPEC"
    MASK_TYPE_MASK_FREE = "MASK_TYPE_MASK_FREE"
    MASK_TYPE_CAUSAL_MASK = "MASK_TYPE_CAUSAL_MASK"


class CalcType(str, Enum):
    CALC_TYPE_UNDEFINED = "CALC_TYPE_UNDEFINED"
    CALC_TYPE_SPEC = "CALC_TYPE_SPEC"
    CALC_TYPE_RING = "CALC_TYPE_RING"
    CALC_TYPE_SPEC_AND_RING = "CALC_TYPE_SPEC_AND_RING"
    CALC_TYPE_PREFILL = "CALC_TYPE_PREFILL"


class CacheMode(str, Enum):
    KVCACHE = "KVCACHE"
    KROPE_CTKV = "KROPE_CTKV"
    INT8_NZCACHE = "INT8_NZCACHE"
    NZCACHE = "NZCACHE"


def multi_latent_attention(
        q_nope: Tensor,
        q_rope: Tensor,
        ct_kv: Tensor,
        k_rope: Tensor,
        block_tables: Tensor,
        context_lens: Tensor,
        mask: Optional[Tensor] = None,
        qseqlen: Optional[Tensor] = None,
        qk_descale: Optional[Tensor] = None,
        pv_descale: Optional[Tensor] = None,
        head_num: int = 0,
        qk_scale: float = 1.0,
        kv_head_num: int = 0,
        mask_type: MaskType = MaskType.UNDEFINED,
        calc_type: CalcType = CalcType.CALC_TYPE_UNDEFINED,
        cache_mode: CacheMode = CacheMode.KVCACHE
        ) -> Tensor:
    """
    Paged attention for MLA. Computes attention scores using a paged KV-cache. Supports separated inputs: q_nope/q_rope and ctkv/k_rope for enhanced flexibility.
    """
    node_param = {
        "headNum": head_num,
        "qkScale": qk_scale,
        "kvHeadNum": kv_head_num,
        "maskType": mask_type,
        "calcType": calc_type,
        "cacheMode": cache_mode
    }
    inputs = [q_nope, q_rope, ct_kv, k_rope, block_tables, context_lens]
    if mask_type != MaskType.UNDEFINED:
        inputs.append(mask)
    if calc_type == CalcType.CALC_TYPE_SPEC:
        inputs.append(qseqlen)
    if cache_mode == CacheMode.INT8_NZCACHE:
        inputs.append([qk_descale, pv_descale])
    output = Tensor()
    node = Node("MultiLatentAttention", node_param, inputs, [output])
    get_default_net().push_node(node)
    return output