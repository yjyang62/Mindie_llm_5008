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


CALC_TYPE = 'calcType'
PREFILL_KV_NOT_NONE = [True, True, False, False]
DECODE_KV_CACHE_NOT_NONE = [False, False, True, True]


class MaskType(Enum):
    UNDEFINED = "UNDEFINED"
    NORM = "NORM"
    ALIBI = "ALIBI"
    NORM_COMPRESS = "NORM_COMPRESS"
    ALIBI_COMPRESS = "ALIBI_COMPRESS"
    ALIBI_COMPRESS_SQRT = "ALIBI_COMPRESS_SQRT"
    ALIBI_COMPRESS_LEFT_ALIGN = "ALIBI_COMPRESS_LEFT_ALIGN"
    SPEC = "SPEC"


def check_attention_type(
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
) -> bool:
    kv_status = [k is not None, v is not None, k_cache is not None, v_cache is not None]
    if kv_status == PREFILL_KV_NOT_NONE:
        return True
    elif kv_status == DECODE_KV_CACHE_NOT_NONE:
        return False
    else:
        raise ValueError("Key and value are not given correctly. In prefill mode, only `k` and `v` should not be None;"
                         " in decode mode, only `k_cache` and `v_cache` should not be None, but got "
                         f"`k` is {'None' if k is None else 'not None'}, `v` is {'None' if v is None else 'not None'},"
                         f" `k_cache` is {'None' if k_cache is None else 'not None'},"
                         f" `v_cache` is {'None' if v_cache is None else 'not None'}.")


def reshape_and_cache(k: Tensor, v: Tensor, k_cache: Tensor, v_cache: Tensor, slot_mapping: Tensor):
    """
    Reshape and cache the given tensors k and v into k_cache and v_cache.

    Args:
        k (Tensor): The key tensor to be reshaped and cached.
        v (Tensor): The value tensor to be reshaped and cached.
        k_cache (Tensor): The cache tensor for storing keys.
        v_cache (Tensor): The cache tensor for storing values.
        slot_mapping (Tensor): The slot mapping tensor, used to determine the caching position.

    Returns:
        No return value.

    Exception Description:
        This function does not directly handle exceptions; exceptions should be handled in the context where this function is called.
    """
    # Create a node named "ReshapeAndCache" with compression type and KV cache configuration
    node = Node("ReshapeAndCache", {"compressType": "COMPRESS_TYPE_UNDEFINED", "kvCacheCfg": 0},
                [k, v, k_cache, v_cache, slot_mapping], [k_cache, v_cache])
    # Push the created node to the default network for processing
    get_default_net().push_node(node)


def paged_attention(
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        mask_type: MaskType = MaskType.UNDEFINED,
        block_table: Optional[Tensor] = None,
        head_num: int = 0,
        kv_head_num: Optional[int] = None,
        q_lens: Optional[Tensor] = None,
        kv_lens: Optional[Tensor] = None,
        slopes: Optional[Tensor] = None,
        q_scale: float = 1.0,
        qk_scale: float = 1.0,
        is_triu_mask: bool = False,
        high_precision: bool = False,
        mla_v_head_size: int = 0
) -> Tensor:
    out = Tensor()
    is_prefill = check_attention_type(k, v, k_cache, v_cache)
    kv_head_num = head_num if kv_head_num is None else kv_head_num
    node_param = {
        'headNum': head_num,
        'kvHeadNum': kv_head_num,
        'batchRunStatusEnable': False,
        'qScale': q_scale,
        'qkScale': qk_scale,
        'maskType': f"MASK_TYPE_{mask_type.value}",
        'clampType': "CLAMP_TYPE_UNDEFINED",
        'clampMin': 0,
        'clampMax': 0,
        'isTriuMask': 1 if is_triu_mask else 0,
        "kernelType": "KERNELTYPE_HIGH_PRECISION" if high_precision else "KERNELTYPE_DEFAULT",
        "mlaVHeadSize": mla_v_head_size
    }
    op_type = None
    in_tensors = [q]
    out_tensors = [out]
    if is_prefill:
        op_type = "SelfAttention"
        node_param[CALC_TYPE] = 'PA_ENCODER'
        in_tensors.extend([k, v])
        if mask is not None:
            in_tensors.append(mask)
        in_tensors.append(kv_lens)
        need_slopes = slopes is not None and (mask_type == MaskType.ALIBI or mask_type == MaskType.ALIBI_COMPRESS or \
            mask_type == MaskType.ALIBI_COMPRESS_SQRT)
        if need_slopes:
            in_tensors.append(slopes)
    else:
        mask_type_invalid = mask_type not in [MaskType.UNDEFINED, MaskType.NORM, MaskType.ALIBI, MaskType.SPEC]
        if mask_type_invalid:
            raise ValueError("Paged attention decode only support MaskType: UNDEFINED (all zero), NORM, ALIBI, SPEC,"
                             f"while get {mask_type.name}.")
        op_type = "PagedAttention"
        in_tensors.extend([k_cache, v_cache, block_table, kv_lens])
        if mask is not None:
            in_tensors.append(mask)
        if q_lens is not None:
            node_param[CALC_TYPE] = 'CALC_TYPE_SPEC'
            in_tensors.append(q_lens)
    node = Node(op_type, node_param, in_tensors, out_tensors)
    get_default_net().push_node(node)
    return out