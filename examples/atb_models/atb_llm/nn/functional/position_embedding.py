# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.node import Node
from atb_llm.nn.tensor import Tensor


def rope(q: Tensor, k: Tensor, cos_table: Tensor, sin_table: Tensor, seqlen: Tensor, rotary_coeff=2):
    """
    Apply Rotary Positional Encoding (RoPE) to the query (q) and key (k) tensors.

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        cos_table (Tensor): Cosine encoding table.
        sin_table (Tensor): Sine encoding table.
        seqlen (Tensor): Sequence length tensor.
        rotary_coeff (int, optional): Rotary coefficient. Default is 2.

    Returns:
        q_embed (Tensor): Query tensor after applying rotary encoding.
        k_embed (Tensor): Key tensor after applying rotary encoding.
    """
    q_embed = Tensor()
    k_embed = Tensor()
    node = Node("Rope", {'rotaryCoeff': rotary_coeff}, [q, k, cos_table, sin_table, seqlen], [q_embed, k_embed])
    get_default_net().push_node(node)
    return q_embed, k_embed