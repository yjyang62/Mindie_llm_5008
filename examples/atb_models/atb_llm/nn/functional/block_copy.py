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


def copy_blocks(key_cache: Tensor, value_cache: Tensor,
                src_block_indices: Tensor, dst_block_indices: Tensor,
                cum_sum: Tensor):
    """
    Copies specified blocks from source positions to destination positions.

    Args:
        key_cache (Tensor): Key cache tensor.
        value_cache (Tensor): Value cache tensor.
        src_block_indices (Tensor): Source block indices tensor, indicating the positions of blocks to copy.
        dst_block_indices (Tensor): Destination block indices tensor, indicating where the blocks are copied to.
        cum_sum (Tensor): Cumulative sum tensor, possibly used for calculation or indexing operations.

    Returns:
        None.

    Raises:
        None.
    """
    param = {}
    input_tensor = [key_cache, value_cache, src_block_indices, dst_block_indices, cum_sum]
    outputs = {}
    node = Node('BlockCopy', param, input_tensor, outputs)
    get_default_net().push_node(node)
 