# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import List
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.node import Node
from atb_llm.nn.tensor import Tensor


def split(tensor: Tensor, split_size_or_sections: int | List[int] = None, dim: int = 0) -> list[Tensor]:
    """
    Split the input tensor along the specified dimension.

    Args:
        tensor (Tensor): The input tensor to be split.
        dim (int): The dimension along which to split the tensor.
        split_size_or_sections (List[int], optional): A list of sizes for each split part.
            If not provided, defaults to an empty list.

    Returns:
        list[Tensor]: A list of tensors after splitting.
    """
    # If split_sizes is not provided, initialize it as an empty list
    if isinstance(split_size_or_sections, int):
        split_size = []
        split_num = split_size_or_sections
    else:
        split_size = split_size_or_sections
        split_num = len(split_size_or_sections)

    param = {'splitDim': dim, 'splitNum': split_num, "splitSizes": split_size}

    outs = [Tensor() for _ in range(split_num)]
    ins = [tensor]
    node = Node('Split', param, ins, outs)
    get_default_net().push_node(node)
    return outs


def cat(tensors: List[Tensor], dim=0):
    """
    Concatenate two tensors along the specified dimension.

    Args:
        tensors (List[Tensor]): A list of tensors to be concatenated.
        dim (int): The dimension along which to concatenate, default is 0.

    Returns:
        Tensor: The output tensor after concatenation.
    """
    if len(tensors) != 2:
        raise ValueError("The tensors must be given a list containing exactly two tensors.")

    out = Tensor()
    param = {
        "concatDim": dim,
    }
    node = Node('Concat', param, tensors, [out])
    get_default_net().push_node(node)
    return out