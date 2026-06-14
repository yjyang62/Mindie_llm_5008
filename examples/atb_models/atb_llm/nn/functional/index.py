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


def sort(input_: Tensor, num: int = 0):
    """
    Sorts the input tensor and returns the sorted tensor along with the indices of the sorted elements.

    Args:
        input_ (Tensor): The input tensor to be sorted.
        num (int): The number of elements to sort. Default is 0, which sorts all elements.

    Returns:
        out (Tensor): The sorted tensor.
        indices (Tensor): The indices of the sorted elements.
    """
    out = Tensor()
    indices = Tensor()
    param = {
        "num": [num]
    }
    node = Node('Sort', param, [input_], [out, indices])
    get_default_net().push_node(node)
    return out, indices


def argsort(input_: Tensor):
    """
    Sorts the input tensor and returns a tensor of the original indices sorted by their values.

    Args:
        input_ (Tensor): The input tensor to be sorted.

    Returns:
        Tensor: The tensor of original indices sorted by their values.
    """
    out = Tensor()
    param = {}
    node = Node('Argsort', param, [input_], [out])
    get_default_net().push_node(node)
    return out


def gather(input_: Tensor, dim, index: Tensor, batch_dims=0):
    """
    Gather elements from the input tensor and return the output tensor.

    Args::
        input_ (Tensor): The input tensor from which elements are gathered.
        dim (int, optional): The axis along which to gather elements.
        index (Tensor): The index tensor specifying the positions of elements to gather.
        batch_dims (int, optional): The number of batch dimensions, default is 0.
    """
    out = Tensor()
    param = {
        "axis": dim,
        "batchDims": batch_dims
    }
    node = Node('Gather', param, [input_, index], [out])
    get_default_net().push_node(node)
    return out