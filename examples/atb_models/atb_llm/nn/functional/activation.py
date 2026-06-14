# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from enum import Enum
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.node import Node
from atb_llm.nn.tensor import Tensor


class ActType(Enum):
    RELU = 'ACTIVATION_RELU'
    GELU = 'ACTIVATION_GELU'
    FAST_GELU = 'ACTIVATION_FAST_GELU'
    SWISH = 'ACTIVATION_SWISH'
    LOG = 'ACTIVATION_LOG'
    SWIGLU = 'ACTIVATION_SWIGLU_FORWARD'
    SIGMOID = 'ACTIVATION_SIGMOID'


class GeluMode(Enum):
    TANH = 'TANH_MODE'
    NONE = 'NONE_MODE'


def activation(input_: Tensor, act_type: ActType, scale=1.0, dim=-1, gelu_mode: GeluMode = GeluMode.TANH):
    """
    Applies an activation function to the input tensor.

    Args:
        input_ (Tensor): The input tensor to apply the activation function to.
        act_type (ActType): The type of activation function to apply.
        scale (float, optional): The scaling factor for the activation function, default is 1.0.
        dim (int, optional): The dimension along which to apply the activation function, default is -1.
        gelu_mode (GeluMode, optional): The mode for the GELU activation function, default is `GeluMode.TANH`.

    Returns:
        Tensor: The output tensor after applying the activation function.
    """
    out = Tensor()
    if act_type == ActType.SIGMOID:
        param = {}
        node = Node('Sigmoid', param, [input_], [out])
    else:
        param = {
            'activationType': act_type.value,
            'scale': scale,
            'dim': dim,
            'geluMode': gelu_mode.value,
        }
        node = Node('Activation', param, [input_], [out])
    get_default_net().push_node(node)
    return out


def softmax(input_tensor: Tensor, dims: list[int] = None):
    """
    Applies the softmax function to the input tensor.

    Args:
        input_tensor (Tensor): The input tensor to apply the softmax function.
        dims (list[int], optional): Specifies the dimensions along which to apply the softmax function, default is `None`.

    Returns:
        Tensor: The output tensor after applying the softmax function.
    """
    out = Tensor()
    param = {
        "axes": dims
    }
    node = Node('Softmax', param, [input_tensor], [out])
    get_default_net().push_node(node)
    return out