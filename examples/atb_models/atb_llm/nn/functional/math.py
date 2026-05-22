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

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.node import Node
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.quantize.pack_type import DataType


def cos(input_: Tensor):
    """
    Computes the cosine of the input tensor.

    Args:
        input_ (Tensor): Input tensor to compute the cosine of.

    Returns:
        Tensor: The cosine of the input tensor.
    """
    out = Tensor()
    param = {
        "elewiseType": "ELEWISE_COS",
    }
    node = Node('Elewise', param, [input_], [out])
    get_default_net().push_node(node)
    return out


def sin(input_: Tensor):
    """
    Computes the sine of the input tensor.

    Args:
        input_ (Tensor): Input tensor to compute the sine of.

    Returns:
        Tensor: The sine of the input tensor.
    """
    out = Tensor()
    param = {
        "elewiseType": "ELEWISE_SIN",
    }
    node = Node('Elewise', param, [input_], [out])
    get_default_net().push_node(node)
    return out


def neg(input_: Tensor):
    """
    Computes the negative of the input tensor.

    Args:
        input_ (Tensor): Input tensor to compute the negative of.

    Returns:
        Tensor: The negative of the input tensor.
    """
    out = Tensor()
    param = {
        "elewiseType": "ELEWISE_NEG",
    }
    node = Node('Elewise', param, [input_], [out])
    get_default_net().push_node(node)
    return out


def logical_not(input_: Tensor):
    """
    Computes the logical not.

    Args:
        input_ (Tensor): Input tensor to compute the logical not.

    Returns:
        Tensor: The logical not.
    """
    out = Tensor()
    param = {
        "elewiseType": "ELEWISE_LOGICAL_NOT",
    }
    node = Node('Elewise', param, [input_], [out])
    get_default_net().push_node(node)
    return out


def logical_or(input_: Tensor, other: Tensor):
    """
    Computes the logical or of the two input tensors.

    Args:
        input_ (Tensor): First input tensor to compute the logical or of.
        other (Tensor): Second input tensor to compute the logical or of.

    Returns:
        Tensor: Output tensor containing the logical or result of the two input tensors.
    """
    out = Tensor()
    param = {
        "elewiseType": "ELEWISE_LOGICAL_OR",
    }
    node = Node('Elewise', param, [input_, other], [out])
    get_default_net().push_node(node)
    return out


def logical_and(input_: Tensor, other: Tensor):
    """
    Computes the logical and of the two input tensors.

    Args:
        input_ (Tensor): First input tensor to compute the logical or of.
        other (Tensor): Second input tensor to compute the logical or of.

    Returns:
        Tensor: Output tensor containing the logical and result of the two input tensors.
    """
    out = Tensor()
    param = {
        "elewiseType": "ELEWISE_LOGICAL_AND",
    }
    node = Node('Elewise', param, [input_, other], [out])
    get_default_net().push_node(node)
    return out


def equal(input_: Tensor, other: Tensor):
    """
    Computes the equality of the two input tensors.

    Args:
        input_ (Tensor): First input tensor to compute the logical or of.
        other (Tensor): Second input tensor to compute the logical or of.

    Returns:
        Tensor: Output tensor containing the equality of the two input tensors.
    """
    out = Tensor()
    param = {
        "elewiseType": "ELEWISE_EQUAL",
    }
    node = Node('Elewise', param, [input_, other], [out])
    get_default_net().push_node(node)
    return out


def amax(input_: Tensor, dims: list[int]):
    """
    Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.

    Args:
        input_ (Tensor): Input tensor.
        dims (list[int], optional): Specifies the dimensions along which to apply the maximum function.

    Returns:
        Tensor: The maximum value of the input tensor along the specified dimensions.
    """
    out = Tensor()
    node = Node("Reduce", {"reduceType": "REDUCE_MAX", "axis": dims}, [input_], [out])
    get_default_net().push_node(node)
    return out


def amin(input_: Tensor, dims: list[int]):
    """
    Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.

    Args:
        input_ (Tensor): Input tensor.
        dims (list[int], optional): Specifies the dimensions along which to apply the minimum function.

    Returns:
        Tensor: The minimum value of the input tensor along the specified dimensions.
    """
    out = Tensor()
    node = Node("Reduce", {"reduceType": "REDUCE_MIN", "axis": dims}, [input_], [out])
    get_default_net().push_node(node)
    return out


def sum_(input_: Tensor, dims: list[int]):
    """
    Returns the sum of each row of the input tensor in the given dimension dim.

    Args:
        input_ (Tensor): Input tensor.
        dims (list[int], optional): Specifies the dimensions along which to apply the sum function.

    Returns:
        Tensor: The sum of the input tensor along the specified dimensions.
    """
    out = Tensor()
    node = Node("Reduce", {"reduceType": "REDUCE_SUM", "axis": dims}, [input_], [out])
    get_default_net().push_node(node)
    return out


def std(input_: Tensor):
    """
    Computes the standard deviation of the input tensor.

    Args:
        input_ (Tensor): Input tensor.

    Returns:
        Tensor: The standard deviation of the input tensor.
    """
    out = Tensor()
    param = {}
    node = Node("Std", param, [input_], [out])
    get_default_net().push_node(node)
    return out


def norm(input_: Tensor):
    """
    Computes the norm of the input tensor.

    Args:
        input_ (Tensor): Input tensor.

    Returns:
        Tensor: The norm of the input tensor.
    """
    out = Tensor()
    param = {}
    node = Node("VectorNorm", param, [input_], [out])
    get_default_net().push_node(node)
    return out


def linear(input_: Tensor,
           weight: Tensor,
           bias: Tensor = None,
           transpose_a: bool = False,
           transpose_b: bool = True):
    """
    Linear transformation function that performs a linear transformation on the input tensor.

    Args:
        input_ (Tensor): Input tensor.
        weight (Tensor): Weight tensor.
        bias (Tensor, optional): Bias tensor. Defaults to None.
        transpose_a (bool, optional): Whether to transpose the input tensor. Defaults to False.
        transpose_b (bool, optional): Whether to transpose the weight tensor. Defaults to False.

    Returns:
        Tensor: Output tensor after the linear transformation.
    """
    out = Tensor()
    inputs = [input_, weight] + ([bias] if bias is not None else [])
    param = {
        "transposeA": transpose_a,
        "transposeB": transpose_b,
        "hasBias": bias is not None,
    }
    node = Node("Linear", param, inputs, [out])
    get_default_net().push_node(node)
    return out


def rms_norm(input_: Tensor, weight: Tensor, eps: float):
    """
    Computes the RMSProp normalization of the input tensor.

    Args:
        input_ (Tensor): Input tensor to be normalized.
        weight (Tensor): Weight tensor used for normalization.
        eps (float, optional): A small value added for numerical stability.

    Returns:
        Tensor: The output tensor after the RMSProp normalization.
    """
    out = Tensor()
    param = {
        'layerType': 'RMS_NORM_NORM',
        'normParam': {
            'epsilon': eps
        }
    }
    node = Node('RmsNorm', param, [input_, weight], [out])
    get_default_net().push_node(node)
    return out


def grouped_matmul(
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        group_list: Tensor = None,
        transpose_b: bool = False) -> Tensor:
    """
    Perform grouped matrix multiplication with optional bias and transposition.

    Args:
        input_ (Tensor): Input tensor for matrix multiplication.
        weight (Tensor): Weight tensor for matrix multiplication.
        bias (Tensor, optional): Bias tensor to be added. Defaults to None.
        group_list (Tensor, optional): The tensor specifying the group information. Defaults to None.
        transpose_b (bool, optional): Whether to transpose the weight tensor before multiplication. Defaults to False.

    Returns:
        Tensor: The output tensor after the grouped matrix multiplication.
    """
    has_bias = True if bias is not None else False
    param = {
        "transposeB": transpose_b,
        "quantType": 0,
        "hasBias": has_bias,
        "outDataType": DataType.ACL_DT_UNDEFINED.value
    }
    inputs = [input_, weight]
    inputs.append(bias) if has_bias else None
    inputs.append(group_list)
    out = Tensor()
    node = Node("GroupedMatmul", param, inputs, [out])
    get_default_net().push_node(node)
    return out