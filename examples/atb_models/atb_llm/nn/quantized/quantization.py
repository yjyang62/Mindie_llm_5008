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

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.node import Node
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.quantize.pack_type import DataType


def quantize_per_token(input_: Tensor) -> list[Tensor]:
    """
    Performs per-token quantization on the input tensor.
    
    This function applies dynamic quantization to the input tensor, producing 
    a quantized tensor and corresponding scale values for each token.
    
    Calculation formula:
        scale = row_max(abs(input_))/dtypeMax
        y = round(input_/scale)

    Args:
        input_ (Tensor): The input tensor to be quantized.
        
    Returns:
        list[Tensor]: A list containing two tensors:
            - y: The quantized output tensor
            - scale: The scale values for each token
    
    Restrictions:
        Only supports symmetric quantization. 
    """
    param = {}
    y = Tensor()
    scale = Tensor()
    node = Node('DynamicQuant', param, [input_], [y, scale])
    get_default_net().push_node(node)
    return [y, scale]


def quantize_per_channel(input_: Tensor, scales: Tensor, zero_points: Tensor) -> Tensor:
    """
    Performs per-channel quantization on the input tensor.
    
    Applies quantization using separate scale and zero_point values for each channel
    of the input tensor.

    Calculation formula:
        y = round(input_/scale) + zero_points

    Args:
        input_ (Tensor): The input tensor to be quantized
        scales (Tensor): Scale values for each channel
        zero_points (Tensor): Zero point values for each channel
        
    Returns:
        Tensor: The quantized output tensor
    """
    out = Tensor()
    param = {
        "elewiseType": "ELEWISE_QUANT_PER_CHANNEL",
    }
    node = Node('Elewise', param, [input_, scales, zero_points], [out])
    get_default_net().push_node(node)
    return out


def dequantize(input_: Tensor, weight_scale: Tensor, output_dtype: DataType, activation_scale=None, bias=None) -> Tensor:
    """
    Dequantizes the input tensor with optional bias addition and activation scaling.
    
    Converts quantized input back to specified output data type, with support for
    weight scaling, activation scaling, and bias addition.

    Calculation formula:
        y = (input_+ zero_points) * scale

    Args:
        input_ (Tensor): Quantized input tensor to be dequantized
        weight_scale (Tensor): Scale values for weight dequantization
        output_dtype (DataType): Target data type for dequantized output
        activation_scale (Tensor, optional): Scale values for activation dequantization
        bias (Tensor, optional): Bias tensor to be added after dequantization
        
    Returns:
        Tensor: Dequantized output tensor in the specified data type
    
    Restrictions:
        This operation doesn't support Atlas 300I DUO.
    """
    inputs = [input_, weight_scale]
    if activation_scale is not None:
        inputs.append(activation_scale)
    if bias is not None:
        inputs.append(bias)

    out = Tensor()
    param = {
        "outputDtype": output_dtype.value,
        "hasActivateScale": activation_scale is not None,
        "hasBias": bias is not None
    }
    node = Node('DequantBias', param, inputs, [out])
    get_default_net().push_node(node)
    return out