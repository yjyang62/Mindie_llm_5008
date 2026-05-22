# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import abstractmethod
from typing import List

import torch

from ...layers import QuantTypeV3
from ... import nn
from ...nn.modules.linear import TransposeType
from ...nn.tensor import Tensor
from ...utils.quantize.pack_type import DataType


BIAS = "bias"


class BaseLinearMethod:
    @abstractmethod
    def create_weights(self, linear: nn.Module, prefix: str, bias=False) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(self, linear: nn.Module, input_tensor: Tensor) -> Tensor | List[Tensor]:
        raise NotImplementedError


class UnquantizedLinearMethod(BaseLinearMethod):
    def create_weights(self, linear: nn.Module, prefix: str, bias=False) -> None:
        weight_parameter = nn.Parameter(
            prefix=prefix, suffix="weight", enable_auto_transpose=True, enable_nd_nz=True)
        weight_parameter.register_processor(lambda x: x.to(linear.config.torch_dtype))
        linear.register_parameter("weight", weight_parameter)
        if bias:
            bias_parameter = nn.Parameter(prefix=prefix, suffix=BIAS)
            bias_parameter.register_processor(lambda x: x.to(linear.config.torch_dtype))
            linear.register_parameter(BIAS, bias_parameter)
        else:
            linear.register_parameter(BIAS, None)

    def apply(self, linear: nn.Module, input_tensor: Tensor) -> Tensor | List[Tensor]:
        if linear.bias is not None:
            out = nn.functional.linear(input_tensor, linear.weight.get_tensor(), linear.bias.get_tensor(), transpose_b=linear.weight.trans_flag == TransposeType.TRANSPOSE)
        else:
            out = nn.functional.linear(input_tensor, linear.weight.get_tensor(), transpose_b=linear.weight.trans_flag == TransposeType.TRANSPOSE)
        return out


class W8A8PerTensorLinearMethod(BaseLinearMethod):
    def create_weights(self, linear: nn.Module, prefix: str, bias=False) -> None:
        weight_parameter = nn.Parameter(
            prefix=prefix, suffix="weight", enable_auto_transpose=True, enable_nd_nz=True)
        weight_parameter.register_processor(lambda x: x.to(torch.int8))
        linear.register_parameter("weight", weight_parameter)
        linear.register_parameter("input_scale", nn.Parameter(prefix=prefix, suffix="input_scale"))
        linear.register_parameter("input_offset", nn.Parameter(prefix=prefix, suffix="input_offset"))
        linear.register_parameter("deq_scale", nn.Parameter(prefix=prefix, suffix="deq_scale"))
        linear.register_parameter("quant_bias", nn.Parameter(prefix=prefix, suffix="quant_bias"))

    def apply(self, linear: nn.Module, input_tensor: Tensor) -> Tensor | List[Tensor]:
        output_dtype = DataType.ACL_FLOAT16 if linear.config.torch_dtype == torch.float16 else DataType.ACL_BF16
        input_tensor = nn.quantized.quantize_per_channel(
            input_tensor, linear.input_scale.get_tensor(), linear.input_offset.get_tensor())

        out = nn.functional.linear(
            input_tensor, linear.weight.get_tensor(),
            transpose_b=linear.weight.trans_flag == TransposeType.TRANSPOSE)
        out = nn.quantized.dequantize(
            out, linear.deq_scale.get_tensor(), output_dtype=output_dtype, bias=linear.quant_bias.get_tensor())
        return out


class W8A8PerTokenLinearMethod(BaseLinearMethod):
    def create_weights(self, linear: nn.Module, prefix: str, bias=False) -> None:
        weight_parameter = nn.Parameter(
            prefix=prefix, suffix="weight", enable_auto_transpose=True, enable_nd_nz=True)
        weight_parameter.register_processor(lambda tensor: tensor.to(torch.int8))
        linear.register_parameter("weight", weight_parameter)
        
        weight_scale_parameter = nn.Parameter(prefix=prefix, suffix="weight_scale")
        weight_scale_type = torch.float32 if linear.config.torch_dtype == torch.float16 else torch.bfloat16
        weight_scale_parameter.register_processor(lambda tensor: tensor.to(weight_scale_type).flatten())
        linear.register_parameter("weight_scale", weight_scale_parameter)

        if bias:
            bias_parameter = nn.Parameter(prefix=prefix, suffix=BIAS)
            linear.register_parameter(BIAS, bias_parameter)
        else:
            linear.register_parameter(BIAS, None)

    def apply(self, linear: nn.Module, input_tensor: Tensor) -> Tensor | List[Tensor]:
        output_dtype = DataType.ACL_FLOAT16 if linear.config.torch_dtype == torch.float16 else DataType.ACL_BF16
        intermediate_quant_input, intermediate_quant_scale = nn.quantized.quantize_per_token(input_tensor)
        intermediate_quant_scale.reshape(lambda org_shape: [org_shape[0] * org_shape[1]])

        out = nn.functional.linear(
            intermediate_quant_input, linear.weight.get_tensor(),
            transpose_b=linear.weight.trans_flag == TransposeType.TRANSPOSE)
        out = nn.quantized.dequantize(
            out, linear.weight_scale.get_tensor(), output_dtype=output_dtype, activation_scale=intermediate_quant_scale)
        if linear.bias is not None:
            out = out + linear.bias.get_tensor()
        return out


LINEAR_METHOD_ROUTER = {
    QuantTypeV3.FLOAT16: UnquantizedLinearMethod,
    QuantTypeV3.BFLOAT16: UnquantizedLinearMethod,
    QuantTypeV3.W8A8: W8A8PerTensorLinearMethod,
    QuantTypeV3.W8A8_DYNAMIC: W8A8PerTokenLinearMethod
}
