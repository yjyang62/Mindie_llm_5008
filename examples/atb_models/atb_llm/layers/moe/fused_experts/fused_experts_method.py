# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch

from atb_llm.layers import QuantTypeV3
from atb_llm.nn.functional import activation, grouped_matmul
from atb_llm.nn.modules.linear import TransposeType
from atb_llm.nn.modules.module import Module
from atb_llm.nn.parameter import Parameter
from atb_llm.nn.tensor import Tensor


class UnquantizedFusedExpertsMethod:
    @staticmethod
    def create_weights(fused_experts: Module, prefix: str, dtype: torch.dtype, bias=False) -> None:
        gate_up_weight = Parameter(
            prefix=f"{prefix}.gate_up_proj", suffix="weight", enable_auto_transpose=True, enable_nd_nz=True)
        gate_up_weight.register_processor(lambda tensor: tensor.to(dtype))
        fused_experts.register_parameter("gate_up_weight", gate_up_weight)

        down_weight = Parameter(
            prefix=f"{prefix}.down_proj", suffix="weight", enable_auto_transpose=True, enable_nd_nz=True)
        down_weight.register_processor(lambda tensor: tensor.to(dtype))
        fused_experts.register_parameter("down_weight", down_weight)

        if bias:
            gate_up_bias = Parameter(
                prefix=f"{prefix}.gate_up_proj", suffix="bias", enable_auto_transpose=True, enable_nd_nz=True)
            gate_up_bias.register_processor(lambda tensor: tensor.to(dtype))
            fused_experts.register_parameter("gate_up_bias", gate_up_bias)

            down_bias = Parameter(
                prefix=f"{prefix}.down_proj", suffix="bias", enable_auto_transpose=True, enable_nd_nz=True)
            down_bias.register_processor(lambda tensor: tensor.to(dtype))
            fused_experts.register_parameter("down_bias", down_bias)
        else:
            fused_experts.register_parameter("gate_up_bias", None)
            fused_experts.register_parameter("down_bias", None)

    @staticmethod
    def apply(fused_experts: Module, sorted_hidden_states: Tensor, group_list: Tensor) -> Tensor:
        gate_up_args = {
            "input_": sorted_hidden_states,
            "weight": fused_experts.gate_up_weight.get_tensor(),
            "group_list": group_list,
            "transpose_b": fused_experts.gate_up_weight.trans_flag == TransposeType.TRANSPOSE
        }
        if fused_experts.gate_up_bias is not None:
            gate_up_args.update({
                "bias": fused_experts.gate_up_bias.get_tensor()
            })
        gate_up_out = grouped_matmul(**gate_up_args)

        swish_out = activation(gate_up_out, fused_experts.act_type)

        down_args = {
            "input_": swish_out,
            "weight": fused_experts.down_weight.get_tensor(),
            "group_list": group_list,
            "transpose_b": fused_experts.down_weight.trans_flag == TransposeType.TRANSPOSE
        }
        if fused_experts.down_bias is not None:
            down_args.update({
                "bias": fused_experts.down_bias.get_tensor()
            })
        down_out = grouped_matmul(**down_args)
        return down_out


FUSED_EXPERTS_METHOD_ROUTER = {
    QuantTypeV3.FLOAT16: UnquantizedFusedExpertsMethod,
    QuantTypeV3.BFLOAT16: UnquantizedFusedExpertsMethod
}
