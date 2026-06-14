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

import torch_npu

from atb_llm.nn.fusion_pass_base import FusionPassBase
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.node import Node
from atb_llm.utils.quantize.pack_type import DataType


HAS_BIAS = "hasBias"


class W8A8LinearDequantPerTokenWithBiasPass(FusionPassBase):
    """
    This fusion pass identifies a pattern of two consecutive nodes: a matmul node and a dequant node,
    which is replaced by a single fusion node.

    When performing inference on quantized models, activations and weights are quantized before
    the matmul operation to reduce memory usage and accelerate inference. During the matmul operation,
    the quantized int8 input tensors are computed, and the result is then dequantized to a floating-point type
    using the quantization parameters.

    This fusion pass is suitable for per-token quantization algorithms. The quantization parameters for the
    activations are dynamically calculated during inference.

    Restrictions:
    This fusion pass does not support Atlas 300I DUO.
    """
    def __init__(self):
        """
        This fusion pass will be automatically applied to the following calculations:

        weight = nn.Parameter(prefix="model", suffix="weight")
        weight_scale = nn.Parameter(prefix="model", suffix="weight_scale")
        activation_scale = nn.Parameter(prefix="model", suffix="activation_scale")
        bias = nn.Parameter(prefix="model", suffix="bias")

        output_dtype = DataType.ACL_BF16 if dtype == torch.bfloat16 else DataType.ACL_FLOAT16
        out = nn.functional.matmul(
            input_tensor, weight.get_tensor(),
            transpose_b=True)
        out = nn.quantized.dequantize(
            out, weight_scale.get_tensor(), output_dtype=output_dtype,
            activation_scale=activation_scale.get_tensor())
        out = out + bias.get_tensor()

        This fusion pass must be enabled in W8A8 quantization.
        """
        super().__init__()
        # input
        input_tensor = Tensor()
        weight = Tensor()
        weight_scale = Tensor()
        activation_scale = Tensor()
        bias = Tensor()
        # output
        out = Tensor()
        # internal
        linear_out = Tensor()
        dequant_out = Tensor()

        # nodes before fusing
        param = {HAS_BIAS: False}
        node_linear = Node("Linear", param, [input_tensor, weight], [linear_out])
        self.nodes_before_fusing.append(node_linear)
        param = {
          "hasActivateScale": True,
          HAS_BIAS: False
        }
        node_dequant = Node("DequantBias", param, [linear_out, weight_scale, activation_scale], [dequant_out])
        self.nodes_before_fusing.append(node_dequant)
        node_add = Node("Elewise", {'elewiseType': 'ELEWISE_ADD'}, [dequant_out, bias], [out])
        self.nodes_before_fusing.append(node_add)

        # node after fusing
        inputs = [input_tensor, weight, weight_scale, activation_scale, bias]
        param = {
            HAS_BIAS: True,
            'hasPerTokenScale': True
        }
        linear_dequant_node = Node('W8A8MatMul', param, inputs, [out])
        self.nodes_after_fusing.append(linear_dequant_node)

    def _check_param_by_pass(self, target_graph: list[Node]) -> bool:
        if torch_npu._C._npu_get_soc_version() in (100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 205):
            return False
        first_node_param_match = not target_graph[0].op_param[HAS_BIAS] and \
            not target_graph[0].op_param['transposeA']
        second_node_param_match = target_graph[1].op_param['hasActivateScale'] and \
            not target_graph[1].op_param[HAS_BIAS]
        if first_node_param_match and second_node_param_match and \
            target_graph[2].op_param['elewiseType'] == 'ELEWISE_ADD':
            return True
        return False

    def _update_param_by_pass(self, target_graph, nodes_after_fusing_for_return):
        # linear
        nodes_after_fusing_for_return[0].op_param["transposeB"] = \
            target_graph[0].op_param.get("transposeB", True)
        # dequant
        if target_graph[1].op_param.get("outputDtype", DataType.ACL_DT_UNDEFINED) == DataType.ACL_FLOAT16:
            nodes_after_fusing_for_return[0].op_param["isBF16"] = False
