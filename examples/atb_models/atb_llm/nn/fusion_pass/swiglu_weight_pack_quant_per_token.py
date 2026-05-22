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


class SwigluWeightPackQuantPerTokenPass(FusionPassBase):
    """
    The fusion pass combines the SwiGLU activation operation and dynamic quantization operation into a single node to improve computational efficiency.
    """

    def __init__(self):
        """
        Initializes the SwigluQuantSwigluWeightPackDynamicQuant fusion pass.
        This pass automatically applies to the following computation patterns:
        swiglu_out = activation(x, ActType.SWIGLU)
        quant_out, scale_out = dynamic_quant(swiglu_out)

        Accuracy of the fused operation:
        Due to implementation and optimization differences and the fact that the output is of type int,
        the result fluctuates within a range of 1 compared to the unfused version.

        Restrictions:
        This fusion pass does not support Atlas 300I DUO.
        """
        super().__init__()
        # Input tensors
        gate_up = Tensor()

        # Output tensors
        swiglu_quant_out = Tensor()
        scale = Tensor()

        # internal
        swiglu_out = Tensor()
        # Nodes before fusion
        swiglu_node = Node('Activation', {'activationType': 'ACTIVATION_SWIGLU_FORWARD'}, [gate_up], [swiglu_out])
        # SwiGLU activation node, input is 'gate_up', output is 'swiglu_out'
        self.nodes_before_fusing.append(swiglu_node)

        quant_node = Node("DynamicQuant", {}, [swiglu_out],
                          [swiglu_quant_out, scale])
        # Dynamic quantization node, input is 'swiglu_out', outputs are 'swiglu_quant_out' and 'scale'
        self.nodes_before_fusing.append(quant_node)

        # Node after fusion
        param = {
            'activateLeft': True, # Activation function is on the left
            'quantMode': 'dynamic', # Quantization mode is dynamic
            "inTensorsNum": 1,  # Number of input tensors is 1
        }
        swiglu_quant_node = Node("DequantSwigluQuant", param, [gate_up], [swiglu_quant_out, scale])
        # Fused SwiGLU activation and quantization node, input is 'gate_up', outputs are 'swiglu_quant_out' and 'scale'
        self.nodes_after_fusing.append(swiglu_quant_node)


    def _check_param_by_pass(self, target_graph: list[Node]) -> bool:
        if torch_npu._C._npu_get_soc_version() in (100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 205):
            return False
        if target_graph[0].op_param['activationType'] == 'ACTIVATION_SWIGLU_FORWARD':
            return True
        return False
