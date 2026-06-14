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

from atb_llm.nn.fusion_pass_base import FusionPassBase
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.node import Node


class SwigluWeightNoPackPass(FusionPassBase):
    """
    This fusion pass identifies a pattern of two consecutive nodes: an activation node and a mul node,
    which is replaced by a single fusion node.

    This fusion pass is suitable for a self-gated activation function: swish, which involves a gate and an up weight
    that are not combined.
    """
    def __init__(self):
        """
        This fusion pass will be automatically applied to the following calculations:

        gate_linear_out = Tensor("gate_linear_out")
        up_linear_out = Tensor("up_linear_out")
        gate_activation_out = nn.functional.activation(gate_linear_out, ActType.SWISH)
        out = gate_activation_out * up_linear_out

        Accuracy of the fused operation:
        Due to differences in implementation, it fluctuates within a range of 0.001 compared to the unfused version.
        """
        super().__init__()
        # input
        gate = Tensor()
        up = Tensor()
        # output
        output = Tensor()
        # internal
        concat_out = Tensor()
        gate_out = Tensor()
        # nodes before fusing
        node_gate_swish = Node("Activation", {"activationType": "ACTIVATION_SWISH"}, [gate], [gate_out])
        self.nodes_before_fusing.append(node_gate_swish)
        node_mul = Node("Elewise", {"ElewiseType": "ELEWISE_MUL"}, [gate_out, up], [output])
        self.nodes_before_fusing.append(node_mul)
        # nodes after fusing
        node_concat = Node("Concat", {"concatDim": -1}, [gate, up], [concat_out])
        self.nodes_after_fusing.append(node_concat)
        node_swiglu = Node("Activation", {"activationType": "ACTIVATION_SWIGLU_FORWARD"}, [concat_out], [output])
        self.nodes_after_fusing.append(node_swiglu)

    def _check_param_by_pass(self, target_graph: list[Node]) -> bool:
        node_0_match = target_graph[0].op_param['activationType'] == 'ACTIVATION_SWISH'
        node_1_match = target_graph[0].op_param['scale'] == 1.0 and target_graph[0].op_param['dim'] == -1
        node_2_match = target_graph[1].op_param['elewiseType'] == 'ELEWISE_MUL'
        if node_0_match and node_1_match and node_2_match:
            return True
        return False
