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


class SwigluWeightPackPass(FusionPassBase):
    """
    This fusion pass identifies a pattern of three consecutive nodes: a split node, an activation node and a mul node,
    which is replaced by a single fusion node.

    This fusion pass is suitable for a self-gated activation function: swish, with packed gate and up weight.
    """
    def __init__(self):
        """
        This fusion pass will be automatically applied to the following calculations:

        gate_up_linear_out = Tensor("gate_up_linear_out")
        gate_linear_out, up_linear_out = nn.functional.split(gate_up_linear_out, split_dim=-1, split_num=2)
        gate_activation_out = nn.functional.activation(gate_linear_out, ActType.SWISH)
        out = gate_activation_out * up_linear_out

        Accuracy of the fused operation:
        Due to differences in implementation, it fluctuates within a range of 0.001 compared to the unfused version.
        """
        super().__init__()
        # input
        gate_up = Tensor()
        # output
        output = Tensor()
        # internal
        gate_linear_out = Tensor()
        up_linear_out = Tensor()
        gate_activation_out = Tensor()
        # nodes before fusing
        node_split = Node("Split",
            {"splitDim": -1, "splitNum": 2}, [gate_up], [gate_linear_out, up_linear_out])
        self.nodes_before_fusing.append(node_split)
        node_gate_swish = Node("Activation",
            {"activationType": "ACTIVATION_SWISH"}, [gate_linear_out], [gate_activation_out])
        self.nodes_before_fusing.append(node_gate_swish)
        node_mul = Node("Elewise", {"ElewiseType": "ELEWISE_MUL"}, [gate_activation_out, up_linear_out], [output])
        self.nodes_before_fusing.append(node_mul)
        # nodes after fusing
        node_swiglu = Node("Activation", {"activationType": "ACTIVATION_SWIGLU_FORWARD"}, [gate_up], [output])
        self.nodes_after_fusing.append(node_swiglu)

    def _check_param_by_pass(self, target_graph: list[Node]) -> bool:
        node_0_match = target_graph[0].op_param['splitDim'] == -1 and target_graph[0].op_param['splitNum'] == 2
        node_1_match = target_graph[1].op_param['activationType'] == 'ACTIVATION_SWISH'
        node_2_match = target_graph[2].op_param['elewiseType'] == 'ELEWISE_MUL'
        if node_0_match and node_1_match and node_2_match:
            return True
        return False
