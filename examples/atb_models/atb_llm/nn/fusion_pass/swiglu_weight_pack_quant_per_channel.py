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


class SwigluWeightPackQuantPerChannelPass(FusionPassBase):
    """
    This fusion pass identifies a pattern of two consecutive nodes: a SwiGLU activation node and a static quantization node,
    which are replaced by a single fusion node.
    """
    def __init__(self):
        """
        This fusion pass will be automatically applied to the following calculations:
        swiglu_out = activation(x, ActType.SWIGLU)
        quant_out = quantize_per_channel(swiglu_out, quant_scales, quant_offset)

        Accuracy of the fused operation:
        Due to implementation and optimization differences and the fact that the output is of type int,
        the result fluctuates within a range of 1 compared to the unfused version.

        Restrictions:
        This fusion pass does not support Atlas 300I DUO.
        """
        super().__init__()
        # input
        gate_up = Tensor()
        scale = Tensor()
        # output
        swiglu_quant_out = Tensor()
        # internal
        swiglu_out = Tensor()
        quant_scales = Tensor()
        quant_offset = Tensor()

        # nodes before fusing
        swiglu_node = Node('Activation', {'activationType': 'ACTIVATION_SWIGLU_FORWARD'}, [gate_up], [swiglu_out])
        self.nodes_before_fusing.append(swiglu_node)
        quant_node = Node("Elewise", {"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"}, [swiglu_out, quant_scales, quant_offset],
                          [swiglu_quant_out])
        self.nodes_before_fusing.append(quant_node)
        # nodes after fusing
        param = {
            'activateLeft': True,
            'quantMode': 'static',
            "inTensorsNum": 3,
        }
        swiglu_quant_node = Node("DequantSwigluQuant", param, [gate_up, quant_scales, quant_offset], [swiglu_quant_out, scale])
        self.nodes_after_fusing.append(swiglu_quant_node)

    def _check_param_by_pass(self, target_graph: list[Node]) -> bool:
        if torch_npu._C._npu_get_soc_version() in (100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 205):
            return False
        if target_graph[0].op_param['activationType'] == 'ACTIVATION_SWIGLU_FORWARD' and \
            target_graph[1].op_param['elewiseType'] == 'ELEWISE_QUANT_PER_CHANNEL':
            return True
        return False