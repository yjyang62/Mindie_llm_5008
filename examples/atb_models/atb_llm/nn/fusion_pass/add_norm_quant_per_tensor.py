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


class AddNormQuantPerTensorPass(FusionPassBase): 
    """
    A fusion pass that fuses an element-wise addition operation and RMS normalization operation 
    followed by an element-wise quant_per_channel operation into a single operation.
    """   
    def __init__(self):
        """
        This fusion pass will be automatically applied to the following calculations:

        rmsNorm = RmsNorm("norm", eps=1e-5)
        add_out = x + residual
        rmsNorm_out = rmsNorm(add_out)
        add_norm_quant_out = quantize_per_channel(rmsNorm_out, scale, offset)

        Accuracy of the fused operation:
        Due to differences in implementation, it fluctuates within a range of 1 compared to the unfused version.
        """
        super().__init__()
        # input
        a = Tensor()
        b = Tensor()
        scale = Tensor()
        offset = Tensor()
        norm_weight = Tensor()
        # output
        a_b_norm_quant = Tensor()
        a_b = Tensor()
        rmsout = Tensor()
        # internal
        a_b = Tensor()
        a_b_norm = Tensor()

        # nodes before fusing
        node_add = Node("Elewise", {'elewiseType': 'ELEWISE_ADD'}, [a, b], [a_b])
        self.nodes_before_fusing.append(node_add)
        rmsnorm_node = Node('RmsNorm', {'layerType': 'RMS_NORM_NORM'}, [a_b, norm_weight], [a_b_norm])
        self.nodes_before_fusing.append(rmsnorm_node)
        quant_node = Node("Elewise", {'elewiseType': 'ELEWISE_QUANT_PER_CHANNEL'}, [a_b_norm, scale, offset], [a_b_norm_quant])
        self.nodes_before_fusing.append(quant_node)
        
        # node after fusing
        quant_norm_node = Node('AddRmsNormQuant', {'epsilon': 1e-5}, [a, b, norm_weight, scale, offset], [a_b_norm_quant, rmsout, a_b])
        self.nodes_after_fusing.append(quant_norm_node)

    def _check_param_by_pass(self, target_graph: list[Node]) -> bool:
        if target_graph[0].op_param['elewiseType'] == 'ELEWISE_ADD' and \
            target_graph[1].op_param['layerType'] == 'RMS_NORM_NORM' and \
            target_graph[2].op_param['elewiseType'] == 'ELEWISE_QUANT_PER_CHANNEL':
            return True
        return False
    
    def _update_param_by_pass(self, target_graph, nodes_after_fusing_for_return):
        nodes_after_fusing_for_return[0].op_param['epsilon'] = target_graph[1].op_param['normParam']['epsilon']