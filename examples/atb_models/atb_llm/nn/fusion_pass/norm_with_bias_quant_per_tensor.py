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


class NormWithBiasQuantPerTensorPass(FusionPassBase): 
    """
    A fusion pass that fuses a RMS normalization operation and an element-wise addition operation
    followed by an element-wise quantperchannel operation into a single operation.
    """   
    def __init__(self):
        """
        This fusion pass will be automatically applied to the following calculations:
        
        rmsNorm = RmsNorm("norm", eps=1e-5)
        rmsNorm_out = rmsNorm(x)
        add_out = rmsNorm_out + bias
        norm_quant_out = quantize_per_channel(add_out, scale, offset)
        
        Accuracy of the fused operation:
        Due to differences in implementation, it fluctuates within a range of 1 compared to the unfused version.
        """
        super().__init__()
        # input
        a = Tensor()
        norm_weight = Tensor()
        bias = Tensor()
        scale = Tensor()
        offset = Tensor()            
        # output
        a_norm_quant = Tensor()
        # internal
        a_norm = Tensor()
        a_norm_bias = Tensor()

        # nodes before fusing
        rmsnorm_node = Node('RmsNorm', {'layerType': 'RMS_NORM_NORM'}, [a, norm_weight], [a_norm])
        self.nodes_before_fusing.append(rmsnorm_node)
        node_add = Node("Elewise", {'elewiseType': 'ELEWISE_ADD'}, [a_norm, bias], [a_norm_bias])
        self.nodes_before_fusing.append(node_add)
        quant_node = Node("Elewise", {'elewiseType': 'ELEWISE_QUANT_PER_CHANNEL'}, [a_norm_bias, scale, offset], [a_norm_quant])
        self.nodes_before_fusing.append(quant_node)
        
        # nodes after fusing
        quant_norm_param = {'layerType': 'RMS_NORM_NORM', "normParam": {"quantType": 'QUANT_INT8'}}
        quant_norm_node = Node('RmsNorm', quant_norm_param, [a, norm_weight, bias, scale, offset], [a_norm_quant])
        self.nodes_after_fusing.append(quant_norm_node)

    def _check_param_by_pass(self, target_graph: list[Node]) -> bool:
        if target_graph[0].op_param['layerType'] == 'RMS_NORM_NORM' and \
            target_graph[1].op_param['elewiseType'] == 'ELEWISE_ADD' and \
            target_graph[2].op_param['elewiseType'] == 'ELEWISE_QUANT_PER_CHANNEL':
            return True
        return False

    def _update_param_by_pass(self, target_graph, nodes_after_fusing_for_return):
        nodes_after_fusing_for_return[0].op_param['normParam']['epsilon'] = target_graph[0].op_param['normParam']['epsilon']