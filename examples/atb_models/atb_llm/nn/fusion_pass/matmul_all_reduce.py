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
from atb_llm.nn.distributed.distributed import get_backend


class MatmulAllReducePass(FusionPassBase):
    """
    This fusion pass identifies a pattern of two consecutive nodes: a linear node and an AllReduce node,
    which is replaced by a single fusion node.
    """
    def __init__(self):
        """
        This fusion pass will be automatically applied to the following calculations:
        linear = Linear('weight')
        linear_out = linear(x)
        out = nn.distributed.all_reduce(send_tensor=linear_out, rank=rank,
                                        rank_size=world_size)
        Accuracy of the fused operation:
        Due to differences in implementation, it fluctuates within a range of 0.01 compared to the unfused version.

        Restrictions:
        This fusion pass does not support Atlas 300I DUO because Atlas 300I DUO does not support LCCL backend.
        """
        super().__init__()
        # input
        x = Tensor()
        weight = Tensor()
        # output
        all_reduce_out = Tensor()
        # internal
        linear_out = Tensor()
        # nodes before fusing
        node_linear = Node("Linear", {}, [x, weight], [linear_out])
        self.nodes_before_fusing.append(node_linear)
        node_all_reduce = Node("AllReduce", {}, [linear_out], [all_reduce_out])
        self.nodes_before_fusing.append(node_all_reduce)
        # nodes after fusing
        node_linear_all_reduce = Node("LinearParallel", {"type": 'LINEAR_ALL_REDUCE', 'backend': 'lcoc'}, [x, weight], [all_reduce_out])
        self.nodes_after_fusing.append(node_linear_all_reduce)

    def _check_param_by_pass(self, target_graph) -> bool:
        node_0_match = not target_graph[0].op_param['hasBias']
        node_1_match = get_backend(target_graph[1].op_param['processGroup']) == 'lccl'
        if node_0_match and node_1_match:
            return True
        return False

    def _update_param_by_pass(self, target_graph, nodes_after_fusing_for_return):
        nodes_after_fusing_for_return[0].op_param['processGroup'] = target_graph[1].op_param.get('processGroup', 0)
        nodes_after_fusing_for_return[0].op_param['transWeight'] = target_graph[0].op_param.get('transposeB', True)