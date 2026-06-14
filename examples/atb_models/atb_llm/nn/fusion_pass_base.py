# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
import time
from abc import abstractmethod


class FusionPassBase:
    """Base class for graph fusion pass."""
    def __init__(self):
        """Initialize fusion pass."""
        self.name = None  # Name of the fusion pass
        self.nodes_before_fusing = []  # Original nodes before fusion
        self.nodes_after_fusing = []  # Fused nodes after optimization
        self._tensor_pass_target_map = {}
        self._target_key_tensor_map = {}

    @classmethod 
    def _check_view_tensor(cls, target_tensors, pass_tensors):
        """
        Verify tensor view consistency between target and pass tensors.
        
        Args:
            target_tensors: List of tensors from target graph
            pass_tensors: List of tensors from fusion pass
            
        Returns:
            bool: True if the two view tensors are consistent, False otherwise
        """  
        for i, _ in enumerate(target_tensors):
            if target_tensors[i].view_tensor is not None and pass_tensors[i].view_tensor is None:
                return False
            if target_tensors[i].view_tensor is None and pass_tensors[i].view_tensor is not None:
                return False
        return True

    @abstractmethod
    def _check_param_by_pass(self, target_graph) -> bool:
        """
        Abstract method to validate fusion parameters (to be implemented by subclasses).
        
        Args:
            target_graph: List of nodes in the target graph
            
        Returns:
            bool: True if parameters are valid for fusion
        """   
        pass

    def get_fused_nodes_by_pass(self, target_graph):
        """
        Generate fused nodes after applying the fusion graph.
        
        Args:
            target_graph: List of nodes in the target graph
            
        Returns:
            list: Deep copy of fused nodes with updated tensor params, or None if matching fails
        """
        if not self.__match_topo_success_by_pass(target_graph) or not self._check_param_by_pass(target_graph):
            return None
        nodes_after_fusing_for_return = copy.deepcopy(self.nodes_after_fusing)
        for i, _ in enumerate(self.nodes_after_fusing):
            pass_node = self.nodes_after_fusing[i]
            for j, _ in enumerate(pass_node.in_tensors):
                pass_key = id(pass_node.in_tensors[j])
                if pass_key not in self._tensor_pass_target_map:
                    tensor_name = str(pass_key) + ":" + str(time.time())
                    nodes_after_fusing_for_return[i].in_tensors[j].name = tensor_name
                else:
                    target_tensor = self._target_key_tensor_map[self._tensor_pass_target_map[pass_key]]
                    nodes_after_fusing_for_return[i].in_tensors[j] = copy.deepcopy(target_tensor)
            for j, _ in enumerate(pass_node.out_tensors):
                pass_key = id(pass_node.out_tensors[j])
                if pass_key not in self._tensor_pass_target_map:
                    tensor_name = str(pass_key) + ":" + str(time.time())
                    nodes_after_fusing_for_return[i].out_tensors[j].name = tensor_name
                else:
                    target_tensor = self._target_key_tensor_map[self._tensor_pass_target_map[pass_key]]
                    nodes_after_fusing_for_return[i].out_tensors[j] = copy.deepcopy(target_tensor)
        self._update_param_by_pass(target_graph, nodes_after_fusing_for_return)
        return nodes_after_fusing_for_return

    def __match_topo_success_by_pass(self, target_graph):
        """
        Verify topological match between target graph and fusion pattern.
        
        Args:
            target_graph: List of nodes in the target graph
            
        Returns:
            bool: True if topological group graph matches, False otherwise
        """
        tensor_key_map = {}
        self._tensor_pass_target_map = {}
        self._target_key_tensor_map = {}
        if len(self.nodes_before_fusing) != len(target_graph):
            return False
        for i, _ in enumerate(self.nodes_before_fusing):
            if target_graph[i].op_type != self.nodes_before_fusing[i].op_type:
                return False
            if len(target_graph[i].in_tensors) != len(self.nodes_before_fusing[i].in_tensors) or \
                len(target_graph[i].out_tensors) != len(self.nodes_before_fusing[i].out_tensors):
                return False
            if not self._check_view_tensor(target_graph[i].out_tensors, self.nodes_before_fusing[i].out_tensors):
                return False
            for j, _ in enumerate(target_graph[i].in_tensors):
                target_key = id(target_graph[i].in_tensors[j])
                self._target_key_tensor_map[target_key] = target_graph[i].in_tensors[j]
                pass_key = id(self.nodes_before_fusing[i].in_tensors[j])
                if target_key not in tensor_key_map:
                    tensor_key_map[target_key] = pass_key
                    self._tensor_pass_target_map[pass_key] = target_key
                elif tensor_key_map[target_key] != pass_key:
                    return False
            for j, _ in enumerate(target_graph[i].out_tensors):
                target_key = id(target_graph[i].out_tensors[j])
                self._target_key_tensor_map[target_key] = target_graph[i].out_tensors[j]
                pass_key = id(self.nodes_before_fusing[i].out_tensors[j])
                if target_key not in tensor_key_map:
                    tensor_key_map[target_key] = pass_key
                    self._tensor_pass_target_map[pass_key] = target_key
                elif tensor_key_map[target_key] != pass_key:
                    return False
        return True
    
    def _update_param_by_pass(self, target_graph, nodes_after_fusing_for_return):
        """
        Abstract method to update parameters after fusion (to be implemented by subclasses).
        
        Args:
            nodes_after_fusing_for_return: List of fused nodes needing parameter updates
        """
        pass