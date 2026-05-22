# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import OrderedDict, List, Dict
from itertools import combinations

from atb_llm.models.base.graph_manager.graph_wrapper import ATBGraphWrapper, CombinedATBGraphWrapper
from atb_llm.models.base.graph_manager.prefill_graph_wrapper import PrefillGraphWrapper
from atb_llm.models.base.graph_manager.decode_graph_wrapper import DecodeGraphWrapper
from atb_llm.models.base.graph_manager.compatible_matrix import COMPATIBLE_MATRIX 
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.log.logging import logger


class ATBGraphManager:
    """A class manages all the `ATBGraphWrapper` objects."""
    def __init__(self, prefill_graph: ATBGraphWrapper = None, decode_graph: ATBGraphWrapper = None,
                 combine_graph_cls=CombinedATBGraphWrapper, compatible_matrix: Dict = None):
        """Constructor

        Args:
            compatible_matrix (Dict): A compatible matrix to indicate whether features are compatible with each other.
                If no parameters are passed in, the default setting will be used.
        """
        self.base_graphs = [
            prefill_graph if prefill_graph is not None else PrefillGraphWrapper(),
            decode_graph if decode_graph is not None else DecodeGraphWrapper()
        ] # Base Feature
        self.combine_graph_cls = combine_graph_cls 
        self.registered_graphs: List[ATBGraphWrapper] = [] # Registered Feature
        if compatible_matrix is not None:
            COMPATIBLE_MATRIX.update(compatible_matrix)
        self.compatible_graphs = COMPATIBLE_MATRIX
        self.combined_graphs: List[CombinedATBGraphWrapper] = [] # Combined Feature
        self._graph_list: List[ATBGraphWrapper] = [] # All available graphs
    
    @classmethod
    def _find_best_matching(cls, feature_name, specified_dict: OrderedDict):
        """Find best matching in specified_dict.
        
        Args:
            feature_name (str): graph's feature name.
            specified_dict (OrderedDict): Special params
        
        Returns:
            The result of best matching's value in specified_dict.
        """
        if specified_dict is None:
            return None
        
        matches = []
        feature_names = set(feature_name.split('_'))
        for param_key in specified_dict.keys():
            param_keys = param_key.split('_')
            if all(part in feature_names for part in param_keys):
                matches.append((param_key, len(param_keys)))
        
        if len(matches) == 0:
            return None

        param_key_order = {key: idx for idx, key in enumerate(specified_dict.keys())}
        
        best_match = max(
            matches,
            key=lambda x: (x[1], -param_key_order[x[0]])
        )
        
        return specified_dict[best_match[0]]

    def register_graph(self, graph_class: ATBGraphWrapper) -> None:
        """Register a feature

        Args:
            graph_class (ATBGraphWrapper): A feature needed to be registered in ATBGraphManager.
        """
        self.registered_graphs.append(graph_class)
    
    def set_param(self, model_type, params, specified_params: OrderedDict = None):
        """Generate combined features and set param of all the manager's ATB graphs

        Args:
            model_type (str): Cpp graph's class name, e.g: qwen_QwenDecoderModel.
            params (dict): Cpp graph's param.
            specified_params (OrderedDict): Special params used by certain features in the manager.
                e.g: when specified_params = {"decode": param1, "decode_flashcomm": param2}, The graph with
                feature_name "decode_flashcomm" uses param2, and other graphs with feature_name "decode_xxx"
                use param1
        """
        self._generate_combinations()
        for graph in self.combined_graphs + self.base_graphs:
            matched_params = self._find_best_matching(graph.feature_name, specified_params)
            graph.set_param(model_type, matched_params if matched_params is not None else params)
            self._graph_list.append(graph)
    
    def set_weight(self, weights, specified_weights: OrderedDict = None):
        """Set weight of all the manager's available graphs

        Args:
            weights (List[torch.Tensor]): Cpp graph's weight.
            specified_weights (OrderedDict): Special weights used by certain features in the manager.
                The pattern is the same as func set_param
        """
        for graph in self._graph_list:
            matched_weights = self._find_best_matching(graph.feature_name, specified_weights)
            graph.set_weight(matched_weights if matched_weights is not None else weights)
    
    def set_kv_cache(self, k_caches, v_caches, specified_kv_caches: OrderedDict = None):
        """Set kv_cache of all the manager's available graphs

        Args:
            k_caches (List[torch.Tensor]): Cpp graph's k_caches.
            v_caches (List[torch.Tensor]): Cpp graph's v_caches.
            specified_kv_caches (OrderedDict): Special kv_caches used by certain features in the manager.
                The pattern is the same as func set_param, and specified_kv_caches will use 'k' as the key of
                k_caches, 'v' as the key of v_caches, such as {"decode": {"k": k_caches, "v": v_caches}}
        """
        for graph in self._graph_list:
            matched_kv_caches = self._find_best_matching(graph.feature_name, specified_kv_caches)
            graph.set_kv_cache(matched_kv_caches["k"] if matched_kv_caches is not None else k_caches, \
                matched_kv_caches["v"] if matched_kv_caches is not None else v_caches)
    
    def select_and_execute(self, context: FlashForCausalLM, inputs, runtime_param, **kwargs):
        """Choose a graph from manager's available graphs and execute

        Args:
            context (FlashForCausalLM): All the necessary parameters needed to choose graph.
            inputs (List[torch.Tensor]): Cpp graph's input tensors.
            runtime_param (str): A json str to indicate cpp graph's host tensors.
            kwargs: All the inference inputs needed to choose the graph.

        Returns:
            graph's output. When no graph is chosen, it will be None.
        """
        for graph in self._graph_list:
            if graph.activate(context, runtime_param, **kwargs):
                logger.debug(f"graph {graph.feature_name} is activated.")
                return graph.execute(inputs, runtime_param, **kwargs)
        return None
    
    def _check_compatibility(self, atb_graphs: List[ATBGraphWrapper]) -> bool:
        """Check whether the combination can be combined according to the COMPATIBLE_MATRIX.

        Args:
            atb_graphs (List[ATBGraphWrapper]): The graph combination
        
        Returns:
            bool: The result of check.
        """
        for graph in atb_graphs:
            if not all(other.feature_name in self.compatible_graphs[graph.feature_name] \
                       for other in atb_graphs if other is not graph):
                return False
        return True

    def _generate_combinations(self):
        """Auto-generate combined features.
        All the available combined features are in self.combined_graphs
        """
        n = len(self.registered_graphs)

        # 生成所有可能的特性组合
        for base_graph in self.base_graphs:
            for num_graphs in range(n, 0, -1):
                for combined_indices in combinations(range(n), num_graphs):
                    combined_classes = [base_graph]
                    combined_classes.extend([self.registered_graphs[i] for i in combined_indices])

                    if self._check_compatibility(combined_classes):
                        # 创建引擎实例
                        combined_graphs = []
                        for graph_class in combined_classes:
                            combined_graphs.append(graph_class)
                        combined_graph = self.combine_graph_cls(combined_graphs)
                        self.combined_graphs.append(combined_graph)