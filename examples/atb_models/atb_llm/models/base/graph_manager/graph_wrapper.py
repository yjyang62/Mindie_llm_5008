# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
from abc import ABC, abstractmethod
from typing import List, Dict

import torch

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM


class ATBGraphWrapper(ABC):
    """Base class for ATB graphs, representing a feature.
    
    Attributes:
        feature_name (str): feature name, see enum FeatureType.
        feature_params (dict): Special parameters of the feature.
        atb_graph: Instance of torch.classes.ModelTorch.ModelTorch.
    """
    def __init__(self):
        self.feature_name = ""
        self.feature_params = {}
        self.atb_graph = None
    
    def set_param(self, model_type: str, params: Dict):
        self.atb_graph = torch.classes.ModelTorch.ModelTorch(model_type)
        self.atb_graph.set_param(json.dumps({**params, **self.feature_params}))

    def set_weight(self, weights):
        self.atb_graph.set_weight(weights)
    
    def set_kv_cache(self, k_caches, v_caches):
        self.atb_graph.set_kv_cache(k_caches, v_caches)
    
    def execute(self, inputs, runtime_params, **kwargs):
        model_out = self.atb_graph.execute(inputs, runtime_params)
        return model_out

    @abstractmethod
    def activate(self, context: FlashForCausalLM, runtime_params, **kwargs) -> bool:
        """Determine whether to activate this atb_graph."""
        pass


class CombinedATBGraphWrapper(ATBGraphWrapper):
    """ATBGraphWrapper class that automatically merges the parameters and functionalities of multiple atb_graphs."""
    def __init__(self, atb_graphs: List[ATBGraphWrapper]):
        super().__init__()
        # Merge the feature_params of all atb_graphs.
        combined_params = {}
        for graph in atb_graphs:
            combined_params.update(graph.feature_params)
        
        self.feature_name = "_".join([g.feature_name for g in atb_graphs])
        self.feature_params = combined_params
        self.atb_graphs = atb_graphs
    
    def activate(self, context: FlashForCausalLM, runtime_params, **kwargs) -> bool:
        # When all atb_graphs are activated, combined engine can be activated
        for graph in self.atb_graphs:
            if not graph.activate(context, runtime_params, **kwargs):
                return False
        return True