# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from atb_llm.models.base.graph_manager.graph_manager import CombinedATBGraphWrapper, ATBGraphWrapper
from atb_llm.models.base.graph_manager.layerwise_decode_graph_wrapper import get_layerwise_decode_graph
from atb_llm.models.base.graph_manager.layerwise_prefill_graph_wrapper import get_layerwise_prefill_graph
from atb_llm.models.base.graph_manager.compatible_matrix import FeatureType
from atb_llm.models.base.flash_causal_lm import LayerWiseAttr


class LayerwiseCombinedATBGraphWrapper(CombinedATBGraphWrapper):
    """ATBGraphWrapper class that merges the parameters and functionalities of atb_graphs and layerwise base graph"""
    def __init__(self, atb_graphs):
        super().__init__(atb_graphs)
        # The first graph in atb_graphs is LayerwisePrefillGraphWrapper
        base_graph = atb_graphs[0]
        config = getattr(base_graph, "config", None)
        layerwise_attr: LayerWiseAttr = getattr(base_graph, "attr", None)
        if base_graph.feature_name == FeatureType.LAYERWISE_PREFILL:
            self.atb_graph = get_layerwise_prefill_graph(config, layerwise_attr)
        else:
            self.atb_graph = get_layerwise_decode_graph(layerwise_attr)
        
        
    def set_param(self, model_type, params):
        self.atb_graph.add_feature_params(self.feature_params)
        self.atb_graph.set_param(model_type, params)
        
    def set_weight(self, weights):
        self.atb_graph.set_weight(weights)
    
    def set_kv_cache(self, k_caches, v_caches):
        self.atb_graph.set_kv_cache(k_caches, v_caches)
    
    def execute(self, inputs, runtime_params, **kwargs):
        model_out = self.atb_graph.execute(inputs, runtime_params, **kwargs)
        return model_out
       