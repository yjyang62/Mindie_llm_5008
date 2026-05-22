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
from typing import List, Dict

import torch

from atb_llm.models.base.graph_manager.graph_wrapper import ATBGraphWrapper
from atb_llm.models.base.graph_manager.compatible_matrix import FeatureType
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM, LayerWiseAttr, LwdLayerStatus, DistributedType


def get_layerwise_decode_graph(attr: LayerWiseAttr) -> ATBGraphWrapper:
    if attr is not None and attr.split_type == DistributedType.CLOUD:
        return LayerwiseCloudDecodeGraphWrapper(attr)
    return LayerwiseEdgeDecodeGraphWrapper(attr)


class LayerwiseDecodeGraphWrapper(ATBGraphWrapper):
    def __init__(self, attr: LayerWiseAttr):
        super().__init__()
        self.feature_name = FeatureType.LAYERWISE_DECODE
        self.feature_params = {"layerwiseDisaggregated": True}
        self.attr = attr

    def activate(self, context: FlashForCausalLM, runtime_params, **kwargs) -> bool:
        is_prefill = kwargs.get("is_prefill", False)
        return not is_prefill
    
    def add_feature_params(self, params: Dict):
        self.feature_params.update(params)


class LayerwiseEdgeDecodeGraphWrapper(LayerwiseDecodeGraphWrapper):
    def __init__(self, attr: LayerWiseAttr):
        super().__init__(attr)

    def set_param(self, model_type: str, params: Dict):
        head_params = params['head']
        self.head_graph = torch.classes.ModelTorch.ModelTorch(model_type)
        self.head_graph.set_param(json.dumps({**head_params, **self.feature_params}))

        tail_params = params['tail']
        self.tail_graph = torch.classes.ModelTorch.ModelTorch(model_type)
        self.tail_graph.set_param(json.dumps({**tail_params, **self.feature_params}))

    def set_weight(self, weights):
        head_weights = weights['head']
        self.head_graph.set_weight(head_weights)
        
        tail_weights = weights['tail']
        self.tail_graph.set_weight(tail_weights)
    
    def set_kv_cache(self, k_caches, v_caches):
        head_k_caches = k_caches['head']
        head_v_caches = v_caches['head']
        self.head_graph.set_kv_cache(head_k_caches, head_v_caches)
        
        tail_k_caches = k_caches['tail']
        tail_v_caches = v_caches['tail']
        self.tail_graph.set_kv_cache(tail_k_caches, tail_v_caches)
    
    def execute(self, inputs, runtime_params, **kwargs):
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage", None)
        if exe_stage is None: # warmup
            acl_model_out = self.head_graph.execute(inputs, runtime_params)
            inputs[0] = acl_model_out[0]
            acl_model_out = self.tail_graph.execute(inputs, runtime_params)
            return acl_model_out
        if exe_stage.start_exec_layer == 0: # first layer
            acl_model_out = self.head_graph.execute(inputs, runtime_params)
            return acl_model_out
        if exe_stage.end_exec_layer == 1: # last layer
            acl_model_out = self.tail_graph.execute(inputs, runtime_params)
            return acl_model_out
        return None


class LayerwiseCloudDecodeGraphWrapper(LayerwiseDecodeGraphWrapper):
    def __init__(self, attr: LayerWiseAttr):
        super().__init__(attr)

    def set_param(self, model_type: str, params: Dict):
        self.graph = torch.classes.ModelTorch.ModelTorch(model_type)
        self.graph.set_param(json.dumps({**params, **self.feature_params}))

    def set_weight(self, weights):
        self.graph.set_weight(weights)
    
    def set_kv_cache(self, k_caches, v_caches):
        self.graph.set_kv_cache(k_caches, v_caches)
    
    def execute(self, inputs, runtime_params, **kwargs):
        model_out = self.graph.execute(inputs, runtime_params)
        return model_out