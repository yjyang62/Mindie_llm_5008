# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from ... import nn
from ...utils.loader.safetensor_file_loader import SafetensorFileLoader
from ...models.base.config import BaseConfig
from ...nn.network_manager import get_default_net


class BaseLayer(nn.Module, ABC):
    def __init__(self, config: BaseConfig, file_loader: SafetensorFileLoader, prefix: str, layer_idx: int, **kwargs):        
        super().__init__()
        self.config = config
        self.prefix = f"{prefix}.layers.{layer_idx}"
        self.layer_idx = layer_idx

        self.self_attn = None
        self.mlp = None
        self.input_layernorm = None
        self.post_attention_layernorm = None
    
    def __call__(self, inputs, cos_emb, sin_emb, k_cache, v_cache, slots, attention_mask=None,
        seq_len=None, block_table=None, token_offset=None, layer_ids=None, is_prefill: bool = True, **kwargs):
        """
        Should not be overwrited.
        """
        out = self.forward(inputs, cos_emb, sin_emb, k_cache, v_cache, slots, attention_mask,
            seq_len, block_table, token_offset, layer_ids, is_prefill, **kwargs)
        get_default_net().cut()
        return out

    @abstractmethod
    def forward(self, inputs, cos_emb, sin_emb, k_cache, v_cache, slots, attention_mask=None,
        seq_len=None, block_table=None, token_offset=None, layer_ids=None, is_prefill: bool = True, **kwargs):
        pass


class BaseModel(nn.Module, ABC):
    def __init__(self, config: BaseConfig, file_loader: SafetensorFileLoader, prefix: str = "model", **kwargs):
        super().__init__()
        self.config = config
        self.mapping = file_loader.mapping
        self.prefix = prefix

        self.embed_tokens = None
        self.layers = None
        self.norm = None
    
    def __call__(self, input_ids, position_ids, cosine_table, sine_table, k_caches, v_caches, slots_mapping=None,
        attention_mask=None, seq_len=None, block_table=None, token_offset=None, kv_cache_idx=None,
        is_prefill: bool = True, **kwargs):
        """
        Should not be overwrited.
        """
        out = self.forward(input_ids, position_ids, cosine_table, sine_table, k_caches, v_caches, slots_mapping,
            attention_mask, seq_len, block_table, token_offset, kv_cache_idx, is_prefill, **kwargs)
        return out

    @abstractmethod
    def forward(self, input_ids, position_ids, cosine_table, sine_table, k_caches, v_caches, slots_mapping=None,
                attention_mask=None, seq_len=None, block_table=None, token_offset=None, kv_cache_idx=None,
                is_prefill: bool = True, **kwargs):
        pass
