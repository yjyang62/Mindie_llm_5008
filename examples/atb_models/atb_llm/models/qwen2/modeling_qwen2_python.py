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
# Implement part of this file based on transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from ... import nn
from ...nn.functional import gather
from ...layers.attention.attention import Attention
from ...layers.mlp.mlp import Mlp
from ...models.base.mindie_llm_config import ModelStatus
from ...models.base.config import BaseConfig
from ...models.base.modeling_python import BaseLayer, BaseModel
from ...layers.embedding.word_embedding import ParallelEmbedding
from ...layers.linear.linear import MergedColumnParallelLinear, RowParallelLinear
from ...layers.norm.normalization import RmsNorm
from ...utils.loader.safetensor_file_loader import SafetensorFileLoader
from ...utils.mapping import Mapping


QWEN2_EMBEDDING_PARALLEL_THRESHOLD = 152064


class Qwen2Attention(Attention):
    def __init__(
            self,
            config: BaseConfig,
            file_loader: SafetensorFileLoader,
            prefix: str,
            config_metadata: ModelStatus,
            **kwargs
    ) -> None:
        super().__init__(config, file_loader, prefix, config_metadata, **kwargs)

        qkv_linear_names = [f"{self.prefix}.q_proj", f"{self.prefix}.k_proj", f"{self.prefix}.v_proj"]
        o_linear_names = [f"{self.prefix}.o_proj"]

        self.qkv = MergedColumnParallelLinear(
            config, 
            file_loader, 
            qkv_linear_names, 
            bias=config.attention_bias, 
            **kwargs
        )
        self.dense = RowParallelLinear(config, file_loader, o_linear_names, **kwargs)
        
        if config.use_qk_norm:
            self.q_norm = RmsNorm(
                config,
                file_loader,
                f"{self.prefix}.q_norm",
                **kwargs
            )
            self.k_norm = RmsNorm(
                config,
                file_loader,
                f"{self.prefix}.k_norm",
                **kwargs
            )


class Qwen2Mlp(Mlp):
    def __init__(
            self,
            config: BaseConfig,
            file_loader: SafetensorFileLoader,
            prefix: str,
            config_metadata: ModelStatus,
            **kwargs
    ) -> None:
        super().__init__(config, file_loader, prefix, config_metadata, **kwargs)

        gate_up_linear_names = [f"{self.prefix}.gate_proj", f"{self.prefix}.up_proj"]
        down_linear_names = [f"{self.prefix}.down_proj"]

        self.gate_up = MergedColumnParallelLinear(config, file_loader, gate_up_linear_names, **kwargs)
        self.down = RowParallelLinear(config, file_loader, down_linear_names, **kwargs)


class Qwen2Layer(BaseLayer):
    def __init__(
            self,
            config: BaseConfig,
            file_loader: SafetensorFileLoader,
            prefix: str,
            layer_idx: int,
            **kwargs
    ) -> None:
        super().__init__(config, file_loader, prefix, layer_idx, **kwargs)

        self.self_attn = Qwen2Attention(config, file_loader, f"{self.prefix}.self_attn", **kwargs)
        self.mlp = Qwen2Mlp(config, file_loader, f"{self.prefix}.mlp", **kwargs)
        self.input_layernorm = RmsNorm(
            config,
            file_loader,
            f"{self.prefix}.input_layernorm",
            **kwargs
        )
        self.post_attention_layernorm = RmsNorm(
            config,
            file_loader,
            f"{self.prefix}.post_attention_layernorm",
            **kwargs
        )
    
    def forward(self, inputs, cos_emb, sin_emb, k_cache, v_cache, slots, attention_mask=None,
                seq_len=None, block_table=None, token_offset=None, layer_ids=None, is_prefill: bool = True, **kwargs):
        norm_out = self.input_layernorm(inputs)
        attn_out = self.self_attn(norm_out, cos_emb, sin_emb, k_cache, v_cache, slots, attention_mask,
                                  seq_len, block_table, token_offset, layer_ids, is_prefill, **kwargs)
        res_add = attn_out + inputs
        post_norm_out = self.post_attention_layernorm(res_add)
        mlp_out = self.mlp(post_norm_out, **kwargs)
        out = mlp_out + res_add
        return out


class Qwen2Model(BaseModel):
    def __init__(
            self,
            config: BaseConfig,
            file_loader: SafetensorFileLoader,
            mapping: Mapping,
            prefix: str = "model",
            **kwargs
    ) -> None:
        super().__init__(config, file_loader, prefix, **kwargs)

        parallel_embedding = config.vocab_size >= QWEN2_EMBEDDING_PARALLEL_THRESHOLD

        self.embed_tokens = ParallelEmbedding(
            config,
            file_loader,
            f"{self.prefix}.embed_tokens",
            mapping=mapping,
            parallel_embedding=parallel_embedding
        )
        self.layers = nn.ModuleList(
            [
                Qwen2Layer(config, file_loader, self.prefix, layer_idx, **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RmsNorm(config, file_loader, f"{self.prefix}.norm", **kwargs)
    
    def forward(self, input_ids, position_ids, cosine_table, sine_table, k_caches, v_caches, slots_mapping=None,
                attention_mask=None, seq_len=None, block_table=None, token_offset=None, kv_cache_idx=None,
                is_prefill: bool = True, **kwargs):
        hidden_states = self.embed_tokens(input_ids)

        cos_emb = gather(cosine_table, 0, position_ids)
        sin_emb = gather(sine_table, 0, position_ids)
        for i in range(self.config.num_hidden_layers):
            hidden_states = self.layers[i](
                hidden_states, cos_emb, sin_emb, k_caches[i], v_caches[i], slots_mapping,
                attention_mask, seq_len, block_table, token_offset, kv_cache_idx, is_prefill, **kwargs
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states
