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

from torch import nn
from atb_llm.models.base.modeling import FlashAttention, FlashLayer, MLP
from atb_llm.utils.layers import TensorParallelRowLinear, KvCache, RMSNorm
from atb_llm.utils.quantize.pack_type import calc_linear_pack_type

_CHATGLM_TYPE = "chatglm"
_GLM_TYPE = "glm"


class FlashChatglmAttention(FlashAttention):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__(prefix, config, weights)
        self.qkv_names = [f'{prefix}.query_key_value'] if config.model_type == _CHATGLM_TYPE \
            else [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        self.qkv_bias = True

        self.multi_query_group_num = config.multi_query_group_num if config.model_type == _CHATGLM_TYPE \
            else config.num_key_value_heads
        self.kv_head_nums_per_rank = max(self.multi_query_group_num // weights.process_group.size(), 1)
        if config.quantization_config.kv_quant_type is not None:
            prefix_k = f"{prefix}.query_key_value.k_proj" if config.model_type == _CHATGLM_TYPE \
                else f"{prefix}.k_proj"
            prefix_v = f"{prefix}.query_key_value.v_proj" if config.model_type == _CHATGLM_TYPE \
                else f"{prefix}.v_proj"
            self.kv_cache_quant = KvCache.load(prefix_k=prefix_k,
                                               prefix_v=prefix_v,
                                               weights=weights,
                                               gqa_size=self.kv_head_nums_per_rank * config.kv_channels)

        self.pack_type = calc_linear_pack_type(weights, self.qkv_names, self.norm_name)

        dense_prefix = f"{prefix}.dense" if config.model_type == _CHATGLM_TYPE \
            else f"{prefix}.o_proj"
        self.dense = TensorParallelRowLinear.load(
            config,
            prefix=dense_prefix,
            weights=weights,
            bias=False,
        )
        
        self.load_qkv_weights()


class ChatglmMLP(MLP):
    def __init__(self, prefix, config, weights):
        super().__init__(prefix, config, weights)

        self.gate_up_names = [f'{prefix}.dense_h_to_4h'] if config.model_type == _CHATGLM_TYPE \
            else [f'{prefix}.gate_up_proj']
        self.pack_name = f'{prefix}.dense_h_to_4h' if config.model_type == _CHATGLM_TYPE \
            else [f'{prefix}.gate_up_proj']
        self.down_name = f"{prefix}.dense_4h_to_h" if config.model_type == _CHATGLM_TYPE \
            else [f'{prefix}.down_proj']
        self.load_weights()


class FlashChatglmLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, prefix="transformer.encoder"):
        super().__init__(layer_id, config, weights, prefix)

        self.attn_name = "self_attention" if config.model_type == _CHATGLM_TYPE \
            else "self_attn"
        
        if config.model_type == _CHATGLM_TYPE:
            self.self_attention = FlashChatglmAttention(
                prefix=f"{self.prefix}.{self.attn_name}", config=config, weights=weights
            )
        elif config.model_type == _GLM_TYPE:
            self.self_attn = FlashChatglmAttention(
                prefix=f"{self.prefix}.{self.attn_name}", config=config, weights=weights
            )
        self.mlp = ChatglmMLP(prefix=f"{self.prefix}.{self.mlp_name}", config=config, weights=weights)
        self.load_weights()


class GLMTransformer(nn.Module):
    def __init__(self, config, weights):
        super(GLMTransformer, self).__init__()

        if config.quantize == "w8a8sc":
            prefix = "encoder"
        elif config.model_type == _CHATGLM_TYPE:
            prefix = "transformer.encoder"
        elif config.model_type == _GLM_TYPE:
            prefix = "model"

        self.layers = nn.ModuleList(
            [
                FlashChatglmLayer(layer_id, config, weights, prefix)
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        rms_norm_eps = config.layernorm_epsilon if config.model_type == _CHATGLM_TYPE \
            else config.rms_norm_eps
        rms_norm_prefix = f"{prefix}.final_layernorm" if config.model_type == _CHATGLM_TYPE \
            else f"{prefix}.norm"
        self.final_layernorm = RMSNorm(
            prefix=rms_norm_prefix, weights=weights, eps=rms_norm_eps
        )
