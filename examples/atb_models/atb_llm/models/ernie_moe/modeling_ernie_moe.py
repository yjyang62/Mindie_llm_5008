# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#          http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement part of this file based on transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
from torch import nn
from atb_llm.utils.layers.linear import FastLinear
from atb_llm.utils.layers import (
    TensorParallelEmbedding,
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    RMSNorm
)
from atb_llm.models.base.modeling import FlashAttention, MLP, FlashLayer
from atb_llm.utils.moe_utils import assign
from atb_llm.utils.weights import ProcessGroupType
from atb_llm.utils.quantize.pack_type import calc_linear_pack_type, ALL_PACK_LIST


class FlashErniemoeAttention(FlashAttention):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.qkv_names = [f'{self.prefix}.q_proj', f'{self.prefix}.k_proj', f'{self.prefix}.v_proj']
        config_quantize_change_flag = False
        if hasattr(config, 'attn_quantize'):
            config_quantize_change_flag = True
            cache_quantize = config.quantize
            config.quantize = config.attn_quantize
            weights.quantize = config.attn_quantize
        elif config.quantize == "w8a8_dynamic":
            config_quantize_change_flag = True
            cache_quantize = config.quantize
            config.quantize = "w8a8"
            weights.quantize = "w8a8"
        self.load_weights(**kwargs)
        if config_quantize_change_flag:
            config.quantize = cache_quantize
            weights.quantize = cache_quantize


class ErniemoeMLP(MLP):
    def __init__(self, prefix, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)


class ErniemoeMoE(nn.Module):
    def __init__(self, prefix, config, weights, shared_expert_cls=ErniemoeMLP, model_config=None):
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.model_config = model_config
        self.weights = weights
        self.shared_expert_cls = shared_expert_cls
        layer_prefix = '.'.join(self.prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.post_attention_layernorm'

        gate_weight = weights.get_tensor(f"{prefix}.gate.weight")
        moe_statics = weights.get_tensor(f"{prefix}.moe_statics.e_score_correction_bias").squeeze(0)
        moe_statics = moe_statics - moe_statics.min()
        self.gate = FastLinear(gate_weight, moe_statics)

        if weights.mapping.has_moe_ep():
            self.expert_lists = assign(config.moe_num_experts, weights.mapping.moe_ep.group_size)
            self.device_experts = self.expert_lists[weights.mapping.moe_ep.rank]
            if self.model_config.ep_level == 1:
                shuffled_experts = [j for j in range(config.moe_num_experts)]
                shuffled_experts = shuffled_experts[self.device_experts[0]:] + shuffled_experts[:self.device_experts[0]]
                self.gate.weight.data = self.gate.weight.data[shuffled_experts]
                self.gate.bias.data = self.gate.bias.data[shuffled_experts]
        else:
            self.expert_lists = [[i for i in range(config.moe_num_experts)]
                                 for _ in range(weights.mapping.moe_tp.group_size)]

        self.gate.weight.data = self.gate.weight.data.to(torch.float32).contiguous()
        expert_prefix = f"{prefix}.experts"
        self.init_experts(weights, prefix, expert_prefix)

    def init_experts(self, weights, prefix, expert_prefix):
        linear_names = [f"{expert_prefix}.0.gate_proj", f"{expert_prefix}.0.up_proj"]
        pack_name = f"{expert_prefix}.0.gate_up_proj"
        self.pack_type = calc_linear_pack_type(weights, linear_names, self.norm_name, pack_name)
        if self.weights.mapping.has_moe_ep():
            self.weights.switch_process_group(ProcessGroupType.MOE_EP)
        if self.pack_type in ALL_PACK_LIST:
            pack_prefixes = [[f"{expert_prefix}.{i}.gate_proj", f"{expert_prefix}.{i}.up_proj"] \
                            for i in self.expert_lists[weights.mapping.moe_ep.rank]]
            self.gate_up_proj = TensorParallelColumnLinear.load_moe(
                    self.config,
                    prefix_list=pack_prefixes,
                    weights=weights,
                    bias=False
                )
        down_prefixes = [f"{expert_prefix}.{i}.down_proj" \
                        for i in self.expert_lists[weights.mapping.moe_ep.rank]]
        self.down_proj = TensorParallelRowLinear.load_moe(
                self.config,
                prefix_list=down_prefixes,
                process_group=weights.process_group,
                weights=weights,
                bias=False
            )
        if hasattr(self.config, "moe_num_shared_experts") and self.config.moe_num_shared_experts > 0:
            shared_expert_prefixes = f"{self.prefix}.shared_experts"
            if self.weights.mapping.has_moe_ep():
                self.weights.switch_process_group(ProcessGroupType.MLP)
            self.shared_experts = self.shared_expert_cls(prefix=shared_expert_prefixes,
                                                         config=self.config, weights=self.weights)


class FlashErniemoeLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, model_prefix="model", model_config=None, **kwargs):
        super().__init__(layer_id, config, weights, model_prefix, **kwargs)
        self.layer_id = layer_id
        self.model_config = model_config
        self.load_weights(**kwargs)

    def load_weights(self, **kwargs):
        self.weights.switch_process_group(ProcessGroupType.ATTN)
        self.self_attn = FlashErniemoeAttention(
            prefix=f"{self.prefix}.self_attn", config=self.config, weights=self.weights, **kwargs
        )
        if self.layer_id < self.config.moe_layer_start_index:
            self.weights.switch_process_group(ProcessGroupType.ATTN)
            self.mlp = ErniemoeMLP(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights, **kwargs)
        else:
            self.weights.switch_process_group(ProcessGroupType.MLP)
            self.mlp = ErniemoeMoE(prefix=f"{self.prefix}.mlp", config=self.config, weights=self.weights,
                                   shared_expert_cls=ErniemoeMLP, model_config=self.model_config, **kwargs)
        super().load_weights(**kwargs)


class FlashErniemoeModel(nn.Module):
    def __init__(self, config, weights, model_prefix="model", model_config=None, **kwargs):
        super().__init__()
        self.embed_tokens = TensorParallelEmbedding(prefix=f"{model_prefix}.embed_tokens", weights=weights)
        self.layers = nn.ModuleList(
            [
                FlashErniemoeLayer(layer_id, config, weights, model_prefix, model_config, **kwargs)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(prefix=f"{model_prefix}.norm", weights=weights, eps=config.rms_norm_eps)
