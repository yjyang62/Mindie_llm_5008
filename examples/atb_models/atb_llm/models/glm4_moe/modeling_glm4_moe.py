# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
from transformers.activations import ACT2FN

from atb_llm.utils.layers import (
    PositionRotaryEmbedding,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    load_column_multi,
    RMSNorm,
    RMSNormBias,
    RMSNormWrapper
)
from atb_llm.utils.moe_utils import assign

from atb_llm.utils.layers.linear import FastLinear
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.weights import ProcessGroupType
from atb_llm.utils.log.error_code import ErrorCode


TOPK_METHOD = "topk_method"
TOPK_METHOD_NOAUX_TC = "noaux_tc"
EP_LEVEL = "ep_level"


class Glm4moeMLP(nn.Module):
    def __init__(self, prefix, config, weights, intermediate_size=None):
        super().__init__()
        act = config.hidden_act
        approximate = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate=approximate)
        )
        linear_names = [f'{prefix}.up_proj', f'{prefix}.gate_proj']
        pack_name = f'{prefix}.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        self.gate_up_proj = load_column_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            head_size=1,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )


class Glm4moeMoe(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(
            self,
            prefix,
            config,
            weights,
            shared_mlp_cls,
            gate_key="gate",
            shared_expert_key="shared_experts",
            layer_id=None,
            llm_config=None,
            init_expert_table=None,
            mix_shared_routing=False
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.mix_shared_routing = mix_shared_routing

        self.replicate_experts_num_per_expert_per_layer = None
        self.expert_id_map_per_layer = None

        self.ep = weights.mapping.has_moe_ep()
        self.is_static_ep = self.ep and (not hasattr(config, EP_LEVEL) or config.ep_level == 1)
        self.expert_lists = []
        if self.ep:
            self.rank = weights.mapping.moe_ep.rank
            self.world_size = weights.mapping.moe_ep.group_size
            self.expert_lists = assign(config.n_routed_experts, self.world_size)
            self.device_expert = assign(self.config.n_routed_experts,
                                        self.world_size)[weights.mapping.moe_ep.rank]
        else:
            if weights.mapping.has_moe_tp():
                self.rank = weights.mapping.moe_tp.rank
                self.world_size = weights.mapping.moe_tp.group_size
            else:
                self.rank = weights.mapping.mlp_tp.rank
                self.world_size = weights.mapping.mlp_tp.group_size
            self.expert_lists = [[i for i in range(config.n_routed_experts)] for j in range(self.world_size)]
            self.device_expert = [i for i in range(self.config.n_routed_experts)]
            
        temp_list = [j for j in range(config.n_routed_experts)]
        temp_list = temp_list[self.device_expert[0]:] + temp_list[:self.device_expert[0]]

        expert_prefix = f"{prefix}.experts"
        gate_weight = weights.get_tensor(f"{prefix}.gate.weight")
        gate_bias = weights.get_tensor(f"{prefix}.gate.e_score_correction_bias")
        self.gate = FastLinear(gate_weight, gate_bias)
        self.gate.weight.data = self.gate.weight.data.to(torch.float32).contiguous()
        
        if self.is_static_ep:
            if hasattr(config, TOPK_METHOD) and config.topk_method == TOPK_METHOD_NOAUX_TC:
                self.gate.bias.data = self.gate.bias.data[temp_list]
            self.gate.weight.data = self.gate.weight.data[temp_list]

        if hasattr(config, TOPK_METHOD) and config.topk_method == TOPK_METHOD_NOAUX_TC:
            self.gate.weight.data = self.gate.weight.data.to(torch.float32).contiguous()

        self.init_experts(weights, prefix, expert_prefix, shared_expert_key, shared_mlp_cls)
        
    def init_experts(self, weights, prefix, expert_prefix, shared_expert_key, shared_mlp_cls):
        linear_names = [f'{expert_prefix}.0.up_proj', f'{expert_prefix}.0.gate_proj']
        pack_name = f'{expert_prefix}.0.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        config = self.config
        if hasattr(config, "moe_quantize"):
            tmp_quantize = config.quantize
            config.quantize = config.moe_quantize
            weights.quantize = config.moe_quantize
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        self.moe_pack_type = self.pack_type

        if self.ep:
            weights.switch_process_group(ProcessGroupType.MOE_EP)
        pack_prefixes = None
        pack_prefixes = [[f"{expert_prefix}.{i}.gate_proj", f"{expert_prefix}.{i}.up_proj"] \
                        for i in self.expert_lists[self.rank]]

        self.gate_up_proj = TensorParallelColumnLinear.load_moe(
            config,
            prefix_list=pack_prefixes,
            weights=weights,
            bias=False
        )

        down_prefixes = [f"{expert_prefix}.{i}.down_proj" \
                        for i in self.expert_lists[self.rank]]

        self.down_proj = TensorParallelRowLinear.load_moe(
                config,
                prefix_list=down_prefixes,
                process_group=weights.process_group,
                weights=weights,
                bias=False
            )
        if hasattr(config, "moe_quantize"):
            config.quantize = tmp_quantize
            weights.quantize = config.quantize

        self.intermediate_size = ((config.intermediate_size + self.world_size - 1) // self.world_size)
        if not self.mix_shared_routing:
            if config.n_shared_experts is not None:
                intermediate_size = config.moe_intermediate_size * config.n_shared_experts
                shared_expert_prefix = f"{prefix}.shared_experts"
                if self.ep:
                    weights.switch_process_group(ProcessGroupType.MLP)
                    if not (hasattr(config, EP_LEVEL)) or config.ep_level != 1:
                        weights.switch_process_group(ProcessGroupType.MOE_EP)
                
                self.shared_experts = shared_mlp_cls(
                    prefix=shared_expert_prefix,
                    config=config,
                    weights=weights,
                    intermediate_size=intermediate_size
                )


class FlashGlm4moeAttention(nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = config.head_dim

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5
        if (config.num_attention_heads != config.num_key_value_heads and
            self.num_heads % weights.process_group.size() != 0):
            msg = f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} " \
                  f"and `num_shards`: {weights.process_group.size()}"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
        if config.num_key_value_heads < weights.process_group.size():
            repeat_times = weights.process_group.size() // config.num_key_value_heads
        else:
            repeat_times = 1

        self.num_heads = (self.num_heads + weights.process_group.size() - 1) // weights.process_group.size()
        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads * repeat_times
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_heads

        if hasattr(config, 'attn_quantize'):
            cache_quantize = config.quantize
            config.quantize = config.attn_quantize
            weights.quantize = config.attn_quantize

        linear_names = [f'{prefix}.q_proj', f'{prefix}.k_proj', f'{prefix}.v_proj']
        pack_name = f'{prefix}.query_key_value'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)

        if self.pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI,
                              PackType.ALL_W8A16, PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.query_key_value = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size,
                bias=True
            )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
            gqa_size=self.head_size,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(
                prefix=f"{prefix}.q_norm", weights=weights, eps=config.rms_norm_eps
            )
            self.k_norm = RMSNorm(
                prefix=f"{prefix}.k_norm", weights=weights, eps=config.rms_norm_eps
            )

        self.prefix = prefix
        if hasattr(config, 'attn_quantize'):
            config.quantize = cache_quantize
            weights.quantize = cache_quantize
   

class FlashGlm4moeLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"

        weights.switch_process_group(ProcessGroupType.ATTN)
        self.self_attn = FlashGlm4moeAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        
        if config.first_k_dense_replace > layer_id:
            self.mlp = Glm4moeMLP(prefix=f"{prefix}.mlp", config=config, weights=weights,
                                                  intermediate_size=config.intermediate_size)
        else:
            self.mlp = Glm4moeMoe(prefix=f"{prefix}.mlp", config=config, weights=weights,
                              shared_mlp_cls=Glm4moeMLP, layer_id=layer_id)
        
        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16, PackType.ALL_W4A16]:
            self.input_layernorm = RMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
                                          PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
                                          PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
                                          PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.input_layernorm = RMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8_DYNAMIC]:
            self.input_layernorm = RMSNormBias(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        else:
            raise AssertionError(f'self_attn.pack_type: {self.self_attn.pack_type} not supported')
        

        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16]:
            self.post_attention_layernorm = RMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
                                    PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
                                    PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
                                    PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.post_attention_layernorm = RMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8_DYNAMIC]:
            self.post_attention_layernorm = RMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            raise AssertionError(f'mlp.pack_type: {self.mlp.pack_type} not supported')


class FlashGlm4moeModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashGlm4moeLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )
        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
