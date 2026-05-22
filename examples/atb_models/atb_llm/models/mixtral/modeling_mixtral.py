# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from typing import Optional, List, Tuple

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from atb_llm.models.base.modeling import FlashAttention
from atb_llm.utils.layers.linear import FastLinear
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    load_column_multi,
    RMSNorm
    )
from atb_llm.utils.moe_utils import assign
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.weights import ProcessGroupType

TOPK_METHOD = "topk_method"
TOPK_METHOD_NOAUX_TC = "noaux_tc"
EP_LEVEL = "ep_level"


class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.gate_up_proj = load_column_multi(
            config,
            prefixes=[f"{prefix}.w1", f"{prefix}.w3"],
            weights=weights,
            head_size=1,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.w2",
            weights=weights,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.config = config
        self.ep = weights.mapping.has_moe_ep()
        if self.ep:
            self.rank = weights.mapping.moe_ep.rank
            self.world_size = weights.mapping.moe_ep.group_size
        else:
            if weights.mapping.has_moe_tp():
                self.rank = weights.mapping.moe_tp.rank
                self.world_size = weights.mapping.moe_tp.group_size
            else:
                self.rank = weights.mapping.mlp_tp.rank
                self.world_size = weights.mapping.mlp_tp.group_size

        self.expert_lists = []
        if self.ep:
            self.expert_lists = assign(config.n_routed_experts, self.world_size)
        else:
            self.expert_lists = [[i for i in range(config.n_routed_experts)] for j in range(self.world_size)]

        if self.ep:
            self.device_expert = assign(config.n_routed_experts, self.world_size)[weights.mapping.moe_ep.rank]
        else:
            self.device_expert = [i for i in range(config.n_routed_experts)]
            
        temp_list = [j for j in range(config.n_routed_experts)]
        temp_list = temp_list[self.device_expert[0]:] + temp_list[:self.device_expert[0]]

        expert_prefix = f"{prefix}.experts"
        if hasattr(config, TOPK_METHOD) and config.topk_method == TOPK_METHOD_NOAUX_TC:
            self.gate = FastLinear.load(
                prefix=f"{prefix}.gate", weights=weights, bias=True, bias_name="e_score_correction_bias")
        else:
            self.gate = FastLinear.load(prefix=f"{prefix}.gate", weights=weights, bias=False)

        if self.ep:
            if (not hasattr(config, EP_LEVEL) or config.ep_level == 1):
                if hasattr(config, TOPK_METHOD) and config.topk_method == TOPK_METHOD_NOAUX_TC:
                    self.gate.bias.data = self.gate.bias.data[temp_list]
                self.gate.weight.data = self.gate.weight.data[temp_list]

        if hasattr(config, TOPK_METHOD) and config.topk_method == TOPK_METHOD_NOAUX_TC:
            self.gate.weight.data = self.gate.weight.data.to(torch.float32).contiguous()

        self.init_experts(weights, prefix, expert_prefix)
        
    def init_experts(self, weights, prefix, expert_prefix):
        linear_names = [f'{expert_prefix}.0.w3', f'{expert_prefix}.0.w1']
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
        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W4A8, PackType.MIX_W4A8,
            PackType.ALL_W4A8_ANTI, PackType.MIX_W4A8_ANTI,
        ]:
            pack_prefixes = [[f"{expert_prefix}.{i}.w1", f"{expert_prefix}.{i}.w3"] \
                            for i in self.expert_lists[self.rank]]

            self.gate_up_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefix_list=pack_prefixes,
                    weights=weights,
                    bias=False
                )
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            pack_prefixes = [[f"{expert_prefix}.{i}.gate_up_proj"] for i in self.expert_lists[self.rank]]
            self.gate_up_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefix_list=pack_prefixes,
                    weights=weights,
                    bias=False
                )
        else:
            gate_prefixes = [[f"{expert_prefix}.{i}.w1"] for i in self.expert_lists[self.rank]]
            up_prefixes = [[f"{expert_prefix}.{i}.w3"] for i in self.expert_lists[self.rank]]
            self.gate_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefix_list=gate_prefixes,
                    weights=weights,
                    bias=False
                )
            self.up_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefix_list=up_prefixes,
                    weights=weights,
                    bias=False
                )
        down_prefixes = [f"{expert_prefix}.{i}.w2" \
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

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size, sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        return final_hidden_states, router_logits


class FlashMixtralAttention(FlashAttention):
    def __init__(self, prefix: str, config, weights, **kwargs):
        super().__init__(prefix, config, weights, **kwargs)
        self.load_weights(**kwargs)

        super().load_qkv_weights(**kwargs)


class FlashMixtralLayer(nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                     hidden_states: torch.tensor,
                     residual: torch.tensor,
                     cos: torch.tensor,
                     sin: torch.tensor,
                     cu_seqlen_prefill: torch.tensor,
                     kv_cache: Tuple[torch.tensor, torch.tensor],
                     block_tables: List[torch.tensor],
                     slots: torch.tensor,
                     input_lengths: torch.tensor,
                     max_s: torch.tensor):
            self.hidden_states = hidden_states
            self.residual = residual
            self.cos = cos
            self.sin = sin
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s

    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        weights.switch_process_group(ProcessGroupType.ATTN)
        self.self_attn = FlashMixtralAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        weights.switch_process_group(ProcessGroupType.MLP)
        self.block_sparse_moe = MixtralSparseMoeBlock(
            prefix=f"{prefix}.block_sparse_moe", config=config, weights=weights
        )
        self.input_layernorm = RMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
            self,
            input_args: ForwardInputArgs
    ):
        hidden_states = input_args.hidden_states
        residual = input_args.residual
        cos = input_args.cos
        sin = input_args.sin
        cu_seqlen_prefill = input_args.cu_seqlen_prefill
        kv_cache = input_args.kv_cache
        block_tables = input_args.block_tables
        slots = input_args.slots
        input_lengths = input_args.input_lengths
        max_s = input_args.max_s
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashMixtralModel(torch.nn.Module):
    class ForwardInputArgs:
        def __init__(self,
                    input_ids: torch.Tensor,
                    position_ids: torch.Tensor,
                    cu_seqlen_prefill: Optional[torch.Tensor],
                    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                    block_tables: torch.Tensor,
                    slots: torch.Tensor,
                    input_lengths: torch.Tensor,
                    max_s: int,
                    lm_head_indices: Optional[torch.Tensor] = None):
            self.input_ids = input_ids
            self.position_ids = position_ids
            self.cu_seqlen_prefill = cu_seqlen_prefill
            self.kv_cache = kv_cache
            self.block_tables = block_tables
            self.slots = slots
            self.input_lengths = input_lengths
            self.max_s = max_s
            self.lm_head_indices = lm_head_indices

    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashMixtralLayer(
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

