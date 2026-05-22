# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
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

from typing import Any
from dataclasses import dataclass
import copy

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from atb_llm.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorEmbedding,
    TensorParallelEmbedding,
    TensorReplicatedLinear,
    FA3
)
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.models.deepseek.modeling_deepseek import \
    DeepseekMLP, DeepseekMoE
from atb_llm.utils.layers import PositionRotaryEmbedding
from atb_llm.utils.layers.embedding.position_yarn_embedding import PositionYarnEmbedding, _ROPE_SCALING_KEYS
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.weights import ProcessGroupType
from ..base.model_utils import get_tqdm_iterator


@dataclass
class EpSplitParam:
    gatherd_idxs: Any = None
    input_split_sizes: Any = None
    output_splits: Any = None


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps


class DeepseekV2RMSNormBias(DeepseekV2RMSNorm):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__(prefix, weights, eps)
        
        try:
            bias = weights.get_tensor(f"{prefix}.bias")
        except AssertionError:
            bias = torch.zeros(self.weight.shape, dtype=weights.dtype)
        self.bias = nn.Parameter(bias)


class DeepseekV2RMSNormWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = DeepseekV2RMSNorm(prefix, weights, eps)
        self.anti = DeepseekV2RMSNormBias(f'{prefix}.module', weights, eps)


class DeepseekV2RMSNormAntiOutlierWrapper(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()

        self.ori = DeepseekV2RMSNorm(f'{prefix}.ori', weights, eps)
        self.anti = DeepseekV2RMSNormBias(f'{prefix}.anti', weights, eps)


class DeepseekV2MLP(DeepseekMLP):
    def __init__(self, prefix, config, weights, intermediate_size=None, is_nzcasted=False):
        super().__init__(prefix, config, weights, intermediate_size=None, is_nzcasted=is_nzcasted)
        self.act_fn = ACT2FN[config.hidden_act]


class DeepseekV2MoE(DeepseekMoE):
    def __init__(self, prefix, config, weights, shared_mlp_cls, layer_id, llm_config=None,
                    init_expert_table=None, mix_shared_routing=False):
        super().__init__(prefix, config, weights, shared_mlp_cls, layer_id=layer_id, llm_config=llm_config,
                         init_expert_table=init_expert_table, mix_shared_routing=mix_shared_routing)
        if hasattr(self, "shared_experts"):
            self.pack_type = self.shared_experts.pack_type


class FlashDeepseekV2Attention(nn.Module):
    def __init__(self,
                 prefix: str,
                 config,
                 weights,
                 llm_config=None):
        super().__init__()
        self.config = config
        self.config = copy.deepcopy(config)
        if hasattr(self.config, 'mla_quantize'):
            self.config.quantize = self.config.mla_quantize
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        if llm_config is not None:
            parallel_config = llm_config.llm.parallel_options
            o_proj_local_tp = parallel_config.o_proj_local_tp \
                if parallel_config is not None and isinstance(parallel_config.o_proj_local_tp, int) else -1
            self.enable_o_proj_local_tp = (o_proj_local_tp > 1)
        else:
            self.enable_o_proj_local_tp = False
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.is_causal = True
        linear_names = []
        self.fa3 = None
        self.kvcache_quant = False
        if self.config.quantization_config.fa_quant_type is not None:
            self.fa3, self.kvcache_quant = FA3.load_mla(
                prefix_q=f"{prefix}.fa_q", prefix_k=f"{prefix}.fa_k", prefix_v=f"{prefix}.fa_v",
                weights=weights, head_size=self.kv_lora_rank, module_name=f"{prefix}.fa3")
        if self.q_lora_rank is None:
            self.q_proj = TensorParallelColumnLinear.load_multi(
                self.config, prefixes=[f"{prefix}.q_proj"], weights=weights,
                bias=False, dim=0, proj_name="projq", module_name=f"{prefix}.q_proj"
            )
            linear_names.append(f'{prefix}.q_proj')
        else:
            self.q_a_proj = TensorReplicatedLinear.load(
                self.config,
                prefix=f"{prefix}.q_a_proj",
                weights=weights,
                bias=self.config.attention_bias,
            )
            linear_names.append(f'{prefix}.q_a_proj')
            self.q_b_proj = TensorParallelColumnLinear.load_multi(
                self.config, prefixes=[f"{prefix}.q_b_proj"], weights=weights,
                bias=False, dim=0, proj_name="projq", module_name=f"{prefix}.q_b_proj"
            )
            linear_names.append(f'{prefix}.q_b_proj')

        self.kv_a_proj_with_mqa = TensorReplicatedLinear.load(
            self.config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
            weights=weights,
            bias=self.config.attention_bias,
        )
        linear_names.append(f'{prefix}.kv_a_proj_with_mqa')
        self.kv_a_layernorm = DeepseekV2RMSNorm(prefix=f"{prefix}.kv_a_layernorm", weights=weights)
        self.k_b_proj = TensorParallelColumnLinear.load_multi(
            self.config, prefixes=[f"{prefix}.kv_b_proj"], weights=weights,
            bias=False, dim=0, proj_name="projk", module_name=f"{prefix}.k_b_proj"
        )
        if weights.sharded:
            linear_names.append(f'{prefix}.k_b_proj')
        else:
            linear_names.append(f'{prefix}.kv_b_proj')
        self.v_b_proj = TensorParallelColumnLinear.load_multi(
            self.config, prefixes=[f"{prefix}.kv_b_proj"], weights=weights,
            bias=False, dim=0, proj_name="projv", module_name=f"{prefix}.v_b_proj"
        )
        if weights.sharded:
            linear_names.append(f'{prefix}.v_b_proj')
        else:
            linear_names.append(f'{prefix}.kv_b_proj')

        if self.enable_o_proj_local_tp:
            weights.switch_process_group(ProcessGroupType.ATTN_O)
        self.o_proj = TensorParallelRowLinear.load(
            self.config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=self.config.attention_bias,
        )
        linear_names.append(f'{prefix}.o_proj')
        if self.enable_o_proj_local_tp:
            weights.switch_process_group(ProcessGroupType.ATTN)

        self._init_rope(weights.device)

        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        weights.quantize = self.config.quantize
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name)
        weights.quantize = config.quantize

        if self.q_lora_rank is not None:
            if self.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                            PackType.MIX_W8A8SC]:
                self.q_a_layernorm = DeepseekV2RMSNormBias(prefix=f"{prefix}.q_a_layernorm", weights=weights)
            else:
                self.q_a_layernorm = DeepseekV2RMSNorm(prefix=f"{prefix}.q_a_layernorm", weights=weights)

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling_dict is not None:
            mscale_all_dim = self.config.rope_scaling_dict.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling_dict["factor"]
            if mscale_all_dim:
                mscale = PositionYarnEmbedding.yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self, device):
        if self.config.rope_scaling_dict is None:
            self.rotary_emb = PositionRotaryEmbedding.static(dim=self.qk_rope_head_dim,
                                                             base=self.rope_theta,
                                                             device="cpu").to(device)
        else:
            scaling_type = self.config.rope_scaling_dict["type"]
            scaling_factor = self.config.rope_scaling_dict["factor"]
            if scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling_dict[key]
                    for key in _ROPE_SCALING_KEYS
                    if key in self.config.rope_scaling_dict
                }
                yarn_kwargs = PositionYarnEmbedding.StaticInputArgs(
                                                            max_position_embeddings=self.max_position_embeddings,
                                                            scaling_factor=scaling_factor,
                                                            **kwargs,)
                self.rotary_emb = PositionYarnEmbedding.static_yarn(dim=self.qk_rope_head_dim,
                                                                       base=self.rope_theta,
                                                                       device="cpu",
                                                                       yarn_kwargs=yarn_kwargs).to(device)
            else:
                msg = f"Unknown RoPE scaling type {scaling_type}"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)


class FlashDeepseekV2DecoderLayer(nn.Module):
    def __init__(self, layer_idx, config, weights, llm_config=None, init_expert_table=None, mix_shared_routing=False):
        super().__init__()
        prefix = f"model.layers.{layer_idx}"
        self.hidden_size = config.hidden_size
        self.mix_shared_routing = mix_shared_routing
        weights.switch_process_group(ProcessGroupType.ATTN)
        self.self_attn = FlashDeepseekV2Attention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights, llm_config=llm_config
        )

        if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            ):
            weights.switch_process_group(ProcessGroupType.MLP)
            self.mlp = (
                DeepseekV2MoE(prefix=f"{prefix}.mlp", config=config, weights=weights,
                              shared_mlp_cls=DeepseekV2MLP, layer_id=layer_idx, llm_config=llm_config,
                              init_expert_table=init_expert_table, mix_shared_routing=self.mix_shared_routing)
            )
        else:
            if weights.mapping.enable_dense_tp:
                weights.switch_process_group(ProcessGroupType.DENSE_TP)
            else:
                weights.switch_process_group(ProcessGroupType.ATTN)
            self.mlp = DeepseekV2MLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16, PackType.MIX_W8A16,
                                        PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
                                        PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.input_layernorm = DeepseekV2RMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            self.input_layernorm = DeepseekV2RMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.input_layernorm = DeepseekV2RMSNormAntiOutlierWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                          PackType.MIX_W8A8SC, PackType.MIX_W4A16, PackType.ALL_W4A16]:
            self.input_layernorm = DeepseekV2RMSNormBias(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        else:
            msg = f'self_attn.pack_type: {self.self_attn.pack_type} not supported'
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(msg)

        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W4A16, 
                                  PackType.ALL_W8A16, PackType.MIX_W8A16,
                                  PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
                                  PackType.ALL_W4A8, PackType.ALL_W4A8_ANTI,
                                  PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.post_attention_layernorm = DeepseekV2RMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [
            PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
            PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
            PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
        ]:
            self.post_attention_layernorm = DeepseekV2RMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            self.post_attention_layernorm = DeepseekV2RMSNormAntiOutlierWrapper(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                    PackType.MIX_W8A8SC]:
            self.post_attention_layernorm = DeepseekV2RMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            msg = f'mlp.pack_type: {self.mlp.pack_type} not supported'
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(msg)


class FlashDeepseekV2Model(torch.nn.Module):
    def __init__(self, config, weights, llm_config=None, init_expert_table=None,
                 mix_shared_routing=False, layerwise_disaggregated=False, load_list=None):
        super().__init__()
        self.parallel_embedding = config.parallel_embedding if config.parallel_embedding else True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.v_head_dim = config.v_head_dim
        self.num_heads = config.num_attention_heads
        self.num_speculative_tokens = config.num_speculative_tokens
        self.mix_shared_routing = mix_shared_routing

        self.embed_tokens = (TensorParallelEmbedding if self.parallel_embedding and 
                            not weights.sharded else TensorEmbedding)(
            prefix="model.embed_tokens", weights=weights
        )

        self.layerwise_disaggregated = layerwise_disaggregated
        if self.layerwise_disaggregated and load_list is not None and len(load_list) > 0:
            layer_list = []
            layers_to_load = [i for i in range(config.num_hidden_layers) if i in load_list]
            iterator = get_tqdm_iterator(layers_to_load, 
                                         weights.mapping.rank % weights.mapping.local_world_size)
            for layer_id in iterator:
                layer_list.append(FlashDeepseekV2DecoderLayer(
                        layer_id,
                        config,
                        weights,
                        llm_config=llm_config,
                        init_expert_table=init_expert_table,
                        mix_shared_routing=self.mix_shared_routing
                    ))
            self.layers = nn.ModuleList(layer_list)
        else:
            layer_list = []
            iterator = get_tqdm_iterator(range(config.num_hidden_layers), 
                                         weights.mapping.rank % weights.mapping.local_world_size)
            for layer_id in iterator:
                layer_list.append(FlashDeepseekV2DecoderLayer(
                        layer_id,
                        config,
                        weights,
                        llm_config=llm_config,
                        init_expert_table=init_expert_table,
                        mix_shared_routing=self.mix_shared_routing
                    ))
            self.layers = nn.ModuleList(layer_list)
             

        self.kvcache_quant_layers = [layer.self_attn.kvcache_quant for layer in self.layers]
        self.norm = DeepseekV2RMSNorm(prefix="model.norm", weights=weights, eps=config.rms_norm_eps)

        if self.num_speculative_tokens:

            mtp_layer_id = config.num_hidden_layers
            self.mtp_enorm = DeepseekV2RMSNorm(prefix=f"model.layers.{mtp_layer_id}.enorm", 
                                               weights=weights, eps=config.rms_norm_eps)
            self.mtp_hnorm = DeepseekV2RMSNorm(prefix=f"model.layers.{mtp_layer_id}.hnorm", 
                                               weights=weights, eps=config.rms_norm_eps)

            self.eh_proj = nn.Module()
            weight = weights.get_tensor(f"model.layers.{mtp_layer_id}.eh_proj.weight")
            self.eh_proj.weight = nn.Parameter(weight)

            if hasattr(config, "mtp_quantize"):
                config_mtp_quant = copy.deepcopy(config)
                config_mtp_quant.moe_quantize = config.quantize
                self.mtp_layer = FlashDeepseekV2DecoderLayer(
                                mtp_layer_id,
                                config_mtp_quant,
                                weights,
                                llm_config,
                                init_expert_table,
                                self.mix_shared_routing
                            )
            else:
                self.mtp_layer = FlashDeepseekV2DecoderLayer(
                                    mtp_layer_id,
                                    config,
                                    weights,
                                    llm_config,
                                    init_expert_table,
                                    self.mix_shared_routing
                                )
            self.mtp_replicate_experts_num_per_expert_model = \
                self.mtp_layer.mlp.replicate_experts_num_per_expert_per_layer
            self.mtp_expert_id_map_model = self.mtp_layer.mlp.expert_id_map_per_layer

            self.shared_head_norm = DeepseekV2RMSNorm(prefix=f"model.layers.{mtp_layer_id}.shared_head.norm", 
                                                      weights=weights, eps=config.rms_norm_eps)
