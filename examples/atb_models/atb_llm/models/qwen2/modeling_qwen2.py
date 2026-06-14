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

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from atb_llm.models.qwen2.modeling_base import (
    QwenRMSNorm,
    QwenRMSNormBias,
    QwenRMSNormWrapper,
    QwenRMSNormAntiOutlierWrapper
)
from atb_llm.nn.modules import Module
from atb_llm.utils import OpBackend
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    PositionRotaryEmbedding,
    TensorEmbedding,
    TensorParallelEmbedding,
    load_column_multi,
    KvCache,
    FA3
)
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.quantize.quant_type import QuantType
from ..base.model_utils import get_tqdm_iterator

QUANTIZE_SC_DESC = ["w8a8sc", "w16a16sc"]


class QwenMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        approximate = "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate=approximate)
        )
        linear_names = [f'{prefix}.up_proj', f'{prefix}.gate_proj']
        pack_name = f'{prefix}.w2_w1'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if (self.pack_type == PackType.ALL_FP and config.quantize == QuantType.W8A8SC) or \
            self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI, PackType.ALL_W16A16SC]:
            self.w2_w1 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.w2_w1",
                weights=weights,
                bias=False,
            )
        elif self.pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
                              PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
                              PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI,
                              PackType.ALL_W4A8, PackType.ALL_W4A8_ANTI]:
            self.w2_w1 = load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1
            )
        else:
            self.w2 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_proj",
                weights=weights,
                bias=False
            )
            self.w1 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.up_proj",
                weights=weights,
                bias=False
            )
        if config.quantize in QUANTIZE_SC_DESC:
            self.c_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",  # down_proj
                weights=weights,
                bias=False,
            )
        else:
            self.c_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.down_proj",  # down_proj
                weights=weights,
                bias=False
            )
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )


class FlashQwenAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
            attn_decode_backend
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if "qwen3" in config.model_type:
            self.head_size = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        else:
            self.head_size = self.hidden_size // self.num_heads
        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        if config.quantization_config.kv_quant_type is not None:
            self.kv_cache_quant = KvCache.load(prefix_k=f"{prefix}.k_proj", prefix_v=f"{prefix}.v_proj",
                                               weights=weights, gqa_size=self.head_size, backend=attn_decode_backend)

        if config.quantization_config.fa_quant_type is not None:
            self.fa3 = FA3.load(prefix_q=f"{prefix}.fa_q", prefix_k=f"{prefix}.fa_k", prefix_v=f"{prefix}.fa_v",
                                weights=weights, head_size=self.head_size)

        self.softmax_scale = self.head_size ** -0.5

        config_quantize_change_flag = False
        if hasattr(config, 'attn_quantize'):
            config_quantize_change_flag = True
            cache_quantize = config.quantize
            config.quantize = config.attn_quantize
            weights.quantize = config.attn_quantize
        elif "moe" in config.model_type and config.quantize == "w8a8_dynamic":
            config_quantize_change_flag = True
            cache_quantize = config.quantize
            config.quantize = "w8a8"
            weights.quantize = "w8a8"

        # can support self.num_heads % weights.process_group.size() != 0
        linear_names = [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        pack_name = f'{prefix}.c_attn'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.input_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if (self.pack_type in [PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
                               PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
                               PackType.ALL_W8A8_DYNAMIC,
                               PackType.ALL_W8A8_DYNAMIC_ANTI, PackType.ALL_W4A8, PackType.ALL_W4A8_ANTI]):
            self.c_attn = load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size,
                bias=config.attention_bias
            )
            self.c_attn.linear.num_linear_before_pack = 3
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI, PackType.ALL_W16A16SC]:
            self.c_attn = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.c_attn",
                weights=weights,
                bias=config.attention_bias,
            )
            self.c_attn.linear.num_linear_before_pack = 3
        else:
            self.q_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.q_proj",
                weights=weights,
                bias=config.attention_bias
            )
            self.k_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.k_proj",
                weights=weights,
                bias=config.attention_bias
            )
            self.v_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.v_proj",
                weights=weights,
                bias=config.attention_bias
            )
        if config.quantize in QUANTIZE_SC_DESC:
            self.c_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",
                weights=weights,
                bias=False,
            )
        else:
            self.c_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.o_proj",
                weights=weights,
                bias=False,
                gqa_size=self.head_size
            )
        if config.use_qk_norm:
            self.q_norm = QwenRMSNorm(
                prefix=f"{prefix}.q_norm", weights=weights, eps=config.rms_norm_eps
            )
            self.k_norm = QwenRMSNorm(
                prefix=f"{prefix}.k_norm", weights=weights, eps=config.rms_norm_eps
            )

        self.prefix = prefix
        if config_quantize_change_flag:
            config.quantize = cache_quantize
            weights.quantize = cache_quantize


class FlashQwenLayer(nn.Module):
    def __init__(self, layer_id, config, weights, prefix, attn_decode_backend):
        super().__init__()
        if config.quantize in QUANTIZE_SC_DESC:
            prefix = f"transformer.h.{layer_id}"
        else:
            prefix = f"{prefix}.layers.{layer_id}"

        if config.quantize in QUANTIZE_SC_DESC:
            self.attn = FlashQwenAttention(
                prefix=f"{prefix}.attn", config=config, weights=weights, attn_decode_backend=attn_decode_backend
            )
            self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        else:
            self.attn = FlashQwenAttention(
                prefix=f"{prefix}.self_attn", config=config, weights=weights, attn_decode_backend=attn_decode_backend
            )
            self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        if self.attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16, PackType.ALL_W4A16]:
            self.ln_1 = QwenRMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
                                     PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
                                     PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI]:
            self.ln_1 = QwenRMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.attn.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            if config.quantize == QuantType.W8A8SC:
                self.ln_1 = QwenRMSNormAntiOutlierWrapper(
                    prefix=f"{prefix}.ln_1", weights=weights, eps=config.rms_norm_eps
                )
            else:
                self.ln_1 = QwenRMSNormAntiOutlierWrapper(
                    prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
                )
        elif self.attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC, PackType.MIX_W8A8SC,
                                     PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W4A8, PackType.ALL_W16A16SC]:
            if config.quantize == QuantType.W8A8SC:
                self.ln_1 = QwenRMSNormBias(
                    prefix=f"{prefix}.ln_1", weights=weights, eps=config.rms_norm_eps
                )
            elif config.quantize == QuantType.W16A16SC:
                self.ln_1 = QwenRMSNorm(
                    prefix=f"{prefix}.ln_1", weights=weights, eps=config.rms_norm_eps
                )
            else:
                self.ln_1 = QwenRMSNormBias(
                    prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
                )
        else:
            raise AssertionError(f'self_attn.pack_type: {self.self_attn.pack_type} not supported')
        if self.mlp.pack_type == PackType.ALL_FP and config.quantize == QuantType.W8A8SC:
            self.ln_2 = QwenRMSNormBias(
                prefix=f"{prefix}.ln_2",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16]:
            self.ln_2 = QwenRMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
                                    PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
                                    PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI]:
            self.ln_2 = QwenRMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8SC_ANTI, PackType.MIX_W8A8SC_ANTI]:
            if config.quantize == QuantType.W8A8SC:
                self.ln_2 = QwenRMSNormAntiOutlierWrapper(
                    prefix=f"{prefix}.ln_2", weights=weights, eps=config.rms_norm_eps
                )
            else:
                self.ln_2 = QwenRMSNormAntiOutlierWrapper(
                    prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
                )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8SC,
                                    PackType.MIX_W8A8SC, PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W4A8,
                                    PackType.ALL_W16A16SC]:
            if config.quantize == QuantType.W8A8SC:
                self.ln_2 = QwenRMSNormBias(
                    prefix=f"{prefix}.ln_2",
                    weights=weights,
                    eps=config.rms_norm_eps,
                )
            elif config.quantize == QuantType.W16A16SC:
                self.ln_2 = QwenRMSNorm(
                    prefix=f"{prefix}.ln_2",
                    weights=weights,
                    eps=config.rms_norm_eps,
                )

            else:
                self.ln_2 = QwenRMSNormBias(
                    prefix=f"{prefix}.post_attention_layernorm",
                    weights=weights,
                    eps=config.rms_norm_eps,
                )
        else:
            raise AssertionError(f'mlp.pack_type: {self.mlp.pack_type} not supported')


class FlashQwenModel(Module):
    def __init__(self, config, weights, **kwargs):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        model_prefix = kwargs.get("model_prefix", "model")
        if config.quantize in QUANTIZE_SC_DESC:
            self.wte = TensorEmbedding(
                prefix="transformer.wte", weights=weights
            )
        else:
            self.wte = TensorParallelEmbedding(
                prefix=f"{model_prefix}.embed_tokens", weights=weights
            )
        attn_decode_backend = kwargs.get("attn_decode_backend", OpBackend.ATB)
        layerwise_disaggregated = kwargs.get("layerwise_disaggregated")
        load_list = kwargs.get("load_list")


        if layerwise_disaggregated and load_list is not None and len(load_list) > 0:
            layers_to_load = [i for i in range(config.num_hidden_layers) if i in load_list]
            iterator = get_tqdm_iterator(layers_to_load, 
                                         weights.mapping.rank % weights.mapping.local_world_size)
            layer_list = []
            for layer_id in iterator:
                layer_list.append(
                    FlashQwenLayer(
                        layer_id,
                        config,
                        weights,
                        model_prefix,
                        attn_decode_backend
                ))
            self.h = nn.ModuleList(layer_list)
        else:
            iterator = get_tqdm_iterator(range(config.num_hidden_layers), 
                                         weights.mapping.rank % weights.mapping.local_world_size)
            layer_list = []
            for layer_id in iterator:
                layer_list.append(FlashQwenLayer(
                        layer_id,
                        config,
                        weights,
                        model_prefix,
                        attn_decode_backend
                    ))
            self.h = nn.ModuleList(layer_list)

        if config.quantize in QUANTIZE_SC_DESC:
            self.ln_f = QwenRMSNorm(
                prefix="transformer.ln_f", weights=weights, eps=config.rms_norm_eps
            )
        else:
            self.ln_f = QwenRMSNorm(
                prefix=f"{model_prefix}.norm", weights=weights, eps=config.rms_norm_eps
            )
        self.gradient_checkpointing = False

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads

        self.add_aliases({
            'embed_tokens': 'wte',
            'layers': 'h',
            'norm': 'ln_f',
        })