# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any

from torch import nn

from atb_llm.utils.data.layer_adapter import (
    RMSNorm,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
    VocabParallelEmbedding
)
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase


# NOTE: This file will replace modeling_qwen2.py.
class Qwen2Attention(nn.Module):
    def __init__(
        self, config, prefix,
        quant_config: QuantizationConfigBase = None,
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.quant_config = quant_config

        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // config.num_attention_heads

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        )

        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, config.rms_norm_eps, quant_config=quant_config, prefix=f"{prefix}.q_norm")
            self.k_norm = RMSNorm(
                self.head_dim, config.rms_norm_eps, quant_config=quant_config, prefix=f"{prefix}.k_norm")


class Qwen2Mlp(nn.Module):
    def __init__(
            self, config, prefix,
            quant_config: QuantizationConfigBase = None,
        ):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.quant_config = quant_config

        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
        )

        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )


class Qwen2Layer(nn.Module):
    def __init__(
            self,
            config,
            prefix: str,
            layer_idx: int,
            quant_config: QuantizationConfigBase = None,
    ) -> None:
        super().__init__()
        
        self.config = config
        self.prefix = f"{prefix}.layers.{layer_idx}"
        self.layer_idx = layer_idx

        self.self_attn_prefix = f"{self.prefix}.self_attn"
        self.self_attn = Qwen2Attention(config, self.self_attn_prefix, quant_config=quant_config)

        self.mlp = Qwen2Mlp(config, f"{self.prefix}.mlp", quant_config=quant_config)

        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            quant_config=quant_config, prefix=f"{self.prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            quant_config=quant_config, prefix=f"{self.prefix}.post_attention_layernorm")


class Qwen2Model(nn.Module):
    def __init__(
            self,
            config: Any,
            prefix: str = "model",
            quant_config: QuantizationConfigBase = None,
    ) -> None:
        super().__init__()
        
        self.config = config
        self.prefix = prefix
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=f"{prefix}.embed_tokens",
            partition_weights=True,
        )

        self.layers = nn.ModuleList(
            [
                Qwen2Layer(config, self.prefix, layer_idx, quant_config=self.quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, quant_config=quant_config, prefix=f"{prefix}.norm")
