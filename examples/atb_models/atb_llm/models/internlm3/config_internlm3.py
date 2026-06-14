# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from typing import Optional

from atb_llm.models.base.config import BaseConfig


@dataclass
class Internlm3Config(BaseConfig):
    model_type: str = "internlm3"
    vocab_size: int = 128512
    head_dim: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 10240
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    skip_word_embedding: bool = False
    bias: bool = False
    qkv_bias: bool = False
    attn_implementation: str = "eager"
    torch_dtype: str = "float16"
    pe_type: str = "ROPE"
    rope_scaling: Optional[float] = None
    rope_theta: int = 50000000
    alibi_bias_max: float = 8.0
    rope_given_inv_feq_str: Optional[str] = None
    rope_keep_local_base_windows: Optional[str] = None
    rope_mscale: Optional[int] = None
    rope_vanilla_theta: Optional[float] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = False
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
