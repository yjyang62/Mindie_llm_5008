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
from ..base.config import BaseConfig


@dataclass
class Glm4moeConfig(BaseConfig):
    model_type: str = "glm4_moe"
    attention_bias: bool = True
    attention_dropout: float = 0.0
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 4096
    partial_rotary_factor: float = 0.5
    initializer_range: float = 0.02
    intermediate_size: int = 10944
    max_position_embeddings: int = 131072
    moe_intermediate_size: int = 1408
    norm_topk_prob: bool = True
    num_attention_heads: int = 96
    n_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    routed_scaling_factor: float = 1.0
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 1
    num_hidden_layers: int = 46
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-05
    rope_scaling: Optional[float] = None
    rope_theta: float = 1000000.0
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False
    use_cache: bool = True
    use_qk_norm: bool = False
    vocab_size: int = 151552
    rope_ratio: float = 1.0

    def __init__(self, rope_scaling, **kwargs):
        super().__init__(**kwargs)
        self.rope_scaling_dict = rope_scaling