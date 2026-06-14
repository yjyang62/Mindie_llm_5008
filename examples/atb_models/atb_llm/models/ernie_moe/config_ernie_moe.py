# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass
from ..base.config import BaseConfig


@dataclass
class ErniemoeConfig(BaseConfig):
    model_type: str = "ernie_moe"
    hidden_act: str = "silu"
    hidden_size: int = 8192
    intermediate_size: int = 28672
    max_position_embeddings: int = 131072
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    num_hidden_layers: int = 54
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    rms_norm_eps: float = 1e-05
    use_cache: bool = False
    vocab_size: int = 103424
    rope_theta: float = 500000.0
    moe_num_experts: int = 64
    moe_num_shared_experts: int = 0
    moe_layer_start_index: int = 3
    moe_layer_end_index: int = 53
    moe_intermediate_size: int = 3584
    moe_k: int = 8
    moe_layer_interval: int = 1
    tie_word_embeddings: bool = True

    def __init__(self, **kwargs):
        self.attribute_map = {'num_experts_per_tok': 'moe_k'}
        super().__init__(**kwargs)