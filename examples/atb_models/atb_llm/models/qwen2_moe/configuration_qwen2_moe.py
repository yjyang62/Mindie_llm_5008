# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
from ..qwen2.config_qwen2 import Qwen2Config


@dataclass
class Qwen2MoeConfig(Qwen2Config):
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 1408
    shared_expert_intermediate_size: int = 5632
    num_experts_per_tok: int = 4
    num_experts: int = 60
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: Optional[list] = None
    tp: bool = False
    has_shared_expert: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'qwen2_moe'
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []
        else:
            self.mlp_only_layers = self.mlp_only_layers

        self.is_dense_layer = [True if self.__check_dense_layer(layer_id) else False 
                        for layer_id in range(self.num_hidden_layers)]
        self.rope_scaling_dict = kwargs.get("rope_scaling", None)

    def __check_dense_layer(self, layer_id):
        if layer_id in self.mlp_only_layers:
            return True
        if self.num_experts > 0 and (layer_id + 1) % self.decoder_sparse_step != 0:
            return True
        return False