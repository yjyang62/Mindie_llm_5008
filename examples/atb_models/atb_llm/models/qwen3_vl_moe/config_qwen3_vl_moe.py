# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeVisionConfig

from ..base.config import BaseConfig
from ..base.flash_causal_multimodal import MultiModalConfig


@dataclass
class Qwen3vlmoeTextConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_dense_layer = [True if self.__check_dense_layer(layer_id) else False
                               for layer_id in range(self.num_hidden_layers)]

    def __check_dense_layer(self, layer_id):
        if layer_id in self.mlp_only_layers:
            return True
        if self.num_experts > 0 and (layer_id + 1) % self.decoder_sparse_step != 0:
            return True
        return False


@dataclass
class Qwen3vlmoeConfig(MultiModalConfig):

    def __init__(
            self,
            text_config,
            vision_config,
            **kwargs
        ):
        text_config = Qwen3vlmoeTextConfig(**text_config)
        vision_config = Qwen3VLMoeVisionConfig(**vision_config)
        super().__init__(vision_config=vision_config, text_config=text_config, **kwargs)
        self.max_position_embeddings = text_config.max_position_embeddings
        self.vocab_size = text_config.vocab_size