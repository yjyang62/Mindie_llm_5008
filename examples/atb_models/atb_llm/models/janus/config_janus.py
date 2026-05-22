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
from janus.models.modeling_vlm import MultiModalityConfig

from ..base.config import BaseConfig
from ..llama.config_llama import LlamaConfig


@dataclass
class JanusConfig(BaseConfig):
    max_position_embeddings: int = 16384
    vocab_size: int = 102400
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = 'janus'

        multi_modality_config = MultiModalityConfig(**kwargs)
        self.vision_config = multi_modality_config.vision_config
        self.aligner_config = multi_modality_config.aligner_config
        self.gen_vision_config = multi_modality_config.gen_vision_config
        self.gen_aligner_config = multi_modality_config.gen_aligner_config
        self.gen_head_config = multi_modality_config.gen_head_config

        language_config = kwargs.get("language_config", {})
        self.language_config = LlamaConfig.from_dict(language_config)
        self.language_config.head_dim = self.language_config.hidden_size // self.language_config.num_attention_heads