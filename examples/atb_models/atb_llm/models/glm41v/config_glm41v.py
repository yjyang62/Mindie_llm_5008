# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig

from ..base.config import BaseConfig
from ..base.flash_causal_multimodal import MultiModalConfig


@dataclass
class Glm41vConfig(MultiModalConfig):

    def __init__(
            self,
            text_config,
            vision_config,
            **kwargs
        ):
        self.model_type = "glm41v"
        text_config = BaseConfig(**text_config)
        vision_config = Glm4vVisionConfig(**vision_config)
        super().__init__(vision_config=vision_config, text_config=text_config, **kwargs)
        self.max_position_embeddings = text_config.max_position_embeddings
        self.vocab_size = text_config.vocab_size