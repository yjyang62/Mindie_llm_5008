# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
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
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Implement InternVisionConfig based on InternVisionConfig from OpenGVLab/InternVL-Chat-V1-5
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from atb_llm.models.base.config import BaseConfig


@dataclass
class InternVisionConfig(BaseConfig):
    num_channels: int = 3
    patch_size: int = 14
    image_size: int = 224
    qkv_bias: bool = False
    hidden_size: int = 3200
    num_attention_heads: int = 25
    intermediate_size: int = 12800
    qk_normalization: bool = True
    num_hidden_layers: int = 48
    use_flash_attn: bool = True
    hidden_act: str = "gelu"
    norm_type: str = "rms_norm"
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'intern_vit_6b'
