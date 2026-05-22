# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
#
# Implement part of this file based on transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from transformers import PretrainedConfig
from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.models.base.flash_causal_multimodal import MultiModalConfig
from .modeling_navit_siglip import SiglipVisionConfig


SLICE_CONFIG_DEFAULT = {
    'max_slice_nums': 9,
    'patch_size': 14,
    'scale_resolution': 448
}


VISION_CONFIG_DEFAULT = {
    "hidden_size": 1152,
    "model_type": "siglip",
    "intermediate_size": 4304,
    "image_size": 980,
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14,
}


class Minicpmqwen2v2SliceConfig(PretrainedConfig):
    model_type = "minicpmv"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_slice_nums = kwargs.get('max_slice_nums')
        self.patch_size = kwargs.get('patch_size')
        self.scale_resolution = kwargs.get('scale_resolution')


class Minicpmqwen2v2Config(MultiModalConfig):
    model_type = "minicpmv"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        query_num=64,
        slice_config=None,
        vision_config=None,
        **kwargs,
    ):
        self.query_num = query_num

        if isinstance(slice_config, dict):
            self.slice_config = Minicpmqwen2v2SliceConfig(**slice_config)
        else:
            self.slice_config = Minicpmqwen2v2SliceConfig(**SLICE_CONFIG_DEFAULT)
        self.slice_mode = True
        
        if isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)
        else:
            vision_config = SiglipVisionConfig(**VISION_CONFIG_DEFAULT)

        self.patch_size = vision_config.patch_size

        text_config = Qwen2Config(**kwargs)
        super().__init__(vision_config=vision_config, text_config=text_config, **kwargs)