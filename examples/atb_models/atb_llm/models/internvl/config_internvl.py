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
# Implement InternvlConfig based on InternVLChatConfig from OpenGVLab/InternVL
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

from atb_llm.models.base.config import BaseConfig
from atb_llm.models.internvl.config_intern_vit import InternVisionConfig
from atb_llm.models.internvl.flash_causal_internvl import INTERNLM2_ARCHITECTURE, LLAMA_ARCHITECTURE, QWEN2_ARCHITECTURE
from atb_llm.models.internlm2.config_internlm2 import Internlm2Config
from atb_llm.models.llama.config_llama import LlamaConfig
from atb_llm.models.qwen2.config_qwen2 import Qwen2Config

from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger


@dataclass
class InternvlConfig(BaseConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(self,
                 vision_config=None,
                 llm_config=None,
                 use_backbone_lora=0,
                 use_llm_lora=0,
                 select_layer=-1,
                 force_image_size=None,
                 downsample_ratio=0.5,
                 template=None,
                 dynamic_image_size=False,
                 use_thumbnail=False,
                 ps_version='v1',
                 min_dynamic_patch=1,
                 max_dynamic_patch=12,
                 **kwargs):
        llm_config["quantize"] = None
        llm_config["quantization_config"] = None
        super().__init__(**llm_config)

        self.vision_config = InternVisionConfig(**vision_config)
        llm_model_architectures = llm_config['architectures'][0]
        if llm_model_architectures == INTERNLM2_ARCHITECTURE:
            self.llm_config = Internlm2Config(**llm_config)
        elif llm_model_architectures == LLAMA_ARCHITECTURE:
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_model_architectures == QWEN2_ARCHITECTURE:
            self.llm_config = Qwen2Config(**llm_config)
        else:
            error_msg = (f"{llm_model_architectures} is an unsupported architecture, "
                         "check `llm_config['architectures']` in config.json, "
                         "currently only InternLM2ForCausalLM, LlamaForCausalLM and Qwen2ForCausalLM are supported.")
            logger.error(error_msg, ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError(error_msg)
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
