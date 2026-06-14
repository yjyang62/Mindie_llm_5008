# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass\

from atb_llm.models.base.config import BaseConfig
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger
from ..mixtral.config_mixtral import MixtralConfig
from ..qwen2.config_qwen2 import Qwen2Config


@dataclass
class VitaConfig(BaseConfig):
    def __init__(self, **kwargs):
        vision_config, audio_config = self._split_config(kwargs)
        self._init_visionconfig(vision_config)
        self._init_textconfig(kwargs)
        self._init_audioconfig(audio_config)
        self.projector_hidden_act = "gelu"
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_vocab_size", None)
        return output
    
    def _init_visionconfig(self, vision_config):
        if isinstance(vision_config, dict):
            self.mm_vision_tower = vision_config.get("mm_vision_tower", None)

    def _init_textconfig(self, text_config):
        if isinstance(text_config, dict):
            if self.model_type == 'vita-mixtral':
                self.text_config = MixtralConfig(**text_config)
                self.text_config.model_type = "mixtral"
            elif self.model_type == 'vita-Qwen2':
                self.text_config = Qwen2Config(**text_config)
                self.text_config.model_type = "qwen2"
        elif self.text_config is None:
            self.text_config = MixtralConfig(
            )

    def _init_audioconfig(self, audio_config):
        if isinstance(audio_config, dict):
            self.mm_audio_encoder = audio_config.get("mm_audio_encoder", None)
        else:
            logger.error("Vision config type should be dict.",
            ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise TypeError("Vision config type should be dict.")

    def _split_config(self, config):
        vision_keys = ["mm_vision_tower"]
        audio_keys = ["mm_audio_encoder", "mm_hidden_size", "mm_projector_type", "torch_dtype"]
        vision_config = {k: v for k, v in config.items() if k in vision_keys}
        audio_config = {k: v for k, v in config.items() if k in audio_keys}
        return vision_config, audio_config