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
from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from .config_internlm2 import Internlm2Config
from .input_builder_internlm2 import Internlm2InputBuilder


@dataclass
class Internlm2Router(BaseRouter):

    @property
    def model_version(self):
        """
        次级模型名称
        :return:
        """
        return None

    def get_config(self):
        config = Internlm2Config.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False
        )
        return tokenizer

    def get_input_builder(self):
        return Internlm2InputBuilder(self.tokenizer, self.model_version, self.config.generation_config)