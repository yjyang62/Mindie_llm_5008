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
from .input_builder_baichuan import BaichuanInputBuilder
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class BaichuanRouter(BaseRouter):
    @property
    def model_version(self):
        """
        次级模型名称，比如v2_13b
        :return:
        """
        if self.config_dict['num_hidden_layers'] == 40:  # 只有13b才是40层，同时兼容 v1 v2
            model_ver = "v2_13b"
        else:
            model_ver = "v2_7b"
        return model_ver

    def get_config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_dict(self.config_dict)
        if self.max_position_embeddings:
            config.model_max_length = self.max_position_embeddings  # 13b
            config.max_position_embeddings = self.max_position_embeddings
        super().check_config(config)
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=False
        )
        if not self.is_flash_causal_lm:
            # FA需要添加PAD token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def get_input_builder(self):
        return BaichuanInputBuilder(self.tokenizer, self.model_version, self.config.generation_config)