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
from ..base.router import BaseRouter
from ..llama.config_llama import LlamaConfig


@dataclass
class MistralRouter(BaseRouter):
    def __post_init__(self):
        super().__post_init__()
        self.model_type = "llama"  # mistral-7b-instruct-v0.2与llama结构相同
        self.model_type_cap = self.model_type.capitalize()

    def get_config(self):
        config = LlamaConfig.from_pretrained(self.model_name_or_path)
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        super().check_config(config)
        return config

    def get_tokenizer(self):
        tokenizer = super().get_tokenizer()
        # 需要添加PAD token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer