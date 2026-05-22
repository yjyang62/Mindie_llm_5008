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

from .config_llama import LlamaConfig
from .input_builder_llama import LlamaInputBuilder
from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class LlamaRouter(BaseRouter):
    def get_config(self):
        config = LlamaConfig.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_tokenizer(self):
        if self.config_dict['num_hidden_layers'] in [60]:
            # LLaMa 33B use_fast需要使用False
            use_fast = False
        else:
            use_fast = True
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=False,
            use_fast=use_fast
        )
        # 需要添加PAD token
        tokenizer.pad_token_id = 0
        return tokenizer
    
    def get_input_builder(self):
        kwargs = {"model_version": self.model_version}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        return LlamaInputBuilder(self.tokenizer, **kwargs)
    
    def get_tool_call_parser(self):
        return "llama"