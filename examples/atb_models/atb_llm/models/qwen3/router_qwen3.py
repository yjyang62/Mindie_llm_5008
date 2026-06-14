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

from .config_qwen3 import Qwen3Config
from .input_builder_qwen3 import Qwen3InputBuilder
from ..base.reasoning_parser import CommonReasoningParser
from ..qwen2.router_qwen2 import Qwen2Router


@dataclass
class Qwen3Router(Qwen2Router):

    def __post_init__(self):
        # qwen3 继承qwen2
        super().__post_init__()
        self.model_type = "qwen2"
        self.model_type_cap = self.model_type.capitalize()

    def get_config(self):
        self.config_dict.update({"transformers_version": self.transformers_version})
        config = Qwen3Config.from_dict(self.config_dict)
        super().check_config(config)
        return config

    def get_reasoning_parser(self):
        """initialize reasoning parser"""
        return CommonReasoningParser(self.config)

    def get_input_builder(self):
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        return Qwen3InputBuilder(self.tokenizer, **kwargs)
    
    def get_tool_call_parser(self):
        return "qwen3"
        