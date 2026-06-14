# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import re
from typing import Pattern

from ..base.tool_call_parser import ToolsCallProcessorWithXml, ToolParserManager


@ToolParserManager.register_module(["qwen3", "qwen3_moe", "hermes"])
class ToolsCallProcessorQwen3(ToolsCallProcessorWithXml):
    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._tool_call_regex = re.compile(r'<tool_call>\s*({.*?})\s*</tool_call>', re.DOTALL)

    @property
    def tool_call_start_token(self) -> str:
        return "<tool_call>"                    # start_token of qwen3

    @property
    def tool_call_end_token(self) -> str:
        return "</tool_call>"                   # end_token of qwen3

    @property
    def tool_call_start_token_id(self) -> int:
        return 151657                           # start_token_id of qwen3

    @property
    def tool_call_end_token_id(self) -> int:
        return 151658                           # end_token_id of qwen3

    @property
    def tool_call_regex(self) -> Pattern:
        return self._tool_call_regex
