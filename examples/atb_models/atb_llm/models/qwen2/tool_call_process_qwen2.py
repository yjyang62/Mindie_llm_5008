# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import string
import random
import json
import re
from typing import Pattern

from ..base.tool_call_parser import ToolsCallProcessorWithXml, ToolParserManager


FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'
NAME = "name"
ARGUMENTS = "arguments"
CONTENT = "content"


class ToolsCallProcessorQwen2Base(ToolsCallProcessorWithXml):
    """Qwen 2 Base ToolCallProcessor"""
    
    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._tool_call_regex = re.compile(r'<tool_call>\s*({.*?})\s*</tool_call>', re.DOTALL)
    
    @property
    def tool_call_start_token(self) -> str:
        return "<tool_call>"

    @property
    def tool_call_end_token(self) -> str:
        return "</tool_call>"

    @property
    def tool_call_start_token_id(self) -> int:
        return 151657

    @property
    def tool_call_end_token_id(self) -> int:
        return 151658

    @property
    def tool_call_regex(self) -> Pattern:
        return self._tool_call_regex

    def decode(self, content):
        raise NotImplementedError("Subclasses must implement the 'decode' method.")


@ToolParserManager.register_module(["qwen1_5", "qwen_1_5", "qwen2", "qwen_2", "qwen1_5_or_2", "qwen_1_5_or_2"])
class ToolsCallProcessorQwen1_5_or_2(ToolsCallProcessorQwen2Base):
    """Qwen 1.5 ToolCallProcessor using ✿FUNCTION✿ format"""
    
    def decode(self, content):
        lines = content.strip()
        arguments_json = None
        is_tool_call = False
        if FN_NAME in lines and FN_ARGS in lines:
            arguments = lines.split(FN_ARGS)[1].split('✿')[0].strip(':').strip('\n').strip()
            function_name = lines.split(FN_NAME)[1].split('✿')[0].strip(':').strip('\n').strip()

            if function_name:
                is_tool_call = True
                try:
                    arguments_json = json.loads(arguments)
                except json.JSONDecodeError:
                    is_tool_call = False

            if is_tool_call:
                content_result = {
                    NAME: function_name,
                    ARGUMENTS: json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
                                            ensure_ascii=False)
                }
                characters = string.ascii_letters + string.digits
                call_id = "call_" + ''.join(random.choice(characters) for _ in range(8))
                call_res = {
                    "type": "function",
                    "id": call_id,
                    "function": content_result
                }
                return {
                    "tool_calls": [call_res]
                }
        return {CONTENT: content.strip()}


@ToolParserManager.register_module(["qwen2_5", "qwen_2_5"])
class ToolsCallProcessorQwen2_5(ToolsCallProcessorQwen2Base):
    """Qwen 2.5 ToolCallProcessor using <tool_call> XML format"""
    
    def decode(self, content):
        '''
        <tool_call>
        {"name": "get_rectangle_property", "arguments": {"perimeter": 14, "area": 15, "property": "length"}}
        </tool_call>
        '''
        lines = content.strip()
        pattern = re.compile(r'<tool_call>\s*({.*?})\s*</tool_call>', re.DOTALL)
        matches = pattern.findall(lines)
        is_tool_call = True
        if matches:
            try:
                tool_calls = [json.loads(match) for match in matches]
                for item in tool_calls:
                    _ = item[NAME]
                    _ = item[ARGUMENTS]
            except json.JSONDecodeError:
                is_tool_call = False

            if is_tool_call:
                call_res = []
                for item in tool_calls:
                    tool_call = {
                        NAME: item[NAME],
                        ARGUMENTS: json.dumps(item[ARGUMENTS], ensure_ascii=False) \
                            if isinstance(item[ARGUMENTS], dict) else item[ARGUMENTS]
                    }
                    characters = string.ascii_letters + string.digits
                    call_id = "call_" + ''.join(random.choice(characters) for _ in range(8))
                    res = {
                        "type": "function",
                        "id": call_id,
                        "function": tool_call
                    }
                    call_res.append(res)
                before_call_content = content.split("<tool_call>")[0]
                return {CONTENT: before_call_content, "tool_calls": call_res}
        return {CONTENT: content.strip()}


# Backward compatible processor that chooses based on initialization
@ToolParserManager.register_module(["qwen_auto"])
class ToolsCallProcessorQwenAuto(ToolsCallProcessorQwen2Base):
    """Auto-selecting Qwen ToolCallProcessor for backward compatibility"""
    
    def __init__(self, tokenizer=None, is_qwen1_5_or_2=False):
        super().__init__(tokenizer)
        self.is_qwen1_5_or_2 = is_qwen1_5_or_2
        
        # Delegate to specific processors
        if is_qwen1_5_or_2:
            self._processor = ToolsCallProcessorQwen1_5_or_2(tokenizer)
        else:
            self._processor = ToolsCallProcessorQwen2_5(tokenizer)
    
    def decode(self, content):
        return self._processor.decode(content)
