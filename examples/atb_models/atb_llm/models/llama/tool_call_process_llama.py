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

from ..base.tool_call_parser import ToolsCallProcessor, ToolParserManager


@ToolParserManager.register_module(["llama", "llama3", "llama3_1"])
class ToolsCallProcessorLlama(ToolsCallProcessor):
    def __init__(self, tokenizer=None):
        super().__init__(model_version="")
        self.tokenizer = tokenizer
        self.name_key = "name"
        self.param_key = "parameters"
        
    def decode(self, content):
        if not self.__content_is_ok(content):
            return {"content": content}
        tool_call_list = []
        tool_call_content = json.loads(content)
        tool_call_detail = {
            self.name_key: tool_call_content.get(self.name_key),
            "arguments": json.dumps(tool_call_content.get(self.param_key), ensure_ascii=False)
        }
        characters = string.ascii_letters + string.digits
        call_id = "call_" + \
            ''.join(random.choice(characters) for _ in range(8))
        call_res = {
            "type": "function",
            "id": call_id,
            "function": tool_call_detail
        }
        tool_call_list.append(call_res)
        return {"tool_calls": tool_call_list}

    def __content_is_ok(self, content: str):
        tool_call_content = ""
        try:
            tool_call_content = json.loads(content)
        except json.JSONDecodeError:
            return False
        if not isinstance(tool_call_content, dict):
            return False
        if (self.name_key not in tool_call_content) or (self.param_key not in tool_call_content):
            return False
        return True