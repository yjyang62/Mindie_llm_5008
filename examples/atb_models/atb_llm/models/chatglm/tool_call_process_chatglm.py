# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import ast

from ..base.tool_call_parser import ToolsCallProcessor, ToolParserManager


@ToolParserManager.register_module(["chatglm2_6b", "chatglm_v2_6b", "chatglm_v2", "chatglm2"])
class ToolsCallProcessorChatglmV2(ToolsCallProcessor):
    """ChatGLM v2_6b ToolCallProcessor - returns content as-is"""
    
    def __init__(self, tokenizer=None):
        super().__init__(model_version="v2_6b")
        self.tokenizer = tokenizer
    
    def decode(self, content):
        return {"content": content}


@ToolParserManager.register_module(["chatglm3_6b", "chatglm_v3_6b", "chatglm_v3", "chatglm3"])
class ToolsCallProcessorChatglmV3(ToolsCallProcessor):
    """ChatGLM v3_6b ToolCallProcessor"""
    
    def __init__(self, tokenizer=None):
        super().__init__(model_version="v3_6b")
        self.tokenizer = tokenizer
    
    @staticmethod
    def parse_tool_call(tool_call_content):
        expr_ast = ast.parse(tool_call_content, mode='eval')
        node = expr_ast.body
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
                return kwargs
        raise ValueError("failed to parse tool calls.")

    def decode(self, content):
        tool_call_content = ""
        tool_call_list = []
        for response in content.split("<|assistant|>"):
            if len(response) == 0:
                continue
            metadata, tool_call_content = response.split("\n", maxsplit=1)
            tool_call_info = None
            if not metadata.strip():
                tool_call_content = tool_call_content.strip()
                tool_call_content = tool_call_content.replace(
                    "[[训练时间]]", "2023年")
            else:
                tool_call_content = "\n".join(
                    tool_call_content.split("\n")[1:2])
                try:
                    parameters = self.parse_tool_call(tool_call_content)
                except Exception:
                    break
                if isinstance(parameters, dict):
                    tool_call_info = {
                        "name": metadata.strip(),
                        "arguments": json.dumps(parameters, ensure_ascii=False)
                    }
            # decode success
            if isinstance(tool_call_info, dict):
                characters = string.ascii_letters + string.digits
                call_id = "call_" + \
                    ''.join(random.choice(characters) for _ in range(8))
                call_res = {
                    "type": "function",
                    "id": call_id,
                    "function": tool_call_info
                }
                tool_call_list.append(call_res)

        if len(tool_call_list) != 0:
            return {
                "tool_calls": tool_call_list
            }
        return {"content": content}


@ToolParserManager.register_module(["chatglm4_9b", "chatglm_v4_9b", "glm_4", "glm_4_9b"])
class ToolsCallProcessorChatglmV4(ToolsCallProcessor):
    """ChatGLM v4_9b ToolCallProcessor"""
    
    def __init__(self, tokenizer=None):
        super().__init__(model_version="v4_9b")
        self.tokenizer = tokenizer

    def decode(self, content):
        lines = content.strip().split("\n")
        arguments_json = None

        if len(lines) >= 2 and lines[1].startswith("{"):
            function_name = lines[0].strip()
            arguments = "\n".join(lines[1:]).strip()
            if function_name:
                is_tool_call = True
                try:
                    arguments_json = json.loads(arguments)
                except json.JSONDecodeError:
                    is_tool_call = False

                if is_tool_call:
                    content_result = {
                        "name": function_name,
                        "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
                                                ensure_ascii=False)
                    }
                    characters = string.ascii_letters + string.digits
                    call_id = "call_" + \
                        ''.join(random.choice(characters) for _ in range(8))
                    call_res = {
                        "type": "function",
                        "id": call_id,
                        "function": content_result
                    }
                    return {
                        "tool_calls": [call_res]
                    }

        return {"content": content.strip()}


# Default ChatGLM processor for backward compatibility
@ToolParserManager.register_module(["chatglm"])
class ToolsCallProcessorChatglm(ToolsCallProcessor):
    """Backward compatible ChatGLM ToolCallProcessor"""
    
    def __init__(self, tokenizer=None):
        super().__init__(model_version="")
        self.tokenizer = tokenizer
        # Default to v2 behavior
        self.processor = ToolsCallProcessorChatglmV2(tokenizer=self.tokenizer)
    
    def decode(self, content):
        return self.processor.decode(content)