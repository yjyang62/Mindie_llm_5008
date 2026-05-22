# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import re
import ast
import json
from typing import Pattern, Any, Literal
from atb_llm.utils.log import logger

from ..base.tool_call_parser import (
    DeltaToolCall,
    DeltaFunctionCall,
    ToolsCallProcessorWithXml,
    ToolParserManager,
    NAME,
    ARGUMENTS,
    TOOL_CALL_PORTION,
    INIT_RETURN_NONE,
    TOOL_CALLS
)


@ToolParserManager.register_module(["qwen3_coder", "qwen3coder"])
class ToolsCallProcessorQwen3Coder(ToolsCallProcessorWithXml):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._tool_call_regex = \
            re.compile(r'<tool_call>(.*?)</tool_call>', re.IGNORECASE | re.DOTALL)
        self._tool_call_function_regex = \
            re.compile(r"<function=(?P<functoin_name>.*?)>(?P<parameters>.*?)</function>", re.IGNORECASE | re.DOTALL)
        self._tool_call_parameter_regex = \
            re.compile(r"<parameter=(?P<parameter_key>.*?)>\s*(?P<parameter_value>.*?)\s*</parameter>",
                        re.IGNORECASE | re.DOTALL)
        self._tool_call_function_name_regex = \
            re.compile(r"<function=(?P<functoin_name>.*?)>", re.IGNORECASE | re.DOTALL)

    @property
    def tool_call_start_token(self) -> str:
        return "<tool_call>"                    # start_token of qwen3_coder

    @property
    def tool_call_end_token(self) -> str:
        return "</tool_call>"                   # end_token of qwen3_coder

    @property
    def tool_call_start_token_id(self) -> int:
        return 151657                           # start_token_id of qwen3_coder

    @property
    def tool_call_end_token_id(self) -> int:
        return 151658                           # end_token_id of qwen3_coder

    @property
    def tool_call_regex(self) -> Pattern:
        return self._tool_call_regex

    def extract_arguments(self, func_name, param_matches) -> dict:
        """
        Parse tool call text into JSON-like dicts.

        Args:
            param_matches (str): Containing a whole text with tag
            like <parameter=...>...</parameter><parameter=...>...</parameter>.

        Returns:
            dict: Parsed parameters objects. Returns an empty dict if parsing fails.
        """
        param_dict = dict()
        try:
            param_keys_values = self._tool_call_parameter_regex.findall(param_matches)

            for param_key_value in param_keys_values:
                param_key, param_value = param_key_value[0].strip(), param_key_value[1].strip()
                
                param_config = self._get_arguments_config(func_name, self.tools)

                param_dict[param_key] = self._convert_param_value(
                    param_value, param_key, param_config, func_name
                )

        except Exception as e:
            msg = f"Parse arguments text failed, {e}"
            logger.debug(msg)
            param_dict = dict()
        return param_dict


    def extract_single_tool_call(self, toll_call_match) -> dict:
        """
        Parse tool call text into JSON-like dicts.

        Args:
            toll_call_match (str): Containing a whole tool_call text with tag
            like <tool_call>...</tool_call>.

        Returns:
            dict: Parsed tool_call objects. Returns an empty dict if parsing fails.
        """
        try:
            result = self._tool_call_function_regex.search(toll_call_match)
            
            func_name = result.group(1).strip()
            func_args = result.group(2).strip()
            func_args_dict = self.extract_arguments(func_name, func_args)

            single_tool_call = {
                NAME: func_name,
                ARGUMENTS: func_args_dict
            }
            
        except Exception as e:
            msg = f"Parse tool_call text failed, {e}"
            logger.debug(msg)
            single_tool_call = {}

        return single_tool_call

    def get_tool_call_json(self, toll_call_matches: list):
        """
        Parse tool call texts into JSON-like dicts.

        Args:
            toll_call_matches (list of str): Each element is a string containing 
            the text with tag like <tool_call>...</tool_call>.

        Returns:
            list of dict: Parsed tool_call objects. Returns an empty list if
            parsing fails.
        """
        try:
            tool_calls = [self.extract_single_tool_call(match) for match in toll_call_matches]
            # check the key is valid.
            for item in tool_calls:
                _ = item[NAME]
                _ = item[ARGUMENTS]
        except Exception as e:
            logger.warning(f"Failed to parse tool_calls, {e}")
            tool_calls = []
        return tool_calls
    
    def _decode_stream_tool_call(self, tool_call_portion_dict: dict):
        """
        Process a `tool_call_portion` and return the result of `decode_stream`.

        Args:
            tool_call_portion_dict (dict): The dict contains the text segment generated after the '<tool_call>' token.

        Returns:
            dict: A dictionary representing the result of the decode_stream.
        """
        try:
            tool_call_portion = tool_call_portion_dict[TOOL_CALL_PORTION] # str
            current_tool_call_json = \
                self._parse_tool_call_portion_to_json(tool_call_portion) if tool_call_portion else None
        except Exception:
            # Invalid JSON fragment newline characters.
            return INIT_RETURN_NONE

        # case1：send function name
        if not self.current_tool_name_sent:
            if current_tool_call_json is None or not current_tool_call_json.get(NAME):
                return INIT_RETURN_NONE
            self.current_tool_name_sent = True
            return {TOOL_CALLS: [
                DeltaToolCall(index=self.current_tool_id, type="function", id=self.random_tool_call_id(),
                                function=DeltaFunctionCall(name=current_tool_call_json.get(NAME))).model_dump(
                    exclude_none=True)
            ]}
        
        # reset current_tool_name_sent for next function
        if tool_call_portion.rstrip().endswith('</function>'):
            self.current_tool_name_sent = False
            current_tool_call_json[ARGUMENTS] = '}'

        delta = {}
        # case2：send param
        cur_new_arguments = current_tool_call_json.get(ARGUMENTS)
        if cur_new_arguments is not None:
            delta = {TOOL_CALLS: [
                DeltaToolCall(index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=cur_new_arguments).model_dump(exclude_none=True))
                .model_dump(exclude_none=True)
            ]}
            self.current_tool_arguments_sent = True
        return delta
    
    def _parse_tool_call_portion_to_json(self, tool_call_portion: str) -> dict:
        """
        Parse a tool call text into a JSON-like dictionary using `tool_call_portion`.

        Args:
            tool_call_portion (str): The text segment generated after the '<tool_call>' token.

        Returns:
            dict: A dictionary representing the parsed tool call, including the current
                function name (if available) and any newly extracted arguments.
        """
        current_tool_call = {}
        # 捕获function_name, 用于current_tool_name_sent
        func_names = self._tool_call_function_name_regex.findall(tool_call_portion)
        if func_names:
            func_name = func_names[-1] # 最后一组func
            current_tool_call[NAME] = func_name
            tool_call_portion = tool_call_portion.split('<function=%s>' % func_name)[-1] # 取最后一组func的args

        # 捕获args，只在str当下包含最新</parameter>时候，返回新的一组args, 如果是第一组，则返回首括号，否则返回和前面链接符
        if tool_call_portion.rstrip().endswith('</parameter>'):
            param_keys_values = self._tool_call_parameter_regex.findall(tool_call_portion)
            if param_keys_values:
                param_key_value = param_keys_values[-1]
                param_key, param_value = param_key_value[0].strip(), param_key_value[1].strip()
                param_config = self._get_arguments_config(func_name, self.tools)

                param_value = self._convert_param_value(
                    param_value, param_key, param_config, func_name
                )
                current_tool_call[ARGUMENTS] = json.dumps({param_key: param_value})

            if len(param_keys_values) == 1:
                current_tool_call[ARGUMENTS] = current_tool_call[ARGUMENTS].rstrip()[:-1]
            else:
                current_tool_call[ARGUMENTS] = ', ' + current_tool_call[ARGUMENTS].strip()[1:-1]
        return current_tool_call
    
    def _get_arguments_config(self, func_name: str, tools: list) -> dict:
        """Extract argument configuration for a function. """

        if tools is None:
            return {}
        for config in tools:
            config_type = config.get('type')
            func = config.get('function')
            if config_type is None or func is None:
                continue

            name = func.get('name')
            if name is None:
                continue

            if not (config_type == "function" and name == func_name):
                continue

            params = func.get('parameters')
            if params is None:
                return {}
            
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            elif isinstance(params, dict):
                return params

        return {}

    def _convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str):
        """Convert parameter value based on its type in the schema. Adapted from vllm."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.debug(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.",
                    param_name, func_name,
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
            
        if param_type.startswith(("int", "uint", "long", "short", "unsigned")):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not an "
                    "integer in tool '%s', degenerating to string.",
                    param_value, param_name, func_name,
                )
                return param_value

        if param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a float "
                    "in tool '%s', degenerating to string.",
                    param_value, param_name, func_name,
                )
                return param_value
        
        if param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a boolean "
                    "(`true` or `false`) in tool '%s', degenerating to "
                    "false.",
                    param_value, param_name, func_name,
                )
            return param_value == "true"
            
        if (
            param_type in ["object", "array", "arr"]
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                param_value = json.loads(param_value)
                return param_value
            except (TypeError, ValueError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "parsed with json.loads in tool '%s', will try "
                    "other methods to parse it.",
                    param_value, param_name, func_name,
                )

        try:
            param_value = ast.literal_eval(param_value)  # safer
        except (ValueError, SyntaxError, TypeError):
            logger.debug(
                "Parsed value '%s' of parameter '%s' cannot be "
                "converted via Python `ast.literal_eval()` in tool "
                "'%s', degenerating to string.",
                param_value, param_name, func_name,
            )
        return param_value
