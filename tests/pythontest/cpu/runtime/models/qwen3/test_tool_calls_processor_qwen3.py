# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import re
from unittest import mock
import pytest

from mindie_llm.runtime.models.qwen3.tool_calls_processor_qwen3 import ToolCallsProcessorQwen3
from mindie_llm.runtime.models.base.tool_calls_processor import ToolCallsProcessorManager


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(t) for t in token_ids if 32 <= t <= 126)


def test_qwen3_processor_registered_in_manager():
    assert ToolCallsProcessorManager.get_tool_calls_processor("qwen3") is ToolCallsProcessorQwen3
    assert ToolCallsProcessorManager.get_tool_calls_processor("qwen3_moe") is ToolCallsProcessorQwen3
    assert ToolCallsProcessorManager.get_tool_calls_processor("hermes") is ToolCallsProcessorQwen3


def test_qwen3_processor_properties():
    """Test that start and end tokens are different but correct."""
    processor = ToolCallsProcessorQwen3()
    start = processor.tool_call_start_token
    end = processor.tool_call_end_token
    
    # They should be different characters
    assert start != end
    
    # But match expected token IDs
    assert processor.tool_call_start_token_id == 151657
    assert processor.tool_call_end_token_id == 151658
    assert isinstance(processor.tool_call_regex, re.Pattern)


def test_tool_call_regex_matches_valid_json():
    """Test regex with actual start and end tokens from processor."""
    processor = ToolCallsProcessorQwen3()
    start = processor.tool_call_start_token
    end = processor.tool_call_end_token
    regex = processor.tool_call_regex
    
    # Use the actual start and end tokens
    text = f'{start}{{"name": "get_weather", "arguments": {{"city": "Beijing"}}}}{end}'
    matches = regex.findall(text)
    assert len(matches) == 1
    assert json.loads(matches[0]) == {"name": "get_weather", "arguments": {"city": "Beijing"}}


def test_decode_with_valid_tool_calls():
    """Test decode with actual start and end tokens."""
    processor = ToolCallsProcessorQwen3(tokenizer=FakeTokenizer())
    start = processor.tool_call_start_token
    end = processor.tool_call_end_token
    
    content = f'Some text {start}{{"name": "get_weather", "arguments": {{"city": "Paris"}}}}{end} end'
    result = processor.decode(content)
    
    assert "content" in result
    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"


def test_decode_with_multiple_tool_calls():
    """Test multiple tool calls with actual tokens."""
    processor = ToolCallsProcessorQwen3(tokenizer=FakeTokenizer())
    start = processor.tool_call_start_token
    end = processor.tool_call_end_token
    
    content = (
        f'{start}{{"name": "search", "arguments": "AI"}}{end} '
        f'and then {start}{{"name": "summarize", "arguments": {{"text": "result"}}}}{end}'
    )
    result = processor.decode(content)
    
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["function"]["name"] == "search"
    assert result["tool_calls"][1]["function"]["name"] == "summarize"


def test_decode_without_tool_calls():
    processor = ToolCallsProcessorQwen3(tokenizer=FakeTokenizer())
    result = processor.decode("Plain text")
    assert result == {"content": "Plain text"}


def test_random_tool_call_id():
    processor = ToolCallsProcessorQwen3()
    id1 = processor._random_tool_calls_id()
    id2 = processor._random_tool_calls_id()
    assert id1.startswith("call_")
    assert len(id1) == 13
    assert id1 != id2