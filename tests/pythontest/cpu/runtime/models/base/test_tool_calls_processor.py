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

from mindie_llm.runtime.models.base.tool_calls_processor import (
    ToolCallsProcessor,
    ToolCallsProcessorWithXml,
    ToolCallsProcessorManager,
    _count_closing_braces_at_end
)
from mindie_llm.runtime.utils.helpers.json_completor import FillMode


class MockXmlProcessor(ToolCallsProcessorWithXml):
    """Concrete implementation for testing."""
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._tool_call_start_token = "<<<"
        self._tool_call_end_token = ">>>"
        self._tool_call_start_token_id = 1001
        self._tool_call_end_token_id = 1002
        self._tool_call_regex = re.compile(r'<<<(.*?)>>>', re.DOTALL)

    @property
    def tool_call_start_token(self) -> str:
        return self._tool_call_start_token

    @property
    def tool_call_end_token(self) -> str:
        return self._tool_call_end_token

    @property
    def tool_call_start_token_id(self) -> int:
        return self._tool_call_start_token_id

    @property
    def tool_call_end_token_id(self) -> int:
        return self._tool_call_end_token_id

    @property
    def tool_call_regex(self) -> re.Pattern:
        return self._tool_call_regex


class FakeTokenizer:
    """Minimal tokenizer for testing."""
    def decode(self, token_ids, skip_special_tokens=False):
        # Map token IDs to strings for testing
        mapping = {
            1001: "<<<",  # start token
            1002: ">>>",  # end token
            123: "{",
            125: "}",
            34: '"',
            58: ":",
            97: "a",
            98: "b",
            101: "e",
            110: "n",
            111: "o",
            115: "s",
            116: "t",
        }
        return "".join(mapping.get(t, chr(t)) for t in token_ids if 32 <= t <= 126)


def test_tool_parser_manager_register_and_get():
    original_processors = ToolCallsProcessorManager.get_tool_calls_processors().copy()
    ToolCallsProcessorManager._tool_calls_processors.clear()

    try:
        @ToolCallsProcessorManager.register_module(module_names=["test_parser"])
        class TestProcessor(ToolCallsProcessor):
            def __init__(self, model_version: str):
                super().__init__(model_version)

        processor_cls = ToolCallsProcessorManager.get_tool_calls_processor("test_parser")
        assert processor_cls is TestProcessor
        assert "test_parser" in ToolCallsProcessorManager.get_tool_calls_processors()
    finally:
        ToolCallsProcessorManager._tool_calls_processors.clear()
        ToolCallsProcessorManager._tool_calls_processors.update(original_processors)


def test_tool_parser_manager_register_force_override():
    original_processors = ToolCallsProcessorManager.get_tool_calls_processors().copy()
    ToolCallsProcessorManager._tool_calls_processors.clear()

    try:
        class ParserV1(ToolCallsProcessor):
            def __init__(self, model_version: str): super().__init__(model_version)

        class ParserV2(ToolCallsProcessor):
            def __init__(self, model_version: str): super().__init__(model_version)

        ToolCallsProcessorManager.register_module(module_names=["my_parser"], module=ParserV1)
        assert ToolCallsProcessorManager.get_tool_calls_processor("my_parser") is ParserV1

        with mock.patch("mindie_llm.runtime.models.base.tool_calls_processor.logger.warning") as mock_warn:
            ToolCallsProcessorManager.register_module(module_names=["my_parser"], module=ParserV2, force=False)
            mock_warn.assert_called()
        assert ToolCallsProcessorManager.get_tool_calls_processor("my_parser") is ParserV1

        ToolCallsProcessorManager.register_module(module_names=["my_parser"], module=ParserV2, force=True)
        assert ToolCallsProcessorManager.get_tool_calls_processor("my_parser") is ParserV2
    finally:
        ToolCallsProcessorManager._tool_calls_processors.clear()
        ToolCallsProcessorManager._tool_calls_processors.update(original_processors)


def test_tool_parser_manager_remove():
    original_processors = ToolCallsProcessorManager.get_tool_calls_processors().copy()
    ToolCallsProcessorManager._tool_calls_processors.clear()

    try:
        @ToolCallsProcessorManager.register_module(module_names=["to_remove"])
        class RemovableProcessor(ToolCallsProcessor):
            def __init__(self, model_version: str): super().__init__(model_version)

        assert "to_remove" in ToolCallsProcessorManager.get_tool_calls_processors()

        with mock.patch("mindie_llm.runtime.models.base.tool_calls_processor.logger.debug") as mock_debug:
            ToolCallsProcessorManager.remove_tool_calls_processor("to_remove")
            mock_debug.assert_called()
        assert "to_remove" not in ToolCallsProcessorManager.get_tool_calls_processors()

        with mock.patch("mindie_llm.runtime.models.base.tool_calls_processor.logger.warning") as mock_warn:
            ToolCallsProcessorManager.remove_tool_calls_processor("non_existent")
            mock_warn.assert_called()
    finally:
        ToolCallsProcessorManager._tool_calls_processors.clear()
        ToolCallsProcessorManager._tool_calls_processors.update(original_processors)


def test_tool_parser_manager_get_non_existent():
    original_processors = ToolCallsProcessorManager.get_tool_calls_processors().copy()
    ToolCallsProcessorManager._tool_calls_processors.clear()

    try:
        with pytest.raises(KeyError, match="not found in registered"):
            ToolCallsProcessorManager.get_tool_calls_processor("missing")
    finally:
        ToolCallsProcessorManager._tool_calls_processors.clear()
        ToolCallsProcessorManager._tool_calls_processors.update(original_processors)


def test_base_tools_call_processor_decode():
    processor = ToolCallsProcessor(model_version="1.0")
    result = processor.decode("Hello world")
    assert result == {"content": "Hello world"}


def test_mock_xml_processor_decode():
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    content = 'Some text <<<{"name": "get_weather", "arguments": {"city": "Paris"}}>>> end'
    result = processor.decode(content)
    assert "content" in result
    assert "tool_calls" in result
    tool_calls = result["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Paris"}


def test_mock_xml_processor_decode_no_tool_calls():
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    result = processor.decode("Plain text without tool calls")
    assert result == {"content": "Plain text without tool calls"}


def test_get_tool_call_json_valid():
    matches = ['{"name": "func1", "arguments": {"x": 1}}', '{"name": "func2", "arguments": "value"}']
    tool_calls = MockXmlProcessor._get_tool_calls_json(matches)
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "func1"
    assert tool_calls[1]["name"] == "func2"


def test_get_tool_call_json_invalid():
    matches = ['{"arguments": {"x": 1}}']
    tool_calls = MockXmlProcessor._get_tool_calls_json(matches)
    assert tool_calls == []

    matches = ['{invalid json}']
    tool_calls = MockXmlProcessor._get_tool_calls_json(matches)
    assert tool_calls == []


def test_count_closing_braces_at_end():
    assert _count_closing_braces_at_end("hello}}}") == 3
    assert _count_closing_braces_at_end("hello}} }") == 1
    assert _count_closing_braces_at_end("hello") == 0
    assert _count_closing_braces_at_end("}") == 1
    assert _count_closing_braces_at_end("") == 0


def test_random_tool_call_id():
    id1 = MockXmlProcessor._random_tool_calls_id()
    id2 = MockXmlProcessor._random_tool_calls_id()
    assert id1.startswith("call_")
    assert len(id1) == 13
    assert id1 != id2


def test_preprocess_delta_text():
    processor = MockXmlProcessor(FakeTokenizer())
    assert processor._preprocess_delta_text("test") == "test"


def test_count_tool_tokens():
    processor = MockXmlProcessor(FakeTokenizer())
    history = [1001, 1002, 1001]
    current = [1001, 1002, 1001, 1002]
    prev_start, prev_end, curr_start, curr_end = processor._count_tool_tokens(history, current)
    assert prev_start == 2
    assert prev_end == 1
    assert curr_start == 2
    assert curr_end == 2


def test_decode_with_empty_tool_call():
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    content = '<<<{"name": "test", "arguments": {}}>>>'
    result = processor.decode(content)
    assert "tool_calls" in result
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {}


def test_tool_parser_manager_register_with_none_name():
    original_processors = ToolCallsProcessorManager.get_tool_calls_processors().copy()
    ToolCallsProcessorManager._tool_calls_processors.clear()

    try:
        class TestProc(ToolCallsProcessor):
            def __init__(self, model_version: str): super().__init__(model_version)

        ToolCallsProcessorManager.register_module(module_names=None, module=TestProc)
        processor_cls = ToolCallsProcessorManager.get_tool_calls_processor("TestProc")
        assert processor_cls is TestProc
    finally:
        ToolCallsProcessorManager._tool_calls_processors.clear()
        ToolCallsProcessorManager._tool_calls_processors.update(original_processors)


def test_random_tool_call_id_uniqueness():
    ids = set()
    for _ in range(100):
        ids.add(MockXmlProcessor._random_tool_calls_id())
    assert len(ids) == 100


def test_decode_stream_basic():
    """Basic decode_stream call for coverage."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    
    # Before tool calls
    result = processor.decode_stream(
        all_token_ids=[72, 101],  # "He"
        prev_decode_index=0,
        curr_decode_index=2,
        skip_special_tokens=False,
        delta_text="He"
    )
    assert result == {"content": "He"}


def test_decode_stream_with_tool_start():
    """decode_stream with tool start token (for coverage)."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    
    result = processor.decode_stream(
        all_token_ids=[1001],  # "<<<"
        prev_decode_index=0,
        curr_decode_index=1,
        skip_special_tokens=False,
        delta_text="<<<"
    )
    # Should not crash; may return {} or content
    assert isinstance(result, dict)


def test_decode_stream_error_handling():
    """Test error handling in decode_stream."""
    tokenizer = FakeTokenizer()
    tokenizer.decode = mock.Mock(side_effect=Exception("decode error"))
    processor = MockXmlProcessor(tokenizer)
    
    result = processor.decode_stream(
        all_token_ids=[1001],
        prev_decode_index=0,
        curr_decode_index=1,
        skip_special_tokens=False,
        delta_text="<<<"
    )
    assert result == {}  # INIT_RETURN_NONE


def test_decode_stream_tool_call_with_special_delta():
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    processor.current_tool_id = 0
    processor.current_tool_name_sent = True
    processor.current_tool_arguments_sent = False
    
    tool_call_portion_dict = {
        "tool_call_portion": '{"arguments":"{\\"key\\": \\"value"}',
        "delta_text": '{"',
        "fill_mode": FillMode.Full
    }
    result = processor._decode_stream_tool_calls(tool_call_portion_dict)
    assert isinstance(result, dict)


def test_decode_stream_tool_call_arguments_delta():
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    processor.current_tool_id = 0
    processor.current_tool_name_sent = True
    processor.current_tool_arguments_sent = True
    
    tool_call_portion_dict = {
        "tool_call_portion": '{"arguments":"value"}',
        "delta_text": 'e"}',
        "fill_mode": FillMode.Full
    }
    result = processor._decode_stream_tool_calls(tool_call_portion_dict)
    assert "tool_calls" in result


def test_get_tool_call_json_missing_arguments():
    matches = ['{"name": "test"}']
    tool_calls = MockXmlProcessor._get_tool_calls_json(matches)
    assert tool_calls == []


def test_get_tool_call_json_empty_string():
    matches = ['']
    tool_calls = MockXmlProcessor._get_tool_calls_json(matches)
    assert tool_calls == []


def test_decode_stream_with_json_end_quotation():
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    processor.current_tool_id = 0
    processor.current_tool_name_sent = True
    processor.current_tool_arguments_sent = True
    all_token_ids = [1001, 34, 125, 1002]
    result = processor.decode_stream(
        all_token_ids=all_token_ids,
        prev_decode_index=2,
        curr_decode_index=4,
        skip_special_tokens=False,
        delta_text='"}>>>'
    )
    assert isinstance(result, dict)


def test_init_tools_call_processor():
    """Test base class initialization."""
    processor = ToolCallsProcessor("v1.0")
    assert processor.model_version == "v1.0"


def test_init_tools_call_processor_with_xml():
    """Test XML processor initialization."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    assert processor.tokenizer is tokenizer
    assert processor.current_tool_id == -1


def test_decode_spilt_token_property():
    """Test decode_spilt_token property returns start token."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    assert processor.decode_spilt_token == "<<<"


def test_tool_call_start_end_properties():
    """Test token properties."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    assert processor.tool_call_start_token == "<<<"
    assert processor.tool_call_end_token == ">>>"
    assert processor.tool_call_start_token_id == 1001
    assert processor.tool_call_end_token_id == 1002


def test_random_tool_call_id_format():
    """Test random ID format."""
    id_str = MockXmlProcessor._random_tool_calls_id()
    assert id_str.startswith("call_")
    assert len(id_str) == 13


def test_count_closing_braces_at_end_empty():
    """Test edge case for brace counting."""
    assert _count_closing_braces_at_end("") == 0


def test_delta_tool_call_model():
    """Test Pydantic model creation."""
    from mindie_llm.runtime.models.base.tool_calls_processor import DeltaToolCall
    tool_call = DeltaToolCall(index=0, function={"name": "test"})
    assert tool_call.index == 0
    assert tool_call.function.name == "test"


def test_delta_function_call_model():
    """Test Pydantic model creation."""
    from mindie_llm.runtime.models.base.tool_calls_processor import DeltaFunctionCall
    func_call = DeltaFunctionCall(name="test", arguments="{}")
    assert func_call.name == "test"
    assert func_call.arguments == "{}"


def test_message_filter_import():
    """Ensure message_filter is importable (for coverage)."""
    from mindie_llm.utils.log.logging import message_filter
    assert callable(message_filter)


def test_decode_stream_return_types():
    """Test decode_stream returns dict in all paths."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    result = processor.decode_stream([1], 0, 1, False, "")
    assert isinstance(result, dict)


def test_stream_tool_call_properties_return_none():
    """Cover stream_tool_call_* properties that return None."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    assert processor.stream_tool_call_portion_regex is None
    assert processor.stream_tool_call_name_regex is None


def test_get_tool_call_json_empty_input():
    """Cover empty matches input."""
    tool_calls = MockXmlProcessor._get_tool_calls_json([])
    assert tool_calls == []


def test_count_tool_tokens_empty_inputs():
    """Cover _count_tool_tokens with empty lists."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    prev_start, prev_end, curr_start, curr_end = processor._count_tool_tokens([], [])
    assert (prev_start, prev_end, curr_start, curr_end) == (0, 0, 0, 0)


def test_decode_with_none_content():
    """Cover decode with None content (though unlikely)."""
    tokenizer = FakeTokenizer()
    processor = MockXmlProcessor(tokenizer)
    # This will raise in real code, but let's cover the path
    with pytest.raises(AttributeError):  # because None.strip() fails
        processor.decode(None)