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
from unittest import mock
import pytest

from mindie_llm.runtime.tokenizer.tokenizer_wrapper import TokenizerWrapper


class FakeConfig:
    def __init__(self, is_reasoning_model=True):
        self.is_reasoning_model = is_reasoning_model


class FakeTokenizer:
    def __init__(self):
        self.init_kwargs = {"enable_thinking": True}
    
    def __call__(self, text, **kwargs):
        # Return object with tolist() method
        class FakeTensor:
            def tolist(self):
                return [1001, 1002, 1003]
        return {"input_ids": [FakeTensor()]}
    
    def decode(self, token_ids, **kwargs):
        # Return different text based on input length to avoid "no update" in streaming
        if isinstance(token_ids, list):
            return f"decoded_{''.join(str(t) for t in token_ids[-3:])}"  # e.g., "decoded_123"
        return "decoded_text"


class FakeInputBuilder:
    def make_context(self, rank, conversation, **kwargs):
        return [2001, 2002]


class FakeToolCallsParser:
    def __init__(self):
        self.tools = None
        
    def decode(self, content):
        return {"tool_calls": [{"function": {"name": "test_tool"}}]}
    
    def decode_stream(self, all_token_ids, prev_idx, curr_idx, skip_special, delta_text):
        return {"tool_calls": [{"delta": "test"}], "metadata": {}}


class FakeReasoningParser:
    def single_process_reasoning(self, token_ids):
        return [3001, 3002], [3003]
    
    def stream_process_reasoning(self, token_ids, curr_idx):
        return [3001], [3003]
    
    def is_reasoning_end(self, token_ids):
        return 42 in token_ids
    
    def count_reasoning_tokens(self, token_ids):
        return 0  # simple implementation for test


class FakeLlmConfig:
    def __init__(self):
        self.llm = mock.MagicMock()
        self.llm.enable_reasoning = True


class FakeRouter:
    def __init__(self):
        self.config = FakeConfig()
        self.tokenizer = FakeTokenizer()
        self.input_builder = FakeInputBuilder()
        self.tool_calls_processor = FakeToolCallsParser()
        self.reasoning_parser = FakeReasoningParser()
        self.llm_config = FakeLlmConfig()


@pytest.fixture(autouse=True)
def mock_get_model():
    with mock.patch("mindie_llm.runtime.tokenizer.tokenizer_wrapper.get_router_ins") as mock_get:
        mock_get.return_value = FakeRouter()
        yield mock_get


def create_wrapper(**kwargs):
    if "models_dict" not in kwargs:
        kwargs["models_dict"] = "{}"
    return TokenizerWrapper("fake_model_path", **kwargs)


def test_encode_non_chatting():
    wrapper = create_wrapper()
    result = wrapper.encode("Hello world", is_chatting=False)
    assert result == [1001, 1002, 1003]  # from FakeTensor.tolist()


def test_encode_chatting():
    wrapper = create_wrapper()
    conversation = [{"role": "user", "content": "Hi"}]
    result = wrapper.encode(conversation, is_chatting=True)
    assert result == [2001, 2002]


def test_decode_non_streaming_with_reasoning():
    wrapper = create_wrapper()
    result = wrapper.decode(
        all_token_ids=[1, 2, 3],
        skip_special_tokens=True,
        use_tool_calls=False,
        is_chat_req=True,
        stream=False
    )
    assert "reasoning_content" in result
    assert "content" in result


def test_decode_non_streaming_with_tool_calls():
    wrapper = create_wrapper()
    wrapper.config.is_reasoning_model = False
    
    result = wrapper.decode(
        all_token_ids=[1, 2, 3],
        skip_special_tokens=True,
        use_tool_calls=True,
        is_chat_req=True,
        stream=False
    )
    assert "tool_calls" in result


def test_decode_non_streaming_with_reasoning_and_tool_calls():
    wrapper = create_wrapper()
    result = wrapper.decode(
        all_token_ids=[1, 2, 3],
        skip_special_tokens=True,
        use_tool_calls=True,
        is_chat_req=True,
        stream=False,
        metadata={"tools": [{"name": "test"}]}
    )
    assert "tool_calls" in result
    assert "reasoning_content" in result


def test_decode_streaming_reasoning():
    wrapper = create_wrapper()
    with mock.patch.object(wrapper.reasoning_parser, 'is_reasoning_end', return_value=False):
        result = wrapper.decode(
            all_token_ids=[1, 2, 3],
            skip_special_tokens=True,
            use_tool_calls=False,
            is_chat_req=True,
            stream=True,
            curr_decode_index=2
        )
    # Should have content from stream_process_reasoning
    assert "reasoning_content" in result or "content" in result


def test_is_use_reasoning_parser_true():
    wrapper = create_wrapper()
    metadata = {}
    assert wrapper._is_use_reasoning_parser(metadata) is True


def test_is_use_reasoning_parser_disabled_by_llm_config():
    wrapper = create_wrapper()
    wrapper.llm_config.llm.enable_reasoning = False
    assert wrapper._is_use_reasoning_parser({}) is False


def test_is_use_reasoning_parser_request_override():
    wrapper = create_wrapper()
    metadata = {"req_enable_thinking": False}
    assert wrapper._is_use_reasoning_parser(metadata) is False


def test_extract_reasoning_content():
    wrapper = create_wrapper()
    result = wrapper._extract_reasoning_content([1, 2, 3], skip_special_tokens=True)
    assert "reasoning_content" in result
    assert "content" in result


def test_decode_streaming_tool_calls():
    """Cover extract_tool_calls_streaming path (without reasoning)."""
    wrapper = create_wrapper()
    # Disable reasoning to isolate tool_calls path
    wrapper.enable_thinking = False
    result = wrapper.decode(
        all_token_ids=[1, 2, 3, 4, 5],
        skip_special_tokens=True,
        use_tool_calls=True,
        is_chat_req=True,
        stream=True,
        curr_decode_index=3,
        prev_decode_index=1,
        metadata={}
    )
    assert "tool_calls" in result
    assert "metadata" in result


def test_decode_streaming_no_new_content():
    """Cover 'no new content' branch in streaming."""
    wrapper = create_wrapper()
    with mock.patch.object(wrapper, '_tokenizer_decode', side_effect=["same", "same"]):
        result = wrapper.decode(
            all_token_ids=[1, 2, 3],
            skip_special_tokens=True,
            use_tool_calls=False,
            is_chat_req=True,
            stream=True,
            curr_decode_index=2,
            prev_decode_index=1
        )
        assert result == {"update_index": False}


def test_decode_streaming_incomplete_character():
    """Cover 'ends with ' branch."""
    wrapper = create_wrapper()
    with mock.patch.object(wrapper, '_tokenizer_decode', return_value="incomplete"):
        result = wrapper.decode(
            all_token_ids=[1, 2, 3],
            skip_special_tokens=True,
            use_tool_calls=False,
            is_chat_req=True,
            stream=True,
            curr_decode_index=2,
            prev_decode_index=1
        )
        assert result == {"update_index": False}


def test_get_combined_stream_result_reasoning_not_ended():
    """Cover get_combined_stream_result when reasoning not ended."""
    wrapper = create_wrapper()
    with mock.patch.object(wrapper.reasoning_parser, 'is_reasoning_end', return_value=False):
        result = wrapper._get_combined_stream_result(
            all_token_ids=[1, 2, 3],
            prev_decode_index=0,
            curr_decode_index=2,
            skip_special_tokens=True,
            delta_text="new",
            metadata={}
        )
        assert "reasoning_content" in result or "content" in result


def test_get_combined_stream_result_reasoning_ended_in_delta():
    """Cover get_combined_stream_result when reasoning ends in delta."""
    wrapper = create_wrapper()
    with mock.patch.object(wrapper.reasoning_parser, 'is_reasoning_end', return_value=True):
        with mock.patch.object(wrapper.reasoning_parser, 'stream_process_reasoning', return_value=([3001], [3003])):
            result = wrapper._get_combined_stream_result(
                all_token_ids=[1, 2, 3],
                prev_decode_index=1,
                curr_decode_index=2,
                skip_special_tokens=True,
                delta_text="</think>tool",
                metadata={}
            )
            assert "reasoning_content" in result


def test_get_combined_stream_result_reasoning_ended_not_in_delta():
    """Cover get_combined_stream_result when reasoning ended but not in delta."""
    wrapper = create_wrapper()
    
    def mock_is_reasoning_end(token_ids):
        return len(token_ids) == 3  # only full has end
    with mock.patch.object(wrapper.reasoning_parser, 'is_reasoning_end', side_effect=mock_is_reasoning_end):
        result = wrapper._get_combined_stream_result(
            all_token_ids=[1, 2, 3],
            prev_decode_index=1,
            curr_decode_index=2,
            skip_special_tokens=True,
            delta_text="more",
            metadata={}
        )
        # Should return tool_calls (from extract_tool_calls_streaming)
        assert "tool_calls" in result


def test_extract_reasoning_content_streaming_empty():
    """Cover extract_reasoning_content_streaming with empty results."""
    wrapper = create_wrapper()
    with mock.patch.object(wrapper.reasoning_parser, 'stream_process_reasoning', return_value=([], [])):
        with mock.patch.object(wrapper, '_tokenizer_decode', return_value=""):
            result = wrapper._extract_reasoning_content_streaming([1, 2, 3], 2, True)
            assert result == {}  # empty because v is empty string


def test_decode_non_streaming_metadata_reasoning_tokens():
    """Cover the metadata reasoning_tokens addition."""
    wrapper = create_wrapper()
    result = wrapper.decode(
        all_token_ids=[1, 2, 3],
        skip_special_tokens=True,
        use_tool_calls=False,
        is_chat_req=False,
        stream=False
    )
    assert "metadata" in result
    assert "reasoning_tokens" in result["metadata"]
    assert result["metadata"]["reasoning_tokens"] == 0


def test__is_use_reasoning_parser_no_metadata():
    """Cover _is_use_reasoning_parser with None metadata."""
    wrapper = create_wrapper()
    assert wrapper._is_use_reasoning_parser(None) is True


