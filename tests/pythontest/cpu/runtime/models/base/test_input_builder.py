# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from unittest import mock
import pytest

from mindie_llm.runtime.models.base.input_builder import (
    InputBuilder,
    _preprocess_messages
)


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def apply_chat_template(self, conversation, **kwargs):

        tokens = []
        for msg in conversation:
            content = msg.get("content", "") or ""
            # Make sure each char -> 10 tokens to easily exceed max_length
            for c in content:
                tokens.extend([1000 + ord(c)] * 10)
        if kwargs.get("add_generation_prompt", False):
            tokens.append(9999)
        return tokens

    def encode(self, text):
        # Also return long tokens for history
        return [2000 + ord(c) for c in (text or "")] * 5

    def decode(self, tokens):
        return "".join(chr(t - 1000) for t in tokens if 1000 <= t < 2000)


def test_preprocess_messages_sets_content_to_empty_string():
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "tool_calls": [{"function": "get_weather"}], "content": None}
    ]
    _preprocess_messages(conversation)
    assert conversation[1]["content"] == ""


def test_preprocess_messages_no_change_when_not_needed():
    conversation = [{"role": "user", "content": "Hello"}]
    original = [msg.copy() for msg in conversation]
    _preprocess_messages(conversation)
    assert conversation == original


def test_init_sets_custom_chat_template():
    tokenizer = FakeTokenizer()
    builder = InputBuilder(tokenizer, chat_template="custom_jinja")
    assert tokenizer.chat_template == "custom_jinja"


def test_generate_position_ids():
    import numpy as np
    input_ids = np.array([5, 10, 15, 20])
    pos_ids = InputBuilder.generate_position_ids(input_ids)
    assert list(pos_ids) == [0, 1, 2, 3]


def test_apply_chat_template_fails_without_chat_template():
    tokenizer = FakeTokenizer(chat_template=None)
    builder = InputBuilder(tokenizer)
    with pytest.raises(RuntimeError, match="not configured with a `chat_template`"):
        builder._apply_chat_template([{"role": "user", "content": "hi"}])


@mock.patch("mindie_llm.runtime.models.base.input_builder.print_log")
def test_make_context_adapt_truncates_long_input(mock_print_log):
    tokenizer = FakeTokenizer(chat_template="dummy")
    builder = InputBuilder(tokenizer, max_length=10)  # small max_length
    conversation = [
        {"role": "system", "content": "S"},  # short system
        {"role": "user", "content": "This is a very long query that will exceed max_length"}  # long query
    ]
    tokens = builder.make_context(rank=0, conversation=conversation, adapt_to_max_length=True)
    assert len(tokens) == 10  # truncated to max_length

    calls = mock_print_log.call_args_list
    warning_calls = [call for call in calls if "truncated" in str(call)]
    assert len(warning_calls) > 0, "Expected 'truncated' warning not found in print_log calls"


@mock.patch("mindie_llm.runtime.models.base.input_builder.print_log")
def test_make_context_warns_on_non_user_last_message(mock_print_log):
    tokenizer = FakeTokenizer(chat_template="dummy")
    builder = InputBuilder(tokenizer, user_role_name="human")
    #  Last message is "assistant" → should trigger warning
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}  # ← last is assistant, not user!
    ]
    tokens = builder.make_context(rank=0, conversation=conversation, adapt_to_max_length=True)

    # Check ANY call contains the warning
    calls = mock_print_log.call_args_list
    warning_calls = [call for call in calls if "not offered by human" in str(call)]
    assert len(warning_calls) > 0, "Expected non-user warning not found"


def test_make_context_empty_conversation_raises_error():
    tokenizer = FakeTokenizer(chat_template="dummy")
    builder = InputBuilder(tokenizer)
    with pytest.raises(ValueError, match="The conversation is empty!"):
        builder.make_context(rank=0, conversation=[], adapt_to_max_length=True)


def test_make_multi_turns_context_includes_history():
    tokenizer = FakeTokenizer(chat_template="dummy")
    #  Increase max_length to accommodate history
    builder = InputBuilder(tokenizer, max_length=200)  # was 100
    
    #  Use shorter content to control token count
    conversation = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"}
    ]
    system_turn = [{"role": "system", "content": "S"}]
    query_turn = [{"role": "user", "content": "Q3"}]  # short query
    
    last_turn_tokens = tokenizer.apply_chat_template(
        system_turn + query_turn, add_generation_prompt=True
    )
    
    tokens = builder._make_multi_turns_context(
        conversation,
        system_turn,
        query_turn,
        last_turn_tokens,
        add_generation_prompt=True
    )
    
    # Now multi_turns should be within max_length, so tokens > last_turn_tokens
    assert len(tokens) > len(last_turn_tokens)