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

from mindie_llm.runtime.models.qwen3.input_builder_qwen3 import Qwen3InputBuilder


def create_mock_tokenizer(chat_template="dummy", enable_thinking=True):
    tokenizer = mock.MagicMock()
    tokenizer.init_kwargs = {"enable_thinking": enable_thinking}
    tokenizer.chat_template = chat_template
    tokenizer.apply_chat_template = mock.MagicMock(return_value=[1001, 1002, 1003])
    return tokenizer


def test_init_reads_enable_thinking_from_tokenizer():
    """Test that config_enable_thinking is correctly read from tokenizer.init_kwargs."""
    tokenizer = create_mock_tokenizer(enable_thinking=False)
    builder = Qwen3InputBuilder(tokenizer)
    assert builder.config_enable_thinking is False


def test_init_defaults_enable_thinking_to_true():
    """Test that if enable_thinking is not in init_kwargs, it defaults to True."""
    tokenizer = mock.MagicMock()
    tokenizer.init_kwargs = {}  # no enable_thinking
    tokenizer.chat_safe = "dummy"
    tokenizer.chat_template = "dummy"
    builder = Qwen3InputBuilder(tokenizer)
    assert builder.config_enable_thinking is True


def test_apply_chat_template_uses_config_enable_thinking_by_default():
    """Test that when no request-level enable_thinking is provided, config value is used."""
    tokenizer = create_mock_tokenizer(enable_thinking=True)
    builder = Qwen3InputBuilder(tokenizer)
    
    conversation = [{"role": "user", "content": "Hello"}]
    builder._apply_chat_template(conversation)
    
    tokenizer.apply_chat_template.assert_called_once_with(
        conversation,
        enable_thinking=True
    )


def test_apply_chat_template_request_level_overrides_config():
    """Test that request-level enable_thinking overrides the config value."""
    tokenizer = create_mock_tokenizer(enable_thinking=False)  # config says False
    builder = Qwen3InputBuilder(tokenizer)
    
    conversation = [{"role": "user", "content": "Hello"}]
    builder._apply_chat_template(
        conversation,
        chat_template_kwargs={"enable_thinking": True}  # request says True
    )
    
    tokenizer.apply_chat_template.assert_called_once_with(
        conversation,
        chat_template_kwargs={"enable_thinking": True},
        enable_thinking=True
    )


def test_apply_chat_template_with_tools():
    """Test that tools are correctly passed to tokenizer when tools_msg is provided."""
    tokenizer = create_mock_tokenizer()
    builder = Qwen3InputBuilder(tokenizer)
    
    conversation = [{"role": "user", "content": "Call tool"}]
    tools_msg = {"tools": [{"name": "weather", "description": "Get weather"}]}
    
    builder._apply_chat_template(conversation, tools_msg=tools_msg)
    
    tokenizer.apply_chat_template.assert_called_once_with(
        conversation,
        tools=[{"name": "weather", "description": "Get weather"}],
        enable_thinking=True
    )


def test_apply_chat_template_with_empty_tools():
    """Test behavior when tools_msg is provided but tools is empty or None."""
    tokenizer = create_mock_tokenizer()
    builder = Qwen3InputBuilder(tokenizer)
    
    conversation = [{"role": "user", "content": "Hi"}]
    
    # Case 1: tools_msg with empty tools list
    builder._apply_chat_template(conversation, tools_msg={"tools": []})
    call1 = tokenizer.apply_chat_template.call_args_list[-1]
    assert call1[1].get("tools") == []
    
    # Case 2: tools_msg with None tools
    builder._apply_chat_template(conversation, tools_msg={"tools": None})
    call2 = tokenizer.apply_chat_template.call_args_list[-1]
    assert call2[1].get("tools") is None


def test_apply_chat_template_fails_without_chat_template():
    """Test that RuntimeError is raised when tokenizer has no chat_template."""
    tokenizer = mock.MagicMock()
    tokenizer.init_kwargs = {}
    tokenizer.chat_template = None  # no chat template
    
    builder = Qwen3InputBuilder(tokenizer)
    conversation = [{"role": "user", "content": "Hello"}]
    
    with pytest.raises(RuntimeError, match="not configured with a `chat_template`"):
        builder._apply_chat_template(conversation)


def test_apply_chat_template_fails_without_apply_chat_template_method():
    """Test that RuntimeError is raised when tokenizer lacks apply_chat_template."""
    tokenizer = mock.MagicMock()
    del tokenizer.apply_chat_template  # remove the method
    tokenizer.init_kwargs = {}
    tokenizer.chat_template = "dummy"
    
    builder = Qwen3InputBuilder(tokenizer)
    conversation = [{"role": "user", "content": "Hello"}]
    
    with pytest.raises(RuntimeError, match="transformers version is detected to be <4.34"):
        builder._apply_chat_template(conversation)


def test_make_context_calls_apply_chat_template():
    """Test that make_context ultimately calls our _apply_chat_template."""
    tokenizer = create_mock_tokenizer()
    builder = Qwen3InputBuilder(tokenizer)
    
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    
    # Mock _apply_chat_template to avoid real logic
    with mock.patch.object(builder, '_apply_chat_template', return_value=[1, 2, 3]) as mock_apply:
        tokens = builder.make_context(rank=0, conversation=conversation, adapt_to_max_length=False)
        
        mock_apply.assert_called_once()
        assert tokens == [1, 2, 3]


def test_apply_chat_template_with_none_chat_template_kwargs():
    """Test behavior when chat_template_kwargs is None."""
    tokenizer = create_mock_tokenizer(enable_thinking=False)
    builder = Qwen3InputBuilder(tokenizer)
    
    conversation = [{"role": "user", "content": "Hello"}]

    builder._apply_chat_template(conversation)  # no chat_template_kwargs
    
    tokenizer.apply_chat_template.assert_called_once_with(
        conversation,
        enable_thinking=False  # config value
        # chat_template_kwargs is not in kwargs, so not passed
    )


def test_apply_chat_template_passes_other_kwargs():
    """Test that other kwargs are passed through to tokenizer."""
    tokenizer = create_mock_tokenizer()
    builder = Qwen3InputBuilder(tokenizer)
    
    conversation = [{"role": "user", "content": "Hello"}]
    builder._apply_chat_template(
        conversation,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True}
    )
    
    tokenizer.apply_chat_template.assert_called_once_with(
        conversation,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
        enable_thinking=True
    )