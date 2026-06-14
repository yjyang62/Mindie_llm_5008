# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from unittest.mock import patch, MagicMock

import pytest

from mindie_llm.runtime.models.qwen3_moe.input_builder_qwen3_moe import Qwen3MoeInputBuilder


class TestQwen3MoeInputBuilder:
    @pytest.fixture
    def input_builder(self):
        return Qwen3MoeInputBuilder(tokenizer=MagicMock())

    def test_apply_chat_template_parse_assistant_tool_call_json_success(self, input_builder):
        # 正常场景：仅处理assistant的tool_calls，JSON字符串转字典 + 调用父类方法返回结果
        conversation = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"beijing","date":"2026-01-12"}'}}]},
            {"role": "system", "content": "test"}
        ]
        with patch.object(input_builder.__class__.__bases__[0], "_apply_chat_template") as mock_super:
            mock_super.return_value = [100, 200, 300]
            result = input_builder._apply_chat_template(conversation, None)
            assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {"city": "beijing", "date": "2026-01-12"}
            mock_super.assert_called_once()
            assert result == mock_super.return_value

    def test_apply_chat_template_skip_invalid_case(self, input_builder):
        # 所有跳过场景：非assistant角色+tool_calls非列表+无tool_calls+空tool_calls
        conversation = [
            {"role": "user", "tool_calls": [{"function": {"arguments": '{"a":1}'}}]},  # 非assistant，不处理
            {"role": "assistant", "tool_calls": {"func": "dict_type"}},  # 非列表，不处理
            {"role": "assistant", "tool_calls": 123},  # 非列表，不处理
            {"role": "assistant", "content": "no tool call"},  # 无tool_calls
            {"role": "assistant", "tool_calls": []}  # 空tool_calls
        ]
        with patch.object(input_builder.__class__.__bases__[0], "_apply_chat_template"):
            input_builder._apply_chat_template(conversation, None)
            assert conversation[0]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
            assert isinstance(conversation[1]["tool_calls"], dict)
            assert isinstance(conversation[2]["tool_calls"], int)
            assert "tool_calls" not in conversation[3]
            assert not conversation[4]["tool_calls"]

    def test_apply_chat_template_json_parse_exception(self, input_builder):
        # 所有异常场景：JSON格式错误+非字符串arguments+多toolcall混合正常/异常
        conversation = [
            {"role": "assistant", "tool_calls": [
                {"function": {"arguments": '{"name":"func1", "p":1}'}},  # 正常解析
                {"function": {"arguments": "{invalid_json}"}},  # 解析失败，保留原内容
                {"function": {"arguments": 123}},  # 非字符串，跳过
                {"function": {}}  # 无arguments字段，跳过
            ]}
        ]
        with patch.object(input_builder.__class__.__bases__[0], "_apply_chat_template"):
            input_builder._apply_chat_template(conversation, None)
            assert conversation[0]["tool_calls"][0]["function"]["arguments"] == {"name": "func1", "p": 1}
            assert conversation[0]["tool_calls"][1]["function"]["arguments"] == "{invalid_json}"
            assert conversation[0]["tool_calls"][2]["function"]["arguments"] == 123
            assert "arguments" not in conversation[0]["tool_calls"][3]["function"]