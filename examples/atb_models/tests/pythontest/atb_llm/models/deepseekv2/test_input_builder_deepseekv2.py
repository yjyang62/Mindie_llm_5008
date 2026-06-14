# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from ddt import ddt
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.models.deepseekv2.input_builder_deepseekv2 import Deepseekv2InputBuilder
from atb_llm.models import TruncationSide  # 导入截断枚举

ROLE = "role"
CONTENT = "content"
INPUT_STR = "who are you"
FAKE_CONFIG_DICT = {
    'model_type': 'deepseekv2',
    'num_hidden_layers': 61,
    'max_position_embeddings': 4096,
    'vocab_size': 163840,
    'rope_scaling': 1.0,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
}


class MockTokenizer:
    def __init__(self, use_fast=False, thinking=None):
        self.use_fast = use_fast
        self.init_kwargs = {"thinking": thinking}
        self.chat_template = "test"

    def apply_chat_template(self, conversation, thinking=False, tools=None, **kwargs):
        if tools:
            return [i for i in range(500)]
        max_length = kwargs.get("max_length", 100)
        if kwargs.get("truncation"):
            return [i for i in range(min(200, max_length))]
        return [i for i in range(200)]

    @classmethod
    def encode(cls, prompt, add_special_tokens):
        return prompt


class MockTokenizerNoChatTemplate:
    def __init__(self, use_fast=False):
        self.use_fast = use_fast
        self.init_kwargs = {}
        self.chat_template = "test"

    @classmethod
    def encode(cls, prompt, add_special_tokens):
        return prompt


@ddt
class TestDeepseekInputBuilder(unittest.TestCase):
    def setUp(self):
        self.deepseek_config = DeepseekV2Config(**FAKE_CONFIG_DICT)
        self.tokenizer = MockTokenizer()
        self.input_builder = Deepseekv2InputBuilder(self.tokenizer)
        self.input_builder.chat_template_kwargs = "chat_template_kwargs"
        self.input_builder.truncation = "truncation"

    def test_apply_chat_template_default(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        chat_template_kwargs_think = {"enable_thinking": "true", "truncation": "1"}
        chat_template_kwargs_nothink = {"enable_thinking": "false", "truncation": "0"}
        kwargs_think = {self.input_builder.chat_template_kwargs: chat_template_kwargs_think}
        kwargs_nothink = {self.input_builder.chat_template_kwargs: chat_template_kwargs_nothink}
        
        user_prompt_think = self.input_builder._apply_chat_template(
            user_conversation,** kwargs_think
        )
        user_prompt_nothink = self.input_builder._apply_chat_template(
            user_conversation, **kwargs_nothink
        )
        self.assertIsNotNone(user_prompt_think)
        self.assertIsInstance(user_prompt_think, list)
        self.assertIsNotNone(user_prompt_nothink)
        self.assertIsInstance(user_prompt_nothink, list)

    def test_apply_chat_template_raise(self):
        # no apply_chat_template
        tokenizer = MockTokenizerNoChatTemplate()
        input_builder = Deepseekv2InputBuilder(tokenizer)
        input_builder.chat_template_kwargs = "chat_template_kwargs"
        input_builder.truncation = "truncation"

        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        with self.assertRaises(RuntimeError) as cm:
            input_builder._apply_chat_template(user_conversation)
        self.assertIn("transformers version is detected to be <4.34", str(cm.exception))

        # no chat_template
        tokenizer = MockTokenizer()
        tokenizer.chat_template = ""
        input_builder = Deepseekv2InputBuilder(tokenizer)
        input_builder.chat_template_kwargs = "chat_template_kwargs"
        input_builder.truncation = "truncation"
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        with self.assertRaises(RuntimeError) as cm:
            input_builder._apply_chat_template(user_conversation)
        self.assertIn("it is not configured with a `chat_template`", str(cm.exception))

    def test_apply_chat_template_truncation_right(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR * 10}]
        chat_template_kwargs = {
            "enable_thinking": False,
            "truncation": TruncationSide.RIGHT,
            "max_length": 50
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(user_conversation,** kwargs)
        self.assertEqual(len(input_ids), 50)
        self.assertEqual(input_ids, [i for i in range(50)])

    def test_apply_chat_template_truncation_left(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR * 10}]
        chat_template_kwargs = {
            "enable_thinking": False,
            "truncation": TruncationSide.LEFT,
            "max_length": 50
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(user_conversation, **kwargs)
        self.assertEqual(len(input_ids), 50)
        self.assertEqual(input_ids, [i for i in range(150, 200)])

    def test_apply_chat_template_with_tools(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        tools_msg = {
            "tools": [{"name": "code_interpreter", "description": "Execute code"}],
            "tool_choice": "code_interpreter"
        }
        chat_template_kwargs = {
            "enable_thinking": True,
            "truncation": TruncationSide.RIGHT,
            "max_length": 100
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(
            user_conversation, tools_msg=tools_msg,** kwargs
        )
        self.assertEqual(len(input_ids), 500)

    def test_apply_chat_template_tools_truncation_left(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR * 10}]
        tools_msg = {
            "tools": [{"name": "code_interpreter", "description": "Execute code"}],
            "tool_choice": "code_interpreter"
        }
        chat_template_kwargs = {
            "enable_thinking": False,
            "truncation": TruncationSide.LEFT,
            "max_length": 100
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(
            user_conversation, tools_msg=tools_msg, **kwargs
        )
        self.assertEqual(len(input_ids), 100)
        self.assertEqual(input_ids, [i for i in range(400, 500)])


if __name__ == '__main__':
    unittest.main(verbosity=2)