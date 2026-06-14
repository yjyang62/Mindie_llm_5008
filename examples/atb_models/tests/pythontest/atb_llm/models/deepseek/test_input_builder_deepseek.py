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
from atb_llm.models.deepseek.config_deepseek import DeepseekConfig
from atb_llm.models.deepseek.input_builder_deepseek import DeepseekInputBuilder

ROLE = "role"
CONTENT = "content"
INPUT_STR = "who are you"
NUM_HIDDEN_LAYERS = 28
FAKE_CONFIG_DICT = {
    'model_type': 'deepseek',
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'max_position_embeddings': 4096,
    'vocab_size': 102400,
}


class MockTokenizer:
    def __init__(self, use_fast=False):
        self.use_fast = use_fast

    @classmethod
    def encode(cls, prompt, add_special_tokens):
        return prompt


@ddt
class TestDeepseekInputBuilder(unittest.TestCase):
    def setUp(self):
        self.deepseek_config = DeepseekConfig(**FAKE_CONFIG_DICT)
        self.tokenizer = MockTokenizer()
        self.model_version = "deepseek"

    def test_init(self):
        input_builder = DeepseekInputBuilder(self.tokenizer, self.model_version)
        self.assertEqual(input_builder.model_version, self.model_version)

    def test_apply_chat_template_default(self):
        input_builder = DeepseekInputBuilder(self.tokenizer, self.model_version)

        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        user_prompt = input_builder.apply_chat_template_default(user_conversation)
        self.assertIsNotNone(user_prompt)
        self.assertIsInstance(user_prompt, str)

        assist_conversation = [{ROLE: "assistant", CONTENT: INPUT_STR}]
        assist_prompt = input_builder.apply_chat_template_default(assist_conversation)
        self.assertIsNotNone(assist_prompt)
        self.assertIsInstance(assist_prompt, str)

        sys_conversation = [{ROLE: "system", CONTENT: INPUT_STR}]
        sys_prompt = input_builder.apply_chat_template_default(sys_conversation)
        self.assertIsNotNone(sys_prompt)
        self.assertIsInstance(sys_prompt, str)


if __name__ == '__main__':
    unittest.main()