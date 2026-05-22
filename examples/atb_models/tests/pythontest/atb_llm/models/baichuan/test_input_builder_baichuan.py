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
from dataclasses import dataclass
from atb_llm.models.baichuan.input_builder_baichuan import BaichuanInputBuilder


@dataclass
class MockGenerationConfig:
    user_token_id: int = 1
    assistant_token_id: int = 2
    eos_token_id: int = 3


class MockTokenizer:
    def __init__(self):
        self.encode_count = 0

    def encode(self, text):
        self.encode_count += 1
        return [100 + self.encode_count]  # Return different token IDs for each call


class TestBaichuanInputBuilder(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.model_version = "v2_13b"
        self.generation_config = MockGenerationConfig()

    def test_init(self):
        input_builder = BaichuanInputBuilder(self.tokenizer, self.model_version, self.generation_config)
        self.assertEqual(input_builder.model_version, self.model_version)
        self.assertEqual(input_builder.generation_config, self.generation_config)

    def test_apply_chat_template(self):
        input_builder = BaichuanInputBuilder(self.tokenizer, self.model_version, self.generation_config)
        
        # Test single turn conversation
        conversation = [
            {"role": "user", "content": "Hello"}
        ]
        result = input_builder._apply_chat_template(conversation)
        # The result should be: user_token_id + encoded tokens + assistant_token_id
        expected = [self.generation_config.user_token_id, 101, self.generation_config.assistant_token_id]
        self.assertEqual(result, expected)

        # Test multi-turn conversation
        self.tokenizer.encode_count = 0  # Reset counter
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"}
        ]
        result = input_builder._apply_chat_template(conversation)
        # The result should be built from the conversation in reverse order:
        # Last user message: user_token_id + encoded "How are you?" + assistant_token_id
        # Previous assistant reply: assistant_token_id + encoded "Hi" + eos_token_id
        # First user message: user_token_id + encoded "Hello" + assistant_token_id
        expected = [
            self.generation_config.user_token_id, 103, 
            self.generation_config.assistant_token_id, 102, self.generation_config.eos_token_id,
            self.generation_config.user_token_id, 101,
            self.generation_config.assistant_token_id
        ]
        self.assertEqual(result, expected)

    def test_apply_chat_template_invalid_role(self):
        input_builder = BaichuanInputBuilder(self.tokenizer, self.model_version, self.generation_config)
        
        # Test invalid role
        conversation = [
            {"role": "invalid", "content": "Hello"}
        ]
        with self.assertRaises(ValueError) as context:
            input_builder._apply_chat_template(conversation)
        self.assertIn("message role not supported yet", str(context.exception))

if __name__ == '__main__':
    unittest.main()