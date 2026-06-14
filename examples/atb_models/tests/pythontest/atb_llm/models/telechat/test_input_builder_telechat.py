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
from atb_llm.models.telechat.input_builder_telechat import TelechatInputBuilder


@dataclass
class MockConfig:
    seq_length: int = 2048


@dataclass
class MockGenerationConfig:
    user_token_id: int = 1
    system_token_id: int = 2
    bot_token_id: int = 3
    eos_token_id: int = 4
    max_new_tokens: int = 100
    max_length: int = 2000


class MockTokenizer:
    def __init__(self):
        self.encode_count = 0

    def __call__(self, text):
        self.encode_count += 1
        return {
            'input_ids': [100 + self.encode_count] * 10  # Return 10 tokens for each call
        }


class TestTelechatInputBuilder(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.model_version = "v1"
        self.config = MockConfig()
        self.generation_config = MockGenerationConfig()

    def test_init(self):
        input_builder = TelechatInputBuilder(self.tokenizer, self.model_version, self.config, self.generation_config)
        self.assertEqual(input_builder.model_version, self.model_version)
        self.assertEqual(input_builder.config, self.config)
        self.assertEqual(input_builder.generation_config, self.generation_config)

    def test_apply_chat_template(self):
        input_builder = TelechatInputBuilder(self.tokenizer, self.model_version, self.config, self.generation_config)
        
        # Test single turn conversation
        conversation = [
            {"role": "user", "content": "Hello"}
        ]
        result = input_builder._apply_chat_template(conversation)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0], self.generation_config.user_token_id)
        self.assertEqual(result[-1], self.generation_config.bot_token_id)

    def test_apply_chat_template_with_system(self):
        input_builder = TelechatInputBuilder(self.tokenizer, self.model_version, self.config, self.generation_config)
        
        # Test conversation with system message
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        result = input_builder._apply_chat_template(conversation)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0], self.generation_config.system_token_id)

    def test_apply_chat_template_max_length_error(self):
        # Test with invalid max_length
        generation_config = MockGenerationConfig(max_new_tokens=3000)  # Larger than seq_length
        input_builder = TelechatInputBuilder(self.tokenizer, self.model_version, self.config, generation_config)
        
        conversation = [
            {"role": "user", "content": "Hello"}
        ]
        with self.assertRaises(ValueError) as context:
            input_builder._apply_chat_template(conversation)
        self.assertIn("Please change max_new_tokens", str(context.exception))

if __name__ == '__main__':
    unittest.main()