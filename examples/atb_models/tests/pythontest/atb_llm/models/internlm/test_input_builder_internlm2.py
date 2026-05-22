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
from atb_llm.models.internlm2.config_internlm2 import Internlm2Config
from atb_llm.models.internlm2.input_builder_internlm2 import Internlm2InputBuilder

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
NUM_HIDDEN_LAYERS = 2
FAKE_CONFIG_DICT = {
    'model_type': 'internlm2',
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'max_position_embeddings': 2048,
    'vocab_size': 30000
}

FAKE_GENERATION_CONFIG_DICT = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 2,
}


class MockTokenizer:
    def __init__(self, use_fast=False):
        self.use_fast = use_fast
        self.add_bos_token = None
        self.bos_token = "<s>"

    @classmethod
    def __call__(cls, text):
        return {
            'input_ids': [1, 70447, 61297, 60504],
            'attention_mask': [1, 1, 1, 1],
            'input_text': text,
        }


@ddt
class TestInternlm2InputBuilder(unittest.TestCase):
    def setUp(self):
        self.internlm2_config = Internlm2Config(**FAKE_CONFIG_DICT)
        self.tokenizer = MockTokenizer()
        self.model_version = "v2"
        self.generation_config = FAKE_GENERATION_CONFIG_DICT

    def test_init(self):
        input_builder = Internlm2InputBuilder(self.tokenizer, self.model_version, self.generation_config)
        self.assertEqual(input_builder.model_version, self.model_version)
        self.assertEqual(input_builder.generation_config, self.generation_config)
        self.assertIsNotNone(input_builder.meta_instruction)
        self.assertIsInstance(input_builder.meta_instruction, str)

    def test_apply_chat_template(self):
        conversation = []
        input_builder = Internlm2InputBuilder(self.tokenizer, self.model_version, self.generation_config)
        prompt = input_builder._apply_chat_template(conversation)
        self.assertIsNotNone(prompt)
        self.assertIsInstance(prompt, list)


if __name__ == '__main__':
    unittest.main()