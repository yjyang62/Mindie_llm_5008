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
from atb_llm.models.telechat.router_telechat import TelechatRouter


@dataclass
class MockGenerationConfig:
    user_token_id: int = 1
    system_token_id: int = 2
    bot_token_id: int = 3
    eos_token_id: int = 4
    max_new_tokens: int = 100
    max_length: int = 2000


@dataclass
class MockConfig:
    seq_length: int = 2048
    generation_config: MockGenerationConfig = None

    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = MockGenerationConfig()

    @classmethod
    def from_dict(cls, config_dict):
        return cls()

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    'model_type': 'telechat',
    'num_hidden_layers': 32,
    'max_position_embeddings': 2048,
    'vocab_size': 30000
}

ALTERNATIVE_CONFIG_DICT = {
    'model_type': 'telechat',
    'num_hidden_layers': 16,
    'max_position_embeddings': 1024,
    'vocab_size': 15000
}


class MockTokenizer:
    def __init__(self, use_fast=False, trust_remote_code=False):
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code


class TestTelechatRouter(unittest.TestCase):
    def test_get_config(self):
        # Test with default config
        router = TelechatRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT)
        
        config = router.get_config()
        self.assertIsNotNone(config)
        # Check that config has the expected attributes
        self.assertTrue(hasattr(config, 'seq_length'))
        self.assertTrue(hasattr(config, 'generation_config'))
        # Check the values
        self.assertEqual(config.seq_length, 2048)
        # Check generation_config attributes
        self.assertTrue(hasattr(config.generation_config, 'user_token_id'))
        self.assertTrue(hasattr(config.generation_config, 'system_token_id'))
        self.assertTrue(hasattr(config.generation_config, 'bot_token_id'))
        self.assertTrue(hasattr(config.generation_config, 'eos_token_id'))
        
        # Test with alternative config
        router = TelechatRouter(FAKE_MODEL_NAME_OR_PATH, ALTERNATIVE_CONFIG_DICT)
        config = router.get_config()
        self.assertEqual(config.seq_length, 1024)

    def test_model_version_property(self):
        # Test model_version property access
        router = TelechatRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT)
        
        # Since TelechatRouter doesn't override model_version, it should return empty string
        self.assertEqual(router.model_version, "")
        
        # Test with different config
        router = TelechatRouter(FAKE_MODEL_NAME_OR_PATH, ALTERNATIVE_CONFIG_DICT)
        self.assertEqual(router.model_version, "")

    def test_inheritance(self):
        # Test that TelechatRouter properly inherits from BaseRouter
        router = TelechatRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT)
        
        # Check that the router has the expected attributes from BaseRouter
        self.assertTrue(hasattr(router, 'config_dict'))
        self.assertTrue(hasattr(router, 'model_name_or_path'))
        self.assertTrue(hasattr(router, 'model_version'))

if __name__ == '__main__':
    unittest.main()
    