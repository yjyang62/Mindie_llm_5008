# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
import unittest
from unittest.mock import patch
from dataclasses import dataclass
from atb_llm.models.baichuan.router_baichuan import BaichuanRouter


@dataclass
class MockGenerationConfig:
    user_token_id: int = 1
    assistant_token_id: int = 2
    eos_token_id: int = 3


@dataclass
class MockConfig:
    model_max_length: int = 2048
    max_position_embeddings: int = 2048
    generation_config: MockGenerationConfig = None

    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = MockGenerationConfig()

    @classmethod
    def from_pretrained(cls, model_path):
        return cls()

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    'model_type': 'baichuan',
    'num_hidden_layers': 40,
    'max_position_embeddings': 2048,
    'vocab_size': 30000
}


class MockTokenizer:
    def __init__(self, use_fast=False, trust_remote_code=False):
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code
        self.add_special_tokens_called = False

    def add_special_tokens(self, tokens):
        self.add_special_tokens_called = True


def mock_safe_get_tokenizer_from_pretrained(_, **kwargs):
    use_fast = kwargs.get("use_fast", False)
    trust_remote_code = kwargs.get("trust_remote_code", False)
    return MockTokenizer(use_fast, trust_remote_code)


class TestBaichuanRouter(unittest.TestCase):
    def test_get_config(self):
        # Test with default config
        router = BaichuanRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT)
        
        config = router.get_config()
        self.assertIsNotNone(config)
        # Check that config has the expected attributes
        self.assertTrue(hasattr(config, 'max_position_embeddings'))
        self.assertTrue(hasattr(config, 'generation_config'))
        # Check the values
        self.assertEqual(config.max_position_embeddings, 2048)
        
        # Test with max_position_embeddings setting
        router = BaichuanRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT, max_position_embeddings=4096)
        config = router.get_config()
        self.assertEqual(config.max_position_embeddings, 4096)
        
        # Test with max_position_embeddings = None
        router = BaichuanRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT, max_position_embeddings=None)
        config = router.get_config()
        self.assertEqual(config.max_position_embeddings, 2048)

    @patch("atb_llm.models.baichuan.router_baichuan.safe_get_tokenizer_from_pretrained")
    def test_get_tokenizer(self, mock_get_tokenizer):
        # Setup mock tokenizer
        mock_tokenizer = MockTokenizer()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        # Test with flash_causal_lm
        router = BaichuanRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT, is_flash_causal_lm=True)
        
        tokenizer = router.get_tokenizer()
        # Verify safe_get_tokenizer_from_pretrained was called with correct parameters
        mock_get_tokenizer.assert_called_once_with(
            router.tokenizer_path,
            revision=router.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=router.trust_remote_code,
            use_fast=False
        )
        # Verify tokenizer attributes
        self.assertFalse(tokenizer.use_fast)
        self.assertFalse(tokenizer.trust_remote_code)
        self.assertFalse(tokenizer.add_special_tokens_called)
        
        # Reset mock for next test
        mock_get_tokenizer.reset_mock()
        
        # Test without flash_causal_lm
        router = BaichuanRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT, is_flash_causal_lm=False)
        tokenizer = router.get_tokenizer()
        # Verify safe_get_tokenizer_from_pretrained was called with correct parameters
        mock_get_tokenizer.assert_called_once_with(
            router.tokenizer_path,
            revision=router.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=router.trust_remote_code,
            use_fast=False
        )
        # Verify add_special_tokens was called
        self.assertTrue(tokenizer.add_special_tokens_called)

    def test_model_version(self):
        # Test 13B version
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        config_dict['num_hidden_layers'] = 40
        
        router = BaichuanRouter(FAKE_MODEL_NAME_OR_PATH, config_dict)
        self.assertEqual(router.model_version, "v2_13b")

        # Test 7B version
        config_dict['num_hidden_layers'] = 32
        router = BaichuanRouter(FAKE_MODEL_NAME_OR_PATH, config_dict)
        self.assertEqual(router.model_version, "v2_7b")

if __name__ == '__main__':
    unittest.main()