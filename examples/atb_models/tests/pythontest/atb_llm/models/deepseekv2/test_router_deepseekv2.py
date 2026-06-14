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
from unittest.mock import patch, MagicMock
from ddt import ddt, unpack, data
from atb_llm.models.deepseekv2.router_deepseekv2 import Deepseekv2Router
from atb_llm.utils.parameter_validators import DictionaryParameterValidator
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config

NUM_HIDDEN_LAYERS = 61

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    'model_type': 'deepseek_v3',
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'max_position_embeddings': 163840,
    'vocab_size': 129280,
    'rope_scaling': None,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'topk_method': "noaux_tc",
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "first_k_dense_replace": 3,
    "n_routed_experts": 256,
    "q_lora_rank": 1536,
}

FAKE_CONFIG_DICT_1 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_1["n_routed_experts"] = 1
FAKE_CONFIG_DICT_2 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_2["n_routed_experts"] = 4
FAKE_CONFIG_DICT_3 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_3["num_hidden_layers"] = 2
FAKE_CONFIG_DICT_4 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_4["topk_method"] = "fake"
FAKE_CONFIG_DICT_5 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_5["topk_method"] = "greedy"
FAKE_CONFIG_DICT_5["topk_group"] = 1
FAKE_CONFIG_DICT_5["n_group"] = 2

fake_groups = [
    (FAKE_CONFIG_DICT_1,), (FAKE_CONFIG_DICT_2,), (FAKE_CONFIG_DICT_3,), (FAKE_CONFIG_DICT_4,), (FAKE_CONFIG_DICT_5,)
]


class MockTokenizer:
    def __init__(self, use_fast, trust_remote_code):
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code
        self.eos_token_id = 1
        self.unk_token_id = -1
    
    def convert_tokens_to_ids(self, token):
        if token == "<think>":
            return 123456
        elif token == "</think>":
            return 123457
        else:
            return self.unk_token_id


class MockTokenizerNonThink(MockTokenizer):
    def convert_tokens_to_ids(self, token):
        return self.unk_token_id


def mock_safe_get_tokenizer_from_pretrained(_, **kwargs):
    use_fast = kwargs.get("use_fast")
    trust_remote_code = kwargs.get("trust_remote_code")
    return MockTokenizer(use_fast, trust_remote_code)


@ddt
class TestDeepseekv2Router(unittest.TestCase):
    @patch("atb_llm.models.base.router.BaseRouter.check_config", return_value=None)
    def test_get_config(self, _1):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = Deepseekv2Router("", config_dict)
        config = router.get_config()
        self.assertIsNotNone(config)

    @data(*fake_groups)
    @unpack
    def test_check_config_deepseekv2(self, config_dict):
        router = Deepseekv2Router("", config_dict)
        config = DeepseekV2Config.from_dict(config_dict)
        with self.assertRaises(ValueError):
            router.check_config_deepseekv2(config)

    @unpack
    @patch("atb_llm.models.deepseekv2.router_deepseekv2.safe_get_tokenizer_from_pretrained")
    def test_get_tokenizer(self, mock_func):
        mock_func.side_effect = mock_safe_get_tokenizer_from_pretrained
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = Deepseekv2Router(FAKE_MODEL_NAME_OR_PATH, config_dict)
        tokenizer = router.get_tokenizer()
        self.assertIsInstance(router, Deepseekv2Router)
        self.assertEqual(router.config_dict, FAKE_CONFIG_DICT)
        self.assertEqual(router.config_dict['model_type'], 'deepseek_v3')
        self.assertEqual(router.config_dict['num_hidden_layers'], NUM_HIDDEN_LAYERS)
        self.assertFalse(tokenizer.use_fast)
        self.assertFalse(tokenizer.trust_remote_code)

    def test_get_llm_config_validators(self):
        deepseekv2_router = Deepseekv2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=FAKE_CONFIG_DICT
        )

        llm_config_validators = deepseekv2_router.get_llm_config_validators()

        self.assertIn("llm", llm_config_validators)
        llm_validator = llm_config_validators["llm"]
        self.assertIsInstance(llm_validator, DictionaryParameterValidator)
        self.assertIn("models", llm_config_validators)
        deepseekv2_config_validator = llm_config_validators["models"]["deepseekv2"]
        self.assertIsInstance(deepseekv2_config_validator, DictionaryParameterValidator)
    
    def test_get_reasoning_parser_support_thinking(self):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        config_dict["model_type"] = "deepseekv2"
        router = Deepseekv2Router("", config_dict)
        self.assertFalse(router.config.is_reasoning_model)

        router._tokenizer = MockTokenizer(False, False)
        _ = router.reasoning_parser
        self.assertTrue(router.config.is_reasoning_model)
        self.assertEqual(router.config.reasoning_config.start_reasoning_token_id, 123456)
        self.assertEqual(router.config.reasoning_config.end_reasoning_token_id, 123457)
    
    def test_get_reasoning_parser_non_thinking(self):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        config_dict["model_type"] = "deepseekv2"
        router = Deepseekv2Router("", config_dict)
        self.assertFalse(router.config.is_reasoning_model)

        router._tokenizer = MockTokenizerNonThink(False, False)
        _ = router.reasoning_parser
        self.assertFalse(router.config.is_reasoning_model)


if __name__ == '__main__':
    unittest.main()