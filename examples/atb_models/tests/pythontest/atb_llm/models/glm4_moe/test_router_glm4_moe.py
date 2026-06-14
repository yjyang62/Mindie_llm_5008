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
from ddt import ddt, unpack
from atb_llm.models.glm4_moe.router_glm4_moe import Glm4moeRouter

NUM_HIDDEN_LAYERS = 28

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    'model_type': 'glm4_moe',
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'max_position_embeddings': 4096,
    'rope_scaling': None,
    'vocab_size': 102400,
}


class MockTokenizer:
    def __init__(self, use_fast, trust_remote_code):
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code


def mock_safe_get_tokenizer_from_pretrained(_, **kwargs):
    use_fast = kwargs.get("use_fast")
    trust_remote_code = kwargs.get("trust_remote_code")
    return MockTokenizer(use_fast, trust_remote_code)


@ddt
class TestGlm4moeRouter(unittest.TestCase):
    @patch("atb_llm.models.base.router.BaseRouter.check_config", return_value=None)
    def test_get_config(self, _1):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = Glm4moeRouter("", config_dict)
        config = router.get_config()
        self.assertIsNotNone(config)

    @unpack
    @patch("atb_llm.models.glm4_moe.router_glm4_moe.safe_get_tokenizer_from_pretrained")
    def test_get_tokenizer(self, mock_func):
        mock_func.side_effect = mock_safe_get_tokenizer_from_pretrained
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = Glm4moeRouter(FAKE_MODEL_NAME_OR_PATH, config_dict)
        tokenizer = router.get_tokenizer()
        self.assertIsInstance(router, Glm4moeRouter)
        self.assertEqual(router.config_dict, FAKE_CONFIG_DICT)
        self.assertEqual(router.config_dict['model_type'], 'glm4_moe')
        self.assertEqual(router.config_dict['num_hidden_layers'], NUM_HIDDEN_LAYERS)
        self.assertTrue(tokenizer.use_fast)
        self.assertFalse(tokenizer.trust_remote_code)


if __name__ == '__main__':
    unittest.main()