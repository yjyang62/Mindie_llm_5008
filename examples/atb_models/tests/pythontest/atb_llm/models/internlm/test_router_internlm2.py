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
from atb_llm.models.internlm2.router_internlm2 import Internlm2Router

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
NUM_HIDDEN_LAYERS = 2
FAKE_CONFIG_DICT = {
    'model_type': 'internlm2',
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'max_position_embeddings': 2048,
    'vocab_size': 30000
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
class TestInternlm2Router(unittest.TestCase):
    @patch("atb_llm.models.base.router.BaseRouter.check_config", return_value=None)
    def test_get_config(self, _1):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = Internlm2Router("", config_dict)
        config = router.get_config()
        self.assertIsNotNone(config)

    @unpack
    @patch("atb_llm.models.internlm2.router_internlm2.safe_get_tokenizer_from_pretrained")
    def test_get_tokenizer(self, mock_func):
        mock_func.side_effect = mock_safe_get_tokenizer_from_pretrained
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = Internlm2Router(FAKE_MODEL_NAME_OR_PATH, config_dict)
        tokenizer = router.get_tokenizer()
        self.assertIsInstance(router, Internlm2Router)
        self.assertEqual(router.config_dict, FAKE_CONFIG_DICT)
        self.assertEqual(router.config_dict['model_type'], 'internlm2')
        self.assertEqual(router.config_dict['num_hidden_layers'], NUM_HIDDEN_LAYERS)
        self.assertFalse(tokenizer.use_fast)
        self.assertFalse(tokenizer.trust_remote_code)


if __name__ == '__main__':
    unittest.main()