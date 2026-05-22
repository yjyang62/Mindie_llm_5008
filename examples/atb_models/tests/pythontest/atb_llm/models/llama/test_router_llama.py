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
from unittest.mock import patch
import copy
from ddt import ddt, data, unpack
from atb_llm.models.llama.router_llama import LlamaRouter
from atb_llm.models.llama.config_llama import LlamaConfig

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
LLAMA_33B_NUM_HIDDEN_LAYERS = 60
FAKE_CONFIG_DICT = {
    "model_type": "llama",
    'max_position_embeddings': 2048,
    'vocab_size': 65536
}


class MockTokenizer:
    def __init__(self, use_fast):
        self.use_fast = use_fast
        self.pad_token_id = None


def mock_safe_get_tokenizer_from_pretrained(_, **kwargs):
    use_fast = kwargs.get("use_fast")
    return MockTokenizer(use_fast)


@ddt
class TestLlamaRouter(unittest.TestCase):
    @patch("atb_llm.models.base.router.BaseRouter.check_config", return_value=None)
    def test_get_config(self, _1):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = LlamaRouter("", config_dict)
        config = router.get_config()
        self.assertIsInstance(config, LlamaConfig)

    @data((0, True), (LLAMA_33B_NUM_HIDDEN_LAYERS, False))
    @unpack
    @patch("atb_llm.models.llama.router_llama.safe_get_tokenizer_from_pretrained")
    def test_get_tokenizer(self, num_hidden_layers, use_fast_value, mock_func):
        mock_func.side_effect = mock_safe_get_tokenizer_from_pretrained
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        config_dict['num_hidden_layers'] = num_hidden_layers
        router = LlamaRouter(FAKE_MODEL_NAME_OR_PATH, config_dict)
        tokenizer = router.get_tokenizer()
        self.assertEqual(tokenizer.use_fast, use_fast_value)
        self.assertEqual(tokenizer.pad_token_id, 0)
