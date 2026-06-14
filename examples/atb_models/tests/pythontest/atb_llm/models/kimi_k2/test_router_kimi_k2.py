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
from ddt import ddt, unpack, data
from atb_llm.models.kimi_k2.router_kimi_k2 import Kimik2Router
from atb_llm.models.kimi_k2.config_kimi_k2 import KimiK2Config

NUM_HIDDEN_LAYERS = 61

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    'model_type': 'kimi_k2',
    'num_hidden_layers': NUM_HIDDEN_LAYERS,
    'max_position_embeddings': 131072,
    'vocab_size': 163840,
    'rope_scaling': None,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'topk_method': "noaux_tc",
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "n_routed_experts": 384,
    "q_lora_rank": 1536,
}

FAKE_CONFIG_DICT_1 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_1["n_routed_experts"] = 1
FAKE_CONFIG_DICT_2 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_2["n_routed_experts"] = 4
FAKE_CONFIG_DICT_3 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_3["num_hidden_layers"] = 0
FAKE_CONFIG_DICT_4 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_4["topk_method"] = "fake"
FAKE_CONFIG_DICT_5 = copy.deepcopy(FAKE_CONFIG_DICT)
FAKE_CONFIG_DICT_5["topk_method"] = "greedy"
FAKE_CONFIG_DICT_5["topk_group"] = 1
FAKE_CONFIG_DICT_5["n_group"] = 2

fake_groups = [
    (FAKE_CONFIG_DICT_1,), (FAKE_CONFIG_DICT_2,), (FAKE_CONFIG_DICT_3,), (FAKE_CONFIG_DICT_4,), (FAKE_CONFIG_DICT_5,)
]


class FakeTokenizer:
    def __init__(self):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        
    def decode(self, token_ids, **kwargs):
        return "decode"


FAKE_TOKENIZER = FakeTokenizer()


@ddt
class TestKimik2Router(unittest.TestCase):
    @patch("atb_llm.models.base.router.BaseRouter.check_config", return_value=None)
    def test_get_config(self, _1):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = Kimik2Router("", config_dict)
        config = router.get_config()
        self.assertIsNotNone(config)

    @data(*fake_groups)
    @unpack
    def test_check_config_kimi_k2(self, config_dict):
        router = Kimik2Router("", config_dict)
        config = KimiK2Config.from_dict(config_dict)
        with self.assertRaises(ValueError):
            router.check_config_kimi_k2(config)

    @unpack
    @patch("atb_llm.models.kimi_k2.router_kimi_k2.safe_get_tokenizer_from_pretrained", return_value=FAKE_TOKENIZER)
    def test_get_tokenizer(self, mock_func):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = Kimik2Router(FAKE_MODEL_NAME_OR_PATH, config_dict)
        tokenizer = router.get_tokenizer()
        self.assertIsInstance(router, Kimik2Router)
        self.assertEqual(router.config_dict, FAKE_CONFIG_DICT)
        self.assertEqual(router.config_dict['model_type'], 'kimi_k2')
        self.assertEqual(router.config_dict['num_hidden_layers'], NUM_HIDDEN_LAYERS)
        self.assertEqual(tokenizer.decode([1,2], skip_special_tokens=True), "decode")


if __name__ == '__main__':
    unittest.main()