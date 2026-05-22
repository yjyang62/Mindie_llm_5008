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
from ddt import ddt, data, unpack
from atb_llm.models.ernie_moe.router_ernie_moe import ErniemoeRouter

FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    "model_type": "ernie_moe",
    "hidden_act": "silu",
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "max_position_embeddings": 131072,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "num_hidden_layers": 54,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "rms_norm_eps": 1e-05,
    "use_cache": False,
    "vocab_size": 103424,
    "rope_theta": 500000.0,
    "moe_num_experts": 64,
    "moe_num_shared_experts": 0,
    "moe_layer_start_index": 3,
    "moe_layer_end_index": 53,
    "moe_intermediate_size": 3584,
    "moe_gate": "topk_fused",
    "moe_k": 8,
    "moe_layer_interval": 1,
    "tie_word_embeddings": True
}


@ddt
class TestErniemoeRouter(unittest.TestCase):
    @patch("atb_llm.models.base.router.BaseRouter.check_config", return_value=None)
    def test_get_config(self, _1):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = ErniemoeRouter(FAKE_MODEL_NAME_OR_PATH, config_dict)
        config = router.get_config()
        self.assertIsNotNone(config)
        self.assertIsInstance(router, ErniemoeRouter)
        self.assertEqual(router.config_dict, FAKE_CONFIG_DICT)
        self.assertEqual(router.config_dict['model_type'], "ernie_moe")
        self.assertEqual(router.config_dict['num_hidden_layers'], 54)

    @data(
        ({"moe_k": 65}, ValueError),
        ({"moe_num_experts": 7}, ValueError),
        ({"moe_layer_start_index": 5, "num_hidden_layers": 3}, ValueError)
    )
    @unpack
    def test_check_config(self, invalid_params, expect_exc):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        router = ErniemoeRouter(FAKE_MODEL_NAME_OR_PATH, config_dict)
        config = router.get_config()
        for k, v in invalid_params.items():
            setattr(config, k, v)
        with self.assertRaises(expect_exc):
            router.check_config_ernie(config)


if __name__ == '__main__':
    unittest.main()