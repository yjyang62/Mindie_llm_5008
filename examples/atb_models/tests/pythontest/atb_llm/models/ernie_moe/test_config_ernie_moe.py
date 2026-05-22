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
from atb_llm.models.ernie_moe.config_ernie_moe import ErniemoeConfig

FAKE_CONFIG_DICT = {
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "intermediate_size": 12288,
    "max_position_embeddings": 131072,
    "moe_intermediate_size": 1536,
    "moe_k": 6,
    "moe_layer_end_index": 27,
    "moe_layer_interval": 1,
    "moe_layer_start_index": 1,
    "moe_num_experts": 64,
    "moe_num_shared_experts": 2,
    "num_attention_heads": 20,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "vocab_size": 103424
}


@ddt
class TestDeepseekV2Config(unittest.TestCase):
    def setUp(self):
        self.ernie_moe_config = ErniemoeConfig(**FAKE_CONFIG_DICT)
    
    def test_init(self):
        self.assertIn(self.ernie_moe_config.hidden_size, [2560, 8192])
        self.assertEqual(self.ernie_moe_config.hidden_act, "silu")
        self.assertEqual(self.ernie_moe_config.max_position_embeddings, 131072)
        self.assertIn(self.ernie_moe_config.moe_intermediate_size, [1536, 3584])
        self.assertIn(self.ernie_moe_config.moe_k, [6, 8])
        self.assertEqual(self.ernie_moe_config.moe_layer_interval, 1)
        self.assertIn(self.ernie_moe_config.moe_layer_start_index, [1, 3])
        self.assertIn(self.ernie_moe_config.moe_layer_end_index, [27, 53])
        self.assertEqual(self.ernie_moe_config.moe_num_experts, 64)
        self.assertIn(self.ernie_moe_config.moe_num_shared_experts, [0, 2])
        self.assertEqual(self.ernie_moe_config.vocab_size, 103424)
        self.assertEqual(self.ernie_moe_config.pad_token_id, 0)
        self.assertEqual(self.ernie_moe_config.bos_token_id, 1)
        self.assertEqual(self.ernie_moe_config.eos_token_id, 2)
        self.assertEqual(self.ernie_moe_config.rope_theta, 500000.0)
        self.assertIn(self.ernie_moe_config.num_hidden_layers, [28, 54])
        self.assertIn(self.ernie_moe_config.num_attention_heads, [20, 64])
        self.assertIn(self.ernie_moe_config.num_key_value_heads, [4, 8])