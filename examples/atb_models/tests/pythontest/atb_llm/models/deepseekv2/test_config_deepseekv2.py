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
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config

FAKE_CONFIG_DICT = {
    'max_position_embeddings': 16384,
    'vocab_size': 10240,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "tie_word_embeddings": False,
    "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
            "parallel_embedding": True
            }
}


@ddt
class TestDeepseekV2Config(unittest.TestCase):
    def setUp(self):
        self.deepseekv2_config = DeepseekV2Config(**FAKE_CONFIG_DICT)
    
    def test_init(self):
        self.assertEqual(self.deepseekv2_config.hidden_size, 5120)
        self.assertEqual(self.deepseekv2_config.model_type, 'deepseekv2')
        self.assertFalse(self.deepseekv2_config.tie_word_embeddings)
        self.assertFalse(self.deepseekv2_config.norm_topk_prob)