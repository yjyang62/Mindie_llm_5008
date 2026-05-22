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
from atb_llm.models.glm4_moe.config_glm4_moe import Glm4moeConfig

FAKE_CONFIG_DICT = {
    'max_position_embeddings': 769,
    'norm_topk_prob': True,
    'rope_scaling': None,
    'seq_aux': False,
    'vocab_size': 1024
}


@ddt
class TestGlm4moeConfig(unittest.TestCase):
    def setUp(self):
        self.glm4_moeconfig = Glm4moeConfig(**FAKE_CONFIG_DICT)
    
    def test_init(self):
        self.assertEqual(self.glm4_moeconfig.num_hidden_layers, 46)
        self.assertTrue(self.glm4_moeconfig.norm_topk_prob)
        self.assertFalse(self.glm4_moeconfig.seq_aux)
    