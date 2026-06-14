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
from ddt import ddt, data, unpack
from atb_llm.models.llama.config_llama import LlamaConfig

FAKE_CONFIG_DICT = {
    'max_position_embeddings': 2048,
    'vocab_size': 65536
}
FAKE_STR = "fakestr"
ALIBI_BIAS_MAX = 'alibi_bias_max'
PE_TYPE = 'pe_type'
ROPE_GIVEN_INV_FEQ_STR = 'rope_given_inv_feq_str'
ROPE_KEEP_LOCAL_BASE_WINDOWS = 'rope_keep_local_base_windows'
ROPE_MSCALE = 'rope_mscale'
ROPE_VANILLA_THETA = 'rope_vanilla_theta'


@ddt
class TestLlamaConfig(unittest.TestCase):
    def setUp(self):
        self.llama_config = LlamaConfig(**FAKE_CONFIG_DICT)
    
    def test_init(self):
        self.assertEqual(self.llama_config.attribute_map['max_sequence_length'], 'max_position_embeddings')
        self.assertEqual(self.llama_config.model_type, 'llama')
        self.assertFalse(self.llama_config.tie_word_embeddings)
        self.assertEqual(self.llama_config.num_key_value_heads, 32)
    
    @data((ALIBI_BIAS_MAX, False), (ALIBI_BIAS_MAX, 0.0), (PE_TYPE, FAKE_STR),
          (ROPE_GIVEN_INV_FEQ_STR, FAKE_STR), (ROPE_KEEP_LOCAL_BASE_WINDOWS, FAKE_STR),
          (ROPE_MSCALE, FAKE_STR), (ROPE_MSCALE, 0), (ROPE_VANILLA_THETA, FAKE_STR),
          (ROPE_VANILLA_THETA, 0.0))
    @unpack
    def test_validate_given_invalid_value_raise_value_error(self, field_name, invalid_value):
        setattr(self.llama_config, field_name, invalid_value)
        with self.assertRaises(ValueError):
            self.llama_config.validate()