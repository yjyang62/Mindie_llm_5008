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
from ddt import ddt, data
import torch

from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.utils.layers.attention.process_mla_linear import (preprocess_kv_weights, preprocess_linear_for_rope,
                                                                      transdata_3d)
from atb_llm.utils.quantize.quant_type import QuantType


@ddt
class TestProcess(unittest.TestCase):
    def setUp(self):
        self.config = DeepseekV2Config.from_dict({"qk_nope_head_dim": 128,
                                                "qk_rope_head_dim": 64,
                                                "kv_lora_rank": 512,
                                                "rope_scaling": {},
                                                "v_head_dim": 128})

    @data("projq", "projk", "projv", "haha")
    def test_preprocess_kv_weights(self, proj_name):
        weight = torch.ones(self.config.tp_num_key_value_heads * \
                            (self.config.qk_nope_head_dim + self.config.v_head_dim), self.config.kv_lora_rank)
        try:
            new_weight = preprocess_kv_weights(weight, self.config, proj_name)
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
        if proj_name == "projk":
            golden_weight = torch.ones(self.config.tp_num_key_value_heads,
                                       self.config.qk_nope_head_dim,
                                       self.config.kv_lora_rank)
            self.assertTrue(torch.allclose(golden_weight, new_weight))
        elif proj_name == "projv":
            golden_weight = torch.ones(self.config.tp_num_key_value_heads,
                                       self.config.kv_lora_rank,
                                       self.config.v_head_dim)
            self.assertTrue(torch.allclose(golden_weight, new_weight))

    @data("projq", "projk", "projv", "haha")
    def test_process_linear_for_rope(self, proj_name):
        if proj_name == "projq":
            first_size = self.config.tp_num_attention_heads * self.config.q_head_dim_before
        elif proj_name == "projk":
            first_size = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        else:
            first_size = 1
        weight = [torch.ones(first_size, self.config.hidden_size)] * 3
        try:
            preprocess_linear_for_rope(weight, self.config, proj_name)
        except ValueError as e:
            self.assertEqual(type(e), ValueError)

    @data(QuantType.W8A8_DYNAMIC, QuantType.W8A8)
    def test_process_linear_for_rope_quant(self, quant_type):
        first_size = self.config.tp_num_attention_heads * self.config.q_head_dim_before
        weight = [torch.ones(first_size, self.config.hidden_size, dtype=torch.int8)] * 3
        self.config.quantize = quant_type
        preprocess_linear_for_rope(weight, self.config, "projq")
    
    def test_transdata_3d(self):
        test_input = torch.randn((128, 128, 512))
        golden_shape = (128, 32, 128, 16)
        test_output = transdata_3d(test_input)
        self.assertEqual(golden_shape, tuple(test_output.shape))

if __name__ == '__main__':
    unittest.main()