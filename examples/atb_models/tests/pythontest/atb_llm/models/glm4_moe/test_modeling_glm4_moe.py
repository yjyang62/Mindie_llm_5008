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
from unittest.mock import MagicMock, patch
from unittest import TestCase
from ddt import ddt, data
import torch


from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.models.glm4_moe.config_glm4_moe import Glm4moeConfig
from atb_llm.models.glm4_moe.modeling_glm4_moe import (
    Glm4moeMLP,
    Glm4moeMoe,
    FlashGlm4moeAttention,
    FlashGlm4moeLayer,
    FlashGlm4moeModel
)

FAKE_CONFIG_DICT = {
    'max_position_embeddings': 769,
    'use_qk_norm': True,
    'norm_topk_prob': True,
    'rope_scaling': None,
    'seq_aux': False,
    'vocab_size': 1024,
    'attn_quantize': 'w8a8',
    'moe_quantize': 'w8a8'
}


@ddt
class TestFlashGlm4moeModel(TestCase):
    def setUp(self):
        self.config = Glm4moeConfig(**FAKE_CONFIG_DICT)
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.quantize = None
        self.weights.get_tensor.return_value = torch.empty(200, 200, dtype=torch.bfloat16)
        self.weights.get_multi_weights_col.return_value = torch.empty(200, 200, dtype=torch.bfloat16)
        self.weights.get_replicated_weights.return_value = torch.empty(200, 200, dtype=torch.bfloat16)
        self.weights.get_multi_weights_row.return_value = torch.empty(200, 200, dtype=torch.bfloat16)
        self.weights.get_whole_tensor.return_value = torch.empty(200, 200, dtype=torch.bfloat16)
        self.weights.get_shape.return_value = [200, 200]
        self.weights.get_sharded.return_value = torch.empty(200, 200, dtype=torch.bfloat16)
        self.weights.get_partial_sharded.return_value = torch.empty(200, 200, dtype=torch.bfloat16)
        self.weights.switch_process_group.return_value = None
        self.weights.process_group.rank.return_value = 8
        self.weights.process_group.size.return_value = 8

    def test_glm4_moemlp(self):
        glm4_moe_mlp = Glm4moeMLP("mlp", self.config, self.weights)
        self.assertEqual(PackType.ALL_FP, glm4_moe_mlp.pack_type)

    @data(
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 2)
    )
    def test_flashglm4_moeattention(self, params):
        num_key_value_heads, num_attention_heads, process_group_size = params
        self.config.num_key_value_heads = num_key_value_heads
        self.config.num_attention_heads = num_attention_heads
        self.weights.process_group.size.return_value = process_group_size
        if num_attention_heads % process_group_size != 0:
            with self.assertRaises(ValueError) as exc_info:
                flash_glm4_moe_attention = FlashGlm4moeAttention("attn", self.config, self.weights)
            expected_msg = f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {num_attention_heads} " \
                           f"and `num_shards`: {process_group_size}"
            assert expected_msg in str(str(exc_info.exception))
        else:
            flash_glm4_moe_attention = FlashGlm4moeAttention("attn", self.config, self.weights)
            self.assertEqual(PackType.ALL_FP, flash_glm4_moe_attention.pack_type)

    @data(PackType.ALL_FP, PackType.ALL_W8A8_ANTI, PackType.ALL_W8A8)
    @patch("atb_llm.models.glm4_moe.modeling_glm4_moe.calc_linear_pack_type")
    def test_flashglm4_moelayer(self, param, mock_calc):
        mock_calc.return_value = param
        FlashGlm4moeLayer(0, self.config, self.weights)
        FlashGlm4moeLayer(1, self.config, self.weights)
        self.weights.get_tensor.assert_any_call("model.layers.0.input_layernorm.weight")

    def test_flashglm4_moemodel(self):
        FlashGlm4moeModel(self.config, self.weights)
        self.weights.get_tensor.assert_any_call("model.layers.0.input_layernorm.weight")

    @data(
        (True, False),
        (False, True),
        (False, False)
    )
    def test_glm4_moemoe(self, params):
        has_moe_ep, has_moe_tp = params
        self.weights.mapping.has_moe_ep.return_value = has_moe_ep
        self.weights.mapping.has_moe_tp.return_value = has_moe_tp
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.mlp_tp.rank = 0
        Glm4moeMoe("moe", self.config, self.weights, Glm4moeMLP)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")


if __name__ == '__main__':
    unittest.main()