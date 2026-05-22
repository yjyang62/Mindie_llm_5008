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
from unittest.mock import MagicMock
from unittest import TestCase
from ddt import ddt, data
import torch

from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.models.ernie_moe.config_ernie_moe import ErniemoeConfig
from atb_llm.models.ernie_moe.modeling_ernie_moe import (
    ErniemoeMLP,
    ErniemoeMoE,
    FlashErniemoeAttention,
    FlashErniemoeLayer,
    FlashErniemoeModel
)


@ddt
class TestFlashErniemoeModel(TestCase):
    def setUp(self):
        config_dict = {
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
            "vocab_size": 103424,
            "quantize": "w8a8_dynamic"
        }

        self.config = ErniemoeConfig.from_dict(config_dict)
        self.config.parallel_embedding = True
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.bfloat16
        self.weights.quantize = None
        self.weights.get_tensor.return_value = torch.empty(256, 64, dtype=torch.bfloat16)
        self.weights.get_multi_weights_col.return_value = torch.empty(256, 64, dtype=torch.bfloat16)
        self.weights.get_multi_weights_row.return_value = torch.empty(256, 64, dtype=torch.bfloat16)
        self.weights.get_whole_tensor.return_value = torch.empty(256, 64, dtype=torch.bfloat16)
        self.weights.get_partial_sharded.return_value = torch.empty(256, 64, dtype=torch.bfloat16)
        self.weights.get_shape.return_value = [256, 64]
        self.weights.process_group.size.return_value = 1
        self.weights.process_group.rank.return_value = 0
        self.model_config = MagicMock()
        self.model_config.ep_level = 1

    @data(False,)
    def test_erniemoe_mlp(self, sharded):
        self.weights.sharded = sharded
        erniemoe_mlp = ErniemoeMLP("mlp", self.config, self.weights)
        self.assertEqual(PackType.ALL_FP, erniemoe_mlp.pack_type)

    @data(False,)
    def test_erniemoe_moe(self, sharded):
        self.weights.sharded = sharded
        ErniemoeMoE("moe", self.config, self.weights, ErniemoeMLP, self.model_config)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    @data(False,)
    def test_flash_erniemoe_attention(self, sharded):
        self.weights.sharded = sharded
        self.config.attn_quantize = "w8a8"
        flash_erniemoe_attention = FlashErniemoeAttention("attn", self.config, self.weights)
        self.assertEqual(PackType.ALL_W8A8, flash_erniemoe_attention.pack_type)

    @data(False,)
    def test_flash_erniemoe_layer(self, sharded):
        self.weights.sharded = sharded
        FlashErniemoeLayer(0, self.config, self.weights, model_config=self.model_config)
        FlashErniemoeLayer(3, self.config, self.weights, model_config=self.model_config)
        self.weights.get_tensor.assert_any_call("model.layers.0.input_layernorm.weight")

    @data(False,)
    def test_flash_erniemoe_model(self, sharded):
        self.weights.sharded = sharded
        FlashErniemoeModel(self.config, self.weights, model_config=self.model_config)
        self.weights.get_tensor.assert_any_call("model.layers.0.input_layernorm.weight")


if __name__ == '__main__':
    unittest.main()