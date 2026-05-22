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
from ddt import ddt
import torch
from torch import nn

from atb_llm.models.internlm3.config_internlm3 import Internlm3Config
from atb_llm.models.internlm3.modeling_internlm3 import (
    Internlm3MLP, FlashInternlm3Attention, FlashInternlm3Layer, FlashInternlm3Model
)
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    RMSNorm,
)

FAKE_CONFIG_DICT = {
    'model_type': 'internlm3',
    'num_hidden_layers': 2,
    'max_position_embeddings': 2048,
    'vocab_size': 65536
}


@ddt
class TestFlashInternlm3Model(unittest.TestCase):
    def setUp(self):
        self.config = Internlm3Config.from_dict(FAKE_CONFIG_DICT)
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.dtype = torch.float16
        self.weights.sharded = False
        self.weights.dtype = self.dtype
        self.weights.quantize = None
        self.weights.get_tensor.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_multi_weights_col.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_replicated_weights.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_multi_weights_row.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_whole_tensor.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_shape.return_value = [100, 100]
        self.weights.switch_process_group.return_value = None

    def test_internlm3_mlp(self):
        mlp = Internlm3MLP("mlp", self.config, self.weights)
        self.assertIsInstance(mlp, Internlm3MLP)
        self.assertIsInstance(mlp.gate_up_proj, TensorParallelColumnLinear)
        self.assertIsInstance(mlp.down_proj, TensorParallelRowLinear)

    def test_flash_internlm3_attention(self):
        self_attn = FlashInternlm3Attention("self_attn", self.config, self.weights)
        self.assertIsInstance(self_attn, FlashInternlm3Attention)
        self.assertIsInstance(self_attn.query_key_value, TensorParallelColumnLinear)
        self.assertIsInstance(self_attn.o_proj, TensorParallelRowLinear)

    def test_flash_internlm3_layer(self):
        layer = FlashInternlm3Layer(0, self.config, self.weights)
        self.assertIsInstance(layer, FlashInternlm3Layer)
        self.assertIsInstance(layer.self_attn, FlashInternlm3Attention)
        self.assertIsInstance(layer.mlp, Internlm3MLP)

    def test_flash_internlm3_model(self):
        model = FlashInternlm3Model(self.config, self.weights)
        self.assertIsInstance(model, FlashInternlm3Model)
        self.assertIsInstance(model.embed_tokens, TensorEmbedding)
        self.assertIsInstance(model.layers, nn.ModuleList)
        self.assertIsInstance(model.norm, RMSNorm)


if __name__ == '__main__':
    unittest.main()