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

from atb_llm.models.internlm2.config_internlm2 import Internlm2Config
from atb_llm.models.internlm2.modeling_internlm2 import (
    Internlm2MLP, FlashInternlm2Attention, FlashInternlm2Layer, FlashInternlm2Model
)
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    RMSNorm,
)

FAKE_CONFIG_DICT = {
    'model_type': 'internlm2',
    'num_hidden_layers': 2,
    'max_position_embeddings': 2048,
    'vocab_size': 65536
}


@ddt
class TestFlashInternlm2Model(unittest.TestCase):
    def setUp(self):
        self.config = Internlm2Config.from_dict(FAKE_CONFIG_DICT)
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

    def test_internlm2_mlp(self):
        feed_forward = Internlm2MLP("feed_forward", self.config, self.weights)
        self.assertIsInstance(feed_forward, Internlm2MLP)
        self.assertIsInstance(feed_forward.w1_w3, TensorParallelColumnLinear)
        self.assertIsInstance(feed_forward.w2, TensorParallelRowLinear)

    def test_flash_internlm2_attention(self):
        attention = FlashInternlm2Attention("attention", self.config, self.weights)
        self.assertIsInstance(attention, FlashInternlm2Attention)
        self.assertIsInstance(attention.wqkv, TensorParallelColumnLinear)
        self.assertIsInstance(attention.wo, TensorParallelRowLinear)

    def test_flash_internlm2_layer(self):
        layer = FlashInternlm2Layer(0, self.config, self.weights)
        self.assertIsInstance(layer, FlashInternlm2Layer)
        self.assertIsInstance(layer.attention, FlashInternlm2Attention)
        self.assertIsInstance(layer.feed_forward, Internlm2MLP)

    def test_flash_internlm2_model(self):
        model = FlashInternlm2Model(self.config, self.weights)
        self.assertIsInstance(model, FlashInternlm2Model)
        self.assertIsInstance(model.tok_embeddings, TensorEmbedding)
        self.assertIsInstance(model.layers, nn.ModuleList)
        self.assertIsInstance(model.norm, RMSNorm)


if __name__ == '__main__':
    unittest.main()