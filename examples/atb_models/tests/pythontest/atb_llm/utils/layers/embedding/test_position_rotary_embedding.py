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
import torch

from atb_llm.models.llama.config_llama import LlamaConfig
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionRotaryEmbedding


class TestPositionRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.base = 10000.0
        self.device = 'cpu'
        self.scaling_factor = 1.0
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim).double())
        self.embedding = PositionRotaryEmbedding(self.inv_freq, self.scaling_factor, self.base)
        
        self.embedding.position_ids_offset = [0]

    def test_static_method(self):
        embed = PositionRotaryEmbedding.static(self.dim, self.base, self.device)
        self.assertIsInstance(embed, PositionRotaryEmbedding)

    def test_load_method(self):
        mock_weights = MagicMock()
        mock_weights.get_tensor.return_value = self.inv_freq
        embed = PositionRotaryEmbedding.load('prefix', mock_weights)
        self.assertIsInstance(embed, PositionRotaryEmbedding)

    def test_update_cos_sin_cache(self):
        seqlen = 10
        self.embedding.update_cos_sin_cache(torch.float16, self.device, seqlen)
        position_ids = torch.arange(10, dtype=torch.int64)
        cos, sin = self.embedding.get_cos_sin(position_ids, 10, torch.float16)
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)
    
    def test_update_cos_sin_cache_total(self):
        seqlen = 10
        self.embedding.update_cos_sin_cache_total(torch.float16, self.device, seqlen)
        cos = self.embedding.get_cos_cached_total()
        sin = self.embedding.get_sin_cached_total()
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)

    def test_update_cos_sin_cache_value_error(self):
        self.embedding.scaling_factor = 0.0
        with self.assertRaises(ValueError):
            self.embedding.update_cos_sin_cache(torch.float32, self.device, 10)

    def test_get_cos_sin(self):
        self.embedding.update_cos_sin_cache(torch.float32, self.device, 10)
        position_ids = torch.arange(10, dtype=torch.int64)
        cos, sin = self.embedding.get_cos_sin(position_ids, 10, torch.float32)
        self.assertEqual(cos.shape, (10, 1, self.dim // 2))
        self.assertEqual(sin.shape, (10, 1, self.dim // 2))

    def test_clear_ntk_cache(self):
        self.embedding.clear_ntk_cache(batch_size=5)
        self.assertEqual(self.embedding.position_ids_offset, [0] * 5)

    def test_dynamic_ntk_inv_freq(self):
        config = MagicMock()
        config.rotary_emb_base = 10000.0
        config.hidden_size = 128
        config.num_attention_heads = 4
        seq_len = 32
        ntk_alpha = 1.0
        self.embedding.clear_ntk_cache(1)
        self.embedding.dynamic_ntk_inv_freq(config, seq_len, self.device, ntk_alpha, 0)

        self.assertGreater(len(self.embedding.ntk_inv_freqs), 0)

    def test_update_llama3_cos_sin_cache_total(self):
        config = LlamaConfig(
            rope_scaling={"factor": 8.0,
                          "low_freq_factor": 1.0,
                          "high_freq_factor": 4.0,
                          "original_max_position_embeddings": 8192,
                          "rope_type": "llama3"
                         }
        )
        self.embedding.update_llama3_cos_sin_cache_total(config, torch.float16, self.device, 10)
        cos = self.embedding.get_cos_cached_total()
        sin = self.embedding.get_sin_cached_total()
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)

if __name__ == '__main__':
    unittest.main()