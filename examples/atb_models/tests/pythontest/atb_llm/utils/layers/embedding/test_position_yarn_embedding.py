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

from atb_llm.utils.layers.embedding.position_yarn_embedding import PositionYarnEmbedding


class StaticInputArgs:
    def __init__(self, scaling_factor=1.0, max_position_embeddings=2048, original_max_position_embeddings=4096,
                    beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0,):
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim


class TestPositionYarnEmbedding(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.base = 10000.0
        self.device = 'cpu'
        self.scaling_factor = 1.0
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim).double())
        self.embedding = PositionYarnEmbedding(self.inv_freq, self.scaling_factor, self.base)
        self.mscale = 1.0
        self.beta_fast = 32
        self.beta_slow = 1
        self.original_max_position_embeddings = 4096

    def test_yarn_find_correction_range(self):
        low, high = PositionYarnEmbedding.yarn_find_correction_range(self.beta_fast, self.beta_slow, 
                                                                 self.dim, self.base, 
                                                                 self.original_max_position_embeddings)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
    
    def test_yarn_find_correction_dim(self):
        low = PositionYarnEmbedding.yarn_find_correction_dim(self.beta_slow, 
                                                        self.dim, self.base, 
                                                        self.original_max_position_embeddings)
        self.assertIsNotNone(low)
    
    def test_yarn_linear_ramp_mask(self):
        inv_freq_mask = PositionYarnEmbedding.yarn_linear_ramp_mask(self.beta_slow, self.beta_fast, 
                                                        self.dim // 2)
        self.assertIsNotNone(inv_freq_mask)

    def test_yarn_get_mscale(self):
        mscale = PositionYarnEmbedding.yarn_get_mscale(self.scaling_factor, self.mscale)
        self.assertIsNotNone(mscale)

    def test_static_yarn(self):
        yarn_kwargs = StaticInputArgs()
        rotary_embedding = PositionYarnEmbedding.static_yarn(self.dim, self.base, "cpu", yarn_kwargs)
        self.assertIsInstance(rotary_embedding, PositionYarnEmbedding)

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


if __name__ == '__main__':
    unittest.main()