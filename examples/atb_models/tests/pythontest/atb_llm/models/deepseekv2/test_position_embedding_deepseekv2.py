#!/usr/bin/env python
# coding=utf-8
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
from unittest import TestCase
import torch
from atb_llm.utils.layers.embedding.position_yarn_embedding import PositionYarnEmbedding


class TestPositionEmbeddingDeepseekV2(TestCase):

    def test_update_cos_sin_cache_total(self):
        deepseekv2_rope = PositionYarnEmbedding(inv_freq=torch.arange(10), dim=1, base=2, scaling_factor=1.0,
                 max_position_embeddings=2048, original_max_position_embeddings=4096,
                 beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0)
        deepseekv2_rope.update_cos_sin_cache_total(torch.float32, torch.device("cpu"), 100)

    def test_static_yarn(self):
        deepseekv2_rope = PositionYarnEmbedding(inv_freq=torch.arange(10), dim=1, base=2, scaling_factor=1.0,
                 max_position_embeddings=2048, original_max_position_embeddings=4096,
                 beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0)
        deepseekv2_static_input = PositionYarnEmbedding.StaticInputArgs(scaling_factor=1.0,
                    max_position_embeddings=2048, original_max_position_embeddings=4096,
                    beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0)
        deepseekv2_rope.static_yarn(1, 2, torch.device("cpu"), deepseekv2_static_input)

if __name__ == '__main__':
    unittest.main()