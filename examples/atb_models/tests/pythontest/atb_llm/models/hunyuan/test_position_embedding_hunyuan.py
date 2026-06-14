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
from atb_llm.models.hunyuan.position_embedding_hunyuan import HunyuanRotaryEmbedding


class TestPositionEmbeddingHunyuan(TestCase):

    def test_position_embedding_hunyuan(self):
        pe = HunyuanRotaryEmbedding.static(dim=1, base=2, device=torch.device("cpu"),
                                            scaling_factor=1.0, scaling_alpha=1.0)
        pe.update_cos_sin_cache_total(torch.float32, torch.device("cpu"), 100)


if __name__ == '__main__':
    unittest.main()