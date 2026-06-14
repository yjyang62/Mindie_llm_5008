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
from unittest.mock import patch
from ddt import ddt

import torch

from atb_llm.utils.layers.linear import W4A8LinearDynamic


@ddt
class TestW4A8LinearDynamic(unittest.TestCase):
    def test_init_pergroup(self):
        weight = torch.ones(256, 256, 7168, dtype=torch.int8)
        scale = torch.ones(256, 512, 1, dtype=torch.float16)
        second_scale = torch.ones(256, 512, 28, dtype=torch.float16)
        bias = torch.ones(256, 512, 1, dtype=torch.float16)
        linear = W4A8LinearDynamic(weight, scale, second_scale, bias)
        self.assertEqual(linear.weight.shape, (256, 7168, 256))
    
    def test_init_perchannel(self):
        weight = torch.ones(256, 256, 7168, dtype=torch.int8)
        scale = torch.ones(256, 512, 1, dtype=torch.float16)
        bias = torch.ones(256, 512, 1, dtype=torch.float16)
        linear = W4A8LinearDynamic(weight, scale, None, bias)
        self.assertEqual(linear.weight.shape, (256, 7168, 256))

    def test_init_dense(self):
        K = 5120
        N = 8192
        weight = torch.ones(N, K//2, dtype=torch.int8)
        scale = torch.ones(N, 1, dtype=torch.float16)
        second_scale = torch.ones(N, 20, dtype=torch.float16)
        bias = torch.ones(N, K, dtype=torch.float16)
        with patch('atb_llm.utils.layers.linear.W4A8LinearDynamic.quant_version', '1.0.0'):
            linear = W4A8LinearDynamic(weight, scale, second_scale, bias)
        self.assertEqual(linear.weight.shape, (5120, 1024))


if __name__ == '__main__':
    unittest.main()