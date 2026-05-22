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
from ddt import ddt

import torch

from atb_llm.utils.layers.linear import W4A16LinearStatic


@ddt
class TestW4A16LinearStatic(unittest.TestCase):
    def test_init_pergroup(self):
        weight = torch.ones(64, 512, 128, dtype=torch.int8)
        scale = torch.ones(64, 256, 32, dtype=torch.float16)
        offset = torch.ones(64, 256, 32, dtype=torch.float16)
        linear = W4A16LinearStatic(weight, scale, offset)
        self.assertEqual(linear.weight.shape, (64, 128, 256))

    def test_init_perchannel_bias(self):
        weight = torch.ones(64, 512, 128, dtype=torch.int8)
        scale = torch.ones(64, 256, 1, dtype=torch.float16)
        offset = torch.ones(64, 256, 1, dtype=torch.float16)
        bias = torch.ones(64, 512, 128, dtype=torch.float16)
        linear = W4A16LinearStatic(weight, scale, offset, bias)
        self.assertEqual(linear.weight.shape, (64, 512, 64))
        self.assertTrue(linear.has_bias)

    def test_init_2d_pergroup(self):
        weight = torch.ones(512, 128, dtype=torch.int8)
        scale = torch.ones(256, 32, dtype=torch.float16)
        offset = torch.ones(256, 32, dtype=torch.float16)
        linear = W4A16LinearStatic(weight, scale, offset)
        self.assertEqual(linear.weight.shape, (128, 256))


if __name__ == '__main__':
    unittest.main()