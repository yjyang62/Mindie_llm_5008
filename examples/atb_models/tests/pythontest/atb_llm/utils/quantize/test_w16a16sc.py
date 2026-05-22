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
import torch

from atb_llm.utils.layers.linear import W16A16SparseCompressedLinear


class TestW16A16SparseCompressedLinear(unittest.TestCase):
    def setUp(self):
        self.weight = torch.randn(10, 10)
        self.index = torch.randint(0, 10, (10,))
        self.quant_bias = torch.randn(10)

    def test_buffer_registration(self):
        linear = W16A16SparseCompressedLinear(self.weight, self.index, self.quant_bias)
        self.assertTrue(hasattr(linear, 'weight'))
        self.assertTrue(hasattr(linear, 'quant_bias'))
        self.assertTrue(hasattr(linear, 'index'))

    def test_has_bias_attribute(self):
        linear = W16A16SparseCompressedLinear(self.weight, self.index, self.quant_bias)
        self.assertTrue(linear.has_bias)

if __name__ == '__main__':
    unittest.main()