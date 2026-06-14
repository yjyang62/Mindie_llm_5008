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
from unittest.mock import MagicMock, call
from ddt import ddt

import torch

from atb_llm.utils.layers.linear import FastLinear


@ddt
class TestFastLinear(unittest.TestCase):
    def test_init(self):
        weight = torch.ones(64, 512, dtype=torch.float16)
        bias = torch.ones(512, dtype=torch.float16)
        linear = FastLinear(weight, None, is_norm=True)
        self.assertTrue(linear.is_norm_head)
        linear = FastLinear(weight, bias, nd_weight=True)
        self.assertTrue(linear.has_bias)
        self.assertTrue(linear.nd_weight)

    def test_load(self):
        weights = MagicMock()
        weights.get_tensor.return_value = torch.ones([64, 512], dtype=torch.float16)
        weights.sharded = False
        linear = FastLinear.load(prefix="fake", weights=weights, bias=False)
        self.assertIsNone(linear.bias)

        linear = FastLinear.load(prefix="fake", weights=weights, bias=True, bias_name="fake_bias")
        weights.assert_has_calls([
            call.get_tensor("fake.weight"), call.get_tensor("fake.fake_bias")
        ])

        weights.sharded = True
        linear = FastLinear.load(prefix="fake", weights=weights, bias=True, bias_name="fake_bias", module_name="module")
        weights.assert_has_calls([
            call.get_tensor("module.weight"), call.get_tensor("module.bias")
        ])

    def test_get_weights(self):
        weight = torch.ones(64, 512, dtype=torch.float16)
        bias = torch.ones(512, dtype=torch.float16)
        linear = FastLinear(weight, bias)
        out = linear.get_weights("fake")
        self.assertIn("fake.weight", out)
        self.assertIn("fake.bias", out)


if __name__ == '__main__':
    unittest.main()