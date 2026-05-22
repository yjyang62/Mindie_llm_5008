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

from atb_llm.utils.layers import load_column_multi, load_row


@ddt
class TestLayerInit(unittest.TestCase):
    def test_load_column_multi(self):
        config, weights = MagicMock(), MagicMock()
        config.quantize = None
        weight = torch.ones(256, 256, dtype=torch.float16)
        bias = torch.zeros(256, dtype=torch.float16)
        weights.get_multi_weights_col.return_value = weight
        weights.get_sharded.return_value = bias
        module = load_column_multi(config, ["prefix1", "prefix2"], weights, head_size=1)
        self.assertTrue(torch.allclose(weight, module.linear.weight))
        module = load_column_multi(config, ["prefix1", "prefix2"], weights, head_size=1, bias=True)
        self.assertTrue(torch.allclose(torch.cat([bias, bias], dim=0), module.linear.bias))
        module = load_column_multi(config, ["prefix1", "prefix2"], weights, head_size=1, lm_head=True)
        self.assertTrue(torch.allclose(weight, module.linear.weight.cpu()))

    def test_load_row(self):
        config, weights = MagicMock(), MagicMock()
        config.quantize = None
        weight = torch.ones(256, 256, dtype=torch.float16)
        weights.get_sharded.return_value = weight
        module = load_row(config, "prefix", weights, head_size=1)
        self.assertTrue(torch.allclose(weight, module.linear.weight))
        weights.get_sharded.assert_called_with("prefix.weight", dim=1, gqa_size=1)


if __name__ == '__main__':
    unittest.main()