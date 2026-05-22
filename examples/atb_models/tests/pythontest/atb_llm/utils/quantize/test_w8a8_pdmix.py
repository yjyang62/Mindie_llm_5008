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

from atb_llm.utils.layers.linear import W8A8PDMixLinear


@ddt
class TestW8A8LinearDynamic(unittest.TestCase):
    @patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=1)
    def test_init_pergroup(self, mock_check_transpose):
        weight = torch.ones(64, 512, 128, dtype=torch.int8)
        weight_scale = torch.ones(64, 512, 1, dtype=torch.float16)
        weight_offset = torch.ones(64, 512, 1, dtype=torch.float16)
        deq_scale = torch.ones(64, 512)
        quant_bias = torch.ones(64, 512)
        input_scale = torch.ones(64)
        input_offset = torch.ones(64)
        linear = W8A8PDMixLinear(weight, weight_scale, weight_offset, deq_scale, quant_bias, input_scale, input_offset)
        self.assertEqual(linear.weight.shape, (64, 512, 128))


if __name__ == '__main__':
    unittest.main()