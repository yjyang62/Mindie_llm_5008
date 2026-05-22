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
from unittest.mock import patch, MagicMock

from ddt import ddt

import torch

from atb_llm.utils.layers.linear import W8A8LinearStatic


@ddt
class TestW8A8LinearStatic(unittest.TestCase):

    def setUp(self):
        self.weight = torch.ones(64, 512, 128, dtype=torch.int8)
        self.deq_scale = torch.ones(64, 512)
        self.input_scale = torch.ones(64)
        self.quant_bias = torch.ones(64, 512)
        self.input_offset = torch.ones(64)
        self.bias = torch.ones(64, 512)

    @patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=0)
    @patch('atb_llm.utils.initial.NPUSocInfo', return_value=MagicMock())
    def test_init_without_bias_offset(self, mock_npu_soc_info, mock_check_transpose):
        mock_npu_soc_info.soc_version = 101
        self.linear = W8A8LinearStatic(self.weight, self.deq_scale, self.input_scale)
        self.assertEqual(self.linear.weight.shape, (64, 128, 512))

    @patch('atb_llm.utils.layers.linear.linear_utils.LinearUtils.check_transpose', return_value=0)
    @patch('atb_llm.utils.initial.NPUSocInfo', return_value=MagicMock())
    def test_init_with_all(self, mock_npu_soc_info, mock_check_transpose):
        mock_npu_soc_info.soc_version = 101
        self.linear = W8A8LinearStatic(self.weight, self.deq_scale, self.input_scale, self.quant_bias,
                                       self.input_offset, self.bias)
        self.assertEqual(self.linear.weight.shape, (64, 128, 512))


if __name__ == '__main__':
    unittest.main()