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
from unittest.mock import MagicMock, patch, call

import torch

from atb_llm.layers.norm.normalization import RmsNorm
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.models.base.config import BaseConfig
from atb_llm.nn.parameter import Parameter
from atb_llm.nn.tensor import Tensor


class TestNormalizationLoader(unittest.TestCase):

    def setUp(self):
        self.config = BaseConfig(epsilon=1e-5)

    def test_rms_norm(self):
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        norm_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        mock_weight_tool_obj.get_tensor = MagicMock(return_value=norm_tensor)
        norm_module = RmsNorm(self.config, mock_weight_tool_obj, "norm")
        self.assertIsInstance(norm_module, RmsNorm)
        self.assertIsInstance(norm_module.module.weight, Parameter)
        self.assertTrue(torch.equal(norm_module.module.weight.data, norm_tensor))
        mock_weight_tool_obj.get_tensor.assert_called_once_with("norm.weight")

        input_tensor = Tensor("input_tensor")
        ret = norm_module.forward(input_tensor)
        self.assertIsInstance(ret, Tensor)

    @patch("atb_llm.layers.norm.normalization.check_weight_exists", return_value=True)
    @patch("atb_llm.layers.norm.normalization.is_all_zero", return_value=False)
    def test_rms_norm_with_bias(self, _1, _2):
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        norm_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        mock_weight_tool_obj.get_tensor = MagicMock(return_value=norm_tensor)
        norm_module = RmsNorm(self.config, mock_weight_tool_obj, "norm")
        self.assertIsInstance(norm_module, RmsNorm)
        self.assertIsInstance(norm_module.bias, Parameter)
        mock_weight_tool_obj.get_tensor.assert_has_calls([call("norm.bias"), call("norm.weight")])

        input_tensor = Tensor("input_tensor")
        ret = norm_module.forward(input_tensor)
        self.assertIsInstance(ret, Tensor)


if __name__ == '__main__':
    unittest.main()
