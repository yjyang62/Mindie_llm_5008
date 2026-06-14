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
from unittest.mock import MagicMock, patch, call, ANY

import torch

from atb_llm.layers.linear.linear import (
    RowParallelLinear, ColumnParallelLinear, ReplicatedLinear, MergedColumnParallelLinear
)
from atb_llm.layers import InferenceMode
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.models.base.config import BaseConfig
from atb_llm.nn.parameter import Parameter
from atb_llm.layers import QuantTypeV3
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.quantize.pack_type import DataType


class TestLinear(unittest.TestCase):

    def setUp(self):
        self.config = BaseConfig(torch_dtype=torch.float16)

        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        mock_weight_tool_obj.mapping = MagicMock(spec=Mapping)
        mock_weight_tool_obj.mapping.rank = 1
        mock_weight_tool_obj.mapping.world_size = 4
        self.mock_weight_tool_obj = mock_weight_tool_obj

    @patch("atb_llm.layers.linear.linear.get_linear_quant_type", return_value=QuantTypeV3.FLOAT16)
    @patch("atb_llm.layers.linear.linear.nn.functional.linear", return_value=Tensor("out"))
    def test_row_parallel_linear_float(self, mock_linear, _):
        linear_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        self.mock_weight_tool_obj.get_sharded = MagicMock(return_value=linear_tensor)
        module = RowParallelLinear(self.config, self.mock_weight_tool_obj, ["linear"])
        self.assertIsInstance(module, RowParallelLinear)
        self.assertIsInstance(module.weight, Parameter)
        self.assertTrue(torch.allclose(module.weight.data, linear_tensor))
        self.mock_weight_tool_obj.get_sharded.assert_called_once_with(
            "linear.weight", dim=1, chunk_id=self.mock_weight_tool_obj.mapping.rank,
            num_chunk=self.mock_weight_tool_obj.mapping.world_size
        )
        out = module.forward(Tensor("input"))
        mock_linear.assert_called_with(Tensor("input"), Tensor("linear.weight"), transpose_b=True)
        self.assertEqual(Tensor("out"), out)

    @patch("atb_llm.layers.linear.linear.get_linear_quant_type", return_value=QuantTypeV3.W8A8_MIX)
    @patch("atb_llm.layers.linear.linear.RowParallelLinear.weight_loader", return_value=torch.tensor([]))
    @patch("atb_llm.layers.linear.linear.nn.quantized.quantize_per_token",
           return_value=[Tensor("quant_per_token_out"), Tensor("quant_scale_out")])
    @patch("atb_llm.layers.linear.linear.nn.functional.linear", return_value=Tensor("linear_out"))
    @patch("atb_llm.layers.linear.linear.nn.quantized.dequantize", return_value=Tensor("dequant_out"))
    def test_row_parallel_linear_w8a8_mix_prefill(
        self, mock_dequant, mock_linear, mock_quant_per_token, _1, _2):
        module = RowParallelLinear(self.config, self.mock_weight_tool_obj, ["linear"])
        self.assertIsInstance(module, RowParallelLinear)
        self.assertIsInstance(module.weight, Parameter)
        self.assertIsInstance(module.deq_scale, Parameter)
        self.assertIsInstance(module.quant_bias, Parameter)
        self.assertIsInstance(module.input_scale, Parameter)
        self.assertIsInstance(module.input_offset, Parameter)
        self.assertIsInstance(module.weight, Parameter)
        self.assertIsInstance(module.weight_scale, Parameter)

        out = module.forward(Tensor("input"), InferenceMode.PREFILL)
        mock_quant_per_token.assert_called_with(Tensor("input"))
        mock_linear.assert_called_with(Tensor("quant_per_token_out"), Tensor("linear.weight"), transpose_b=True)
        mock_dequant.assert_called_with(
            Tensor("linear_out"), Tensor("linear.weight_scale"),
            output_dtype=DataType.ACL_FLOAT16, activation_scale=Tensor("quant_scale_out"))
        self.assertEqual(Tensor("dequant_out"), out)

    @patch("atb_llm.layers.linear.linear.get_linear_quant_type", return_value=QuantTypeV3.W8A8_MIX)
    @patch("atb_llm.layers.linear.linear.RowParallelLinear.weight_loader", return_value=torch.tensor([]))
    @patch("atb_llm.layers.linear.linear.nn.quantized.quantize_per_token",
           return_value=[Tensor("quant_per_token_out"), Tensor("quant_scale_out")])
    @patch("atb_llm.layers.linear.linear.nn.functional.linear", return_value=Tensor("linear_out"))
    @patch("atb_llm.layers.linear.linear.nn.quantized.dequantize", return_value=Tensor("dequant_out"))
    def test_row_parallel_linear_w8a8_mix_prefill_with_bias(
        self, mock_dequant, mock_linear, mock_quant_per_token, _1, _2):
        module = RowParallelLinear(self.config, self.mock_weight_tool_obj, ["linear"], bias=True)
        self.assertIsInstance(module, RowParallelLinear)
        self.assertIsInstance(module.weight, Parameter)
        self.assertIsInstance(module.deq_scale, Parameter)
        self.assertIsInstance(module.quant_bias, Parameter)
        self.assertIsInstance(module.input_scale, Parameter)
        self.assertIsInstance(module.input_offset, Parameter)
        self.assertIsInstance(module.weight, Parameter)
        self.assertIsInstance(module.weight_scale, Parameter)
        self.assertIsInstance(module.bias, Parameter)

        _ = module.forward(Tensor("input"), InferenceMode.PREFILL)
        mock_quant_per_token.assert_called_with(Tensor("input"))
        mock_linear.assert_called_with(Tensor("quant_per_token_out"), Tensor("linear.weight"), transpose_b=True)
        mock_dequant.assert_called_with(
            Tensor("linear_out"), Tensor("linear.weight_scale"),
            output_dtype=DataType.ACL_FLOAT16, activation_scale=Tensor("quant_scale_out"))

    @patch("atb_llm.layers.linear.linear.get_linear_quant_type", return_value=QuantTypeV3.W8A8_MIX)
    @patch("atb_llm.layers.linear.linear.RowParallelLinear.weight_loader", return_value=torch.tensor([]))
    @patch("atb_llm.layers.linear.linear.nn.quantized.quantize_per_channel",
           return_value=Tensor("quant_per_channel_out"))
    @patch("atb_llm.layers.linear.linear.nn.functional.linear", return_value=Tensor("linear_out"))
    @patch("atb_llm.layers.linear.linear.nn.quantized.dequantize", return_value=Tensor("dequant_out"))
    def test_row_parallel_linear_w8a8_mix_decode(
        self, mock_dequant, mock_linear, mock_quant_per_channel, _1, _2):
        module = RowParallelLinear(self.config, self.mock_weight_tool_obj, ["linear"])
        self.assertIsInstance(module, RowParallelLinear)
        self.assertIsInstance(module.weight, Parameter)
        self.assertIsInstance(module.deq_scale, Parameter)
        self.assertIsInstance(module.quant_bias, Parameter)
        self.assertIsInstance(module.input_scale, Parameter)
        self.assertIsInstance(module.input_offset, Parameter)
        self.assertIsInstance(module.weight, Parameter)
        self.assertIsInstance(module.weight_scale, Parameter)

        out = module.forward(Tensor("input"), InferenceMode.DECODE)
        mock_quant_per_channel.assert_called_with(Tensor("input"), Tensor("linear.input_scale"), Tensor("linear.input_offset"))
        mock_linear.assert_called_with(Tensor("quant_per_channel_out"), Tensor("linear.weight"), transpose_b=True)
        mock_dequant.assert_called_with(
            Tensor("linear_out"), Tensor("linear.deq_scale"),
            output_dtype=DataType.ACL_FLOAT16, bias=Tensor("linear.quant_bias"))
        self.assertEqual(Tensor("dequant_out"), out)

    @patch("atb_llm.layers.linear.linear.replicated_loader", return_value=torch.tensor([]))
    @patch("atb_llm.layers.linear.linear.get_linear_quant_type", return_value=QuantTypeV3.W8A8)
    @patch("atb_llm.layers.linear.linear.BaseLinear.load_weight")
    def test_row_parallel_linear_w8a8(self, _1, _2, mock_replicated_loader):
        linear_module = RowParallelLinear(self.config, self.mock_weight_tool_obj, ["linear"])

        deq_scale = Parameter(prefix="linear", suffix="deq_scale")
        linear_module.weight_loader(deq_scale, ["linear"])
        mock_replicated_loader.assert_called_with(deq_scale, self.mock_weight_tool_obj, ["linear"])

        weight_scale = Parameter(prefix="linear", suffix="weight_scale")
        out = linear_module.weight_loader(weight_scale, ["linear"])
        mock_replicated_loader.assert_called_with(weight_scale, self.mock_weight_tool_obj, ["linear"])

        quant_bias = Parameter(prefix="linear", suffix="quant_bias")
        out = linear_module.weight_loader(quant_bias, ["linear"])
        mock_replicated_loader.assert_called_with(quant_bias, self.mock_weight_tool_obj, ["linear"])
        self.assertTrue(torch.all(out == 0))

        input_scale = Parameter(prefix="linear", suffix="input_scale")
        out = linear_module.weight_loader(input_scale, ["linear"])
        mock_replicated_loader.assert_called_with(input_scale, self.mock_weight_tool_obj, ["linear"], is_uniform=True)

        input_offset = Parameter(prefix="linear", suffix="input_offset")
        out = linear_module.weight_loader(input_offset, ["linear"])
        mock_replicated_loader.assert_called_with(input_offset, self.mock_weight_tool_obj, ["linear"], is_uniform=True)

    @patch("atb_llm.layers.linear.linear.replicated_loader", return_value=torch.tensor([]))
    @patch("atb_llm.layers.linear.linear.get_linear_quant_type", return_value=QuantTypeV3.W8A8)
    @patch("atb_llm.layers.linear.linear.BaseLinear.load_weight")
    def test_column_parallel_linear_w8a8(self, _1, _2, mock_replicated_loader):
        linear_module = ColumnParallelLinear(self.config, self.mock_weight_tool_obj, ["linear"])

        input_scale = Parameter(prefix="linear", suffix="input_scale")
        _ = linear_module.weight_loader(input_scale, ["linear"])
        mock_replicated_loader.assert_called_with(input_scale, self.mock_weight_tool_obj, ["linear"], is_uniform=True)

        input_offset = Parameter(prefix="linear", suffix="input_offset")
        _ = linear_module.weight_loader(input_offset, ["linear"])
        mock_replicated_loader.assert_called_with(input_offset, self.mock_weight_tool_obj, ["linear"], is_uniform=True)

    @patch("atb_llm.layers.linear.linear.replicated_loader", return_value=torch.tensor([]))
    @patch("atb_llm.layers.linear.linear.get_linear_quant_type", return_value=QuantTypeV3.FLOAT16)
    @patch("atb_llm.layers.linear.linear.BaseLinear.load_weight")
    def test_replicated_linear(self, _1, _2, mock_replicated_loader):
        linear_module = ReplicatedLinear(self.config, self.mock_weight_tool_obj, ["linear"])

        weight = Parameter(prefix="linear", suffix="weight")
        _ = linear_module.weight_loader(weight, ["linear"])
        mock_replicated_loader.assert_called_with(weight, self.mock_weight_tool_obj, ["linear"])

    @patch("atb_llm.layers.linear.linear.get_linear_quant_type", return_value=QuantTypeV3.FLOAT16)
    def test_merged_column_parallel_linear(self, _):
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        linear_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        mock_weight_tool_obj.get_sharded = MagicMock(return_value=linear_tensor)
        linear_module = MergedColumnParallelLinear(self.config, mock_weight_tool_obj, ["linear1", "linear2"])
        self.assertIsInstance(linear_module, MergedColumnParallelLinear)
        self.assertIsInstance(linear_module[0].weight, Parameter)
        self.assertTrue(torch.allclose(
            linear_module[0].weight.data,
            torch.cat([linear_tensor, linear_tensor])
        ))
        self.assertEqual(mock_weight_tool_obj.get_sharded.call_count, 2)

        input_tensor = Tensor("input_tensor")
        out = linear_module.forward(input_tensor)
        self.assertIsInstance(out[0], Tensor)


if __name__ == '__main__':
    unittest.main()
