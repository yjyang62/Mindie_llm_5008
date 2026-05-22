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

import torch
import torch_npu
from ddt import ddt, data

import atb_llm.nn as nn
from atb_llm.nn.parameter import Parameter
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.quantize.pack_type import DataType
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


CPU = "cpu"


@ddt
class TestW8A8LinearDequantPerTokenPass(unittest.TestCase):
    def setUp(self):
        self.input_tensor = Tensor("input_tensor")
        self.weight = Parameter(prefix="model", suffix="weight")
        self.weight_scale = Parameter(prefix="model", suffix="weight_scale")
        self.activation_scale = Parameter(prefix="model", suffix="activation_scale")
        self.bias = Parameter(prefix="model", suffix="bias")
        self.transpose_b = True

    def forward(self, input_tensor, dtype, bias):
        output_dtype = DataType.ACL_BF16 if dtype == torch.bfloat16 else DataType.ACL_FLOAT16
        out = nn.functional.linear(
            input_tensor, self.weight.get_tensor(),
            transpose_b=self.transpose_b)
        out = nn.quantized.dequantize(
            out, self.weight_scale.get_tensor(), output_dtype=output_dtype,
            activation_scale=self.activation_scale.get_tensor())
        if bias:
            out = out + self.bias.get_tensor()
        return out

    def get_fusion_engine(self, dtype, bias):
        out = self.forward(self.input_tensor, dtype, bias)
        get_default_net().mark_output(out, "out")
        fusion_engine = get_default_net().build_engine()
        return fusion_engine

    def golden_without_bias(self, input_tensor, weight, weight_scale, activation_scale, dtype):
        golden_out = torch.mm(input_tensor.to(torch.float32), weight.to(torch.float32)) * weight_scale
        golden_out = (golden_out.T * activation_scale).T
        golden_out = golden_out.to(dtype)
        return golden_out

    def golden_with_bias(self, input_tensor, weight, weight_scale, activation_scale, bias, dtype):
        golden_out = torch.mm(input_tensor.to(torch.float32), weight.to(torch.float32)) * weight_scale
        golden_out = (golden_out.T * activation_scale).T
        golden_out = (golden_out + bias).to(dtype)
        return golden_out

    @data(torch.float16, torch.bfloat16)
    def test_w8a8_linear_dequant_per_token_without_bias(self, dtype):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This fusion pass doesn't support Atlas 300I DUO.")
        batch_size = 2
        seq_len = 128
        k = 256
        n = 512

        fusion_engine = self.get_fusion_engine(dtype, bias=False)

        input_tensor = torch.randint(low=-5, high=5, size=(batch_size * seq_len, k), dtype=torch.int8, device=CPU)
        out = torch.empty((batch_size * seq_len, n), dtype=dtype).npu()

        weight = torch.randint(low=-5, high=5, size=(n, k), dtype=torch.int8, device=CPU)
        weight_scale = torch.rand((n), dtype=dtype, device=CPU)
        if dtype == torch.float16:
            weight_scale = weight_scale.to(torch.float32)
        activation_scale = torch.rand((batch_size * seq_len), dtype=torch.float32, device=CPU)
        weights = {
            "model.weight": weight.npu(), "model.weight_scale": weight_scale.npu(),
            "model.activation_scale": activation_scale.npu()
        }

        inputs = {"input_tensor": input_tensor.npu()}
        fusion_outputs = {"out": out}

        fusion_engine.set_weights(weights)
        self.assertIn("W8A8MatMul", str(fusion_engine))
        fusion_engine.forward(inputs, fusion_outputs)

        weight_in_k_n = weight
        if self.transpose_b:
            weight_in_k_n = weight.T.contiguous()
        golden_out = self.golden_without_bias(input_tensor, weight_in_k_n, weight_scale, activation_scale, dtype)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(fusion_outputs["out"].cpu(), golden_out, rtol=1e-03, atol=1e-03))


    @data(torch.float16, torch.bfloat16)
    def test_w8a8_linear_dequant_per_token_with_bias(self, dtype):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This fusion pass doesn't support Atlas 300I DUO.")
        batch_size = 2
        seq_len = 128
        k = 256
        n = 512

        fusion_engine = self.get_fusion_engine(dtype, bias=True)

        input_tensor = torch.randint(low=-5, high=5, size=(batch_size * seq_len, k), dtype=torch.int8, device=CPU)
        out = torch.empty((batch_size * seq_len, n), dtype=dtype).npu()

        weight = torch.randint(low=-5, high=5, size=(n, k), dtype=torch.int8, device=CPU)
        weight_scale = torch.rand((n), dtype=dtype, device=CPU)
        if dtype == torch.float16:
            weight_scale = weight_scale.to(torch.float32)
        activation_scale = torch.rand((batch_size * seq_len), dtype=torch.float32, device=CPU)
        bias = torch.ones((n), dtype=dtype, device=CPU)
        weights = {
            "model.weight": weight.npu(), "model.weight_scale": weight_scale.npu(),
            "model.activation_scale": activation_scale.npu(), "model.bias": bias.npu()
        }

        inputs = {"input_tensor": input_tensor.npu()}
        fusion_outputs = {"out": out}

        fusion_engine.set_weights(weights)
        self.assertIn("W8A8MatMul", str(fusion_engine))
        fusion_engine.forward(inputs, fusion_outputs)

        weight_in_k_n = weight
        if self.transpose_b:
            weight_in_k_n = weight.T.contiguous()
        golden_out = self.golden_with_bias(input_tensor, weight_in_k_n, weight_scale, activation_scale, bias, dtype)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(fusion_outputs["out"].cpu(), golden_out, rtol=1e-02, atol=1e-02))


if __name__ == "__main__":
    unittest.main()
