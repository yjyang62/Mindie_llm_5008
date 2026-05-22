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
class TestW8A8LinearDequantPerTensorPass(unittest.TestCase):
    def setUp(self):
        self.input_tensor = Tensor("input_tensor")
        self.weight = Parameter(prefix="model", suffix="weight")
        self.deq_scale = Parameter(prefix="model", suffix="deq_scale")
        self.quant_bias = Parameter(prefix="model", suffix="quant_bias")
        self.transpose_b = True

    def forward(self, input_tensor, dtype):
        output_dtype = DataType.ACL_BF16 if dtype == torch.bfloat16 else DataType.ACL_FLOAT16
        out = nn.functional.linear(
            input_tensor, self.weight.get_tensor(),
            transpose_b=self.transpose_b)
        out = nn.quantized.dequantize(
            out, self.deq_scale.get_tensor(), output_dtype=output_dtype, bias=self.quant_bias.get_tensor())
        return out

    def get_fusion_engine(self, dtype):
        out = self.forward(self.input_tensor, dtype)
        get_default_net().mark_output(out, "out")
        fusion_engine = get_default_net().build_engine()
        return fusion_engine

    @data(torch.float16, torch.bfloat16)
    def test_w8a8_linear_dequant_per_tensor(self, dtype):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")
        batch_size = 2
        seq_len = 128
        k = 256
        n = 512

        fusion_engine = self.get_fusion_engine(dtype)

        input_tensor = torch.randint(low=-5, high=5, size=(batch_size * seq_len, k), dtype=torch.int8, device=CPU)
        out = torch.empty((batch_size * seq_len, n), dtype=dtype).npu()

        weight = torch.randint(low=-5, high=5, size=(n, k), dtype=torch.int8, device=CPU)
        deq_scale = torch.randint(low=-5, high=5, size=(n,), dtype=torch.int64, device=CPU)
        if dtype == torch.bfloat16:
            deq_scale = deq_scale.to(torch.float32)
        quant_bias = torch.randint(low=-5, high=5, size=(n,), dtype=torch.int32, device=CPU)
        weights = {
            "model.weight": weight.npu(), "model.deq_scale": deq_scale.npu(),
            "model.quant_bias": quant_bias.npu()
        }

        inputs = {"input_tensor": input_tensor.npu()}
        fusion_outputs = {"out": out}

        fusion_engine.set_weights(weights)
        self.assertIn("W8A8MatMul", str(fusion_engine))
        fusion_engine.forward(inputs, fusion_outputs)

        torch.npu.synchronize()
        self.assertIsInstance(fusion_outputs["out"].cpu(), torch.Tensor)


if __name__ == "__main__":
    unittest.main()
