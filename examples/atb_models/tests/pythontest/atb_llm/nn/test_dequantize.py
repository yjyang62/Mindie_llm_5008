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
from ddt import ddt, data, unpack

from atb_llm.nn.quantized import dequantize
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.quantize.pack_type import DataType
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


def golden(x, weight_scale, activation_scale=None, bias=None):
    if bias is not None:
        x = x + bias
    out = x * weight_scale
    if activation_scale is not None:
        out = out * activation_scale.reshape(-1, 1)
    return out


@ddt
class TestDequantize(unittest.TestCase):
    def setUp(self):
        self.m = 100
        self.n = 20

    @data((torch.bfloat16, torch.bfloat16), (torch.float16, torch.float32))
    @unpack
    def test_dequantize_per_tensor(self, out_dtype, weight_scale_type):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This operation doesn't support Atlas 300I DUO.")

        out = dequantize(
            Tensor("x"), Tensor("weight_scale"),
            DataType.ACL_BF16 if out_dtype == torch.bfloat16 else DataType.ACL_FLOAT16,
            bias=Tensor("bias")
        )
        get_default_net().mark_output(out, "out")
        dequantize_engine = get_default_net().build_engine()

        x = torch.randint(low=-128, high=127, size=(self.m, self.n), dtype=torch.int32)
        weight_scale = torch.randn((self.n,), dtype=weight_scale_type)
        bias = torch.randint(low=-128, high=127, size=(self.n,), dtype=torch.int32)

        inputs = {"x": x.npu(), "weight_scale": weight_scale.npu(), "bias": bias.npu()}
        out = torch.empty((self.m, self.n)).to(out_dtype).npu()
        outputs = {"out": out}

        dequantize_engine.forward(inputs, outputs)
        golden_y = golden(x, weight_scale, bias=bias)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(golden_y.to(out_dtype), out.cpu(), rtol=1e-05, atol=1e-05))

    @data((torch.bfloat16, torch.bfloat16), (torch.float16, torch.float32))
    @unpack
    def test_dequantize_per_token(self, out_dtype, weight_scale_type):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This operation doesn't support Atlas 300I DUO.")

        out = dequantize(
            Tensor("x"), Tensor("weight_scale"),
            DataType.ACL_BF16 if out_dtype == torch.bfloat16 else DataType.ACL_FLOAT16,
            Tensor("activation_scale"),
            Tensor("bias")
        )
        get_default_net().mark_output(out, "out")
        dequantize_engine = get_default_net().build_engine()

        x = torch.randint(low=-128, high=127, size=(self.m, self.n), dtype=torch.int32)
        weight_scale = torch.randn((self.n,), dtype=weight_scale_type)
        activation_scale = torch.randn((self.m,), dtype=torch.float32)
        bias = torch.randint(low=-128, high=127, size=(self.n,), dtype=torch.int32)

        inputs = {
            "x": x.npu(), "weight_scale": weight_scale.npu(),
            "activation_scale": activation_scale.npu(), "bias": bias.npu()
        }
        out = torch.empty((self.m, self.n)).to(out_dtype).npu()
        outputs = {"out": out}

        dequantize_engine.forward(inputs, outputs)
        golden_y = golden(x, weight_scale, activation_scale, bias)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(golden_y.to(out_dtype), out.cpu(), rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()
