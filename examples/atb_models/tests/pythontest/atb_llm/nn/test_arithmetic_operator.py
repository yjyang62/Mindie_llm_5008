# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import random
import unittest

import torch
import torch_npu
from ddt import ddt, data, unpack

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


@ddt
class TestAddFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return x + y

    def setUp(self):
        get_default_net().mark_output(Tensor("x") + Tensor('y'), "res")
        self.engine = get_default_net().build_engine()

    @data((torch.float16, 0.02), (torch.bfloat16, 0.001))
    @unpack
    def test_add(self, dtype, precision):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")
        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype).npu()
        y = torch.randn(shape, dtype=dtype).npu()
        res = torch.empty(shape, dtype=dtype).npu()

        self.engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x, y)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(res, golden_out, rtol=precision, atol=precision))


@ddt
class TestSubFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return x - y

    def setUp(self):
        get_default_net().mark_output(Tensor("x") - Tensor('y'), "res")
        self.engine = get_default_net().build_engine()

    @data(torch.float16)
    def test_sub_fp(self, dtype):
        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype).npu()
        y = torch.randn(shape, dtype=dtype).npu()
        res = torch.empty(shape, dtype=dtype).npu()

        self.engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(res.cpu(), golden_out, rtol=0.02, atol=0.02))

    def test_sub_int(self):
        shape = (1024, 1024)
        dtype = torch.int64
        x = torch.randint(-10, 10, shape, dtype=dtype).npu()
        y = torch.randint(-10, 10, shape, dtype=dtype).npu()
        res = torch.empty(shape, dtype=dtype).npu()

        self.engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(res.cpu(), golden_out, rtol=0.02, atol=0.02))


@ddt
class TestMulFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return x * y

    @data((torch.float16, 0.02), (torch.bfloat16, 0.02))
    @unpack
    def test_mul_tensor(self, dtype, precision):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")

        get_default_net().mark_output(Tensor("x") * Tensor('y'), "res")
        engine = get_default_net().build_engine()

        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype).npu()
        y = torch.randn(shape, dtype=dtype).npu()
        res = torch.empty(shape, dtype=dtype).npu()

        engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        self.assertTrue(torch.allclose(res.cpu(), golden_out, rtol=precision, atol=precision))

    @data((torch.float16, 0.02), (torch.bfloat16, 0.001))
    @unpack
    def test_mul_scalar(self, dtype, precision):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")

        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype).npu()
        scale = random.uniform(-100, 100)
        res = torch.empty(shape, dtype=dtype).npu()
 
        get_default_net().mark_output(Tensor("x") * scale, "res")
        engine = get_default_net().build_engine()

        engine.forward({"x": x}, {"res": res})
        golden_out = self.golden(x.cpu(), scale)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(res.cpu(), golden_out, rtol=precision, atol=precision))


@ddt
class TestDivFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return x / y

    @data((torch.float16, 0.02), (torch.bfloat16, 0.02))
    @unpack
    def test_div_tensor(self, dtype, precision):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")

        get_default_net().mark_output(Tensor("x") / Tensor('y'), "res")
        engine = get_default_net().build_engine()

        shape = (2, 2)
        x = torch.randn(shape, dtype=dtype).npu()
        y = torch.randn(shape, dtype=dtype).npu()
        res = torch.empty(shape, dtype=dtype).npu()

        engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        self.assertTrue(torch.allclose(res.cpu(), golden_out, rtol=precision, atol=precision))

    @data((torch.float16, 0.02), (torch.bfloat16, 0.001))
    @unpack
    def test_div_scalar(self, dtype, precision):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")

        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype).npu()
        scale = random.uniform(-100, 100)
        res = torch.empty(shape, dtype=dtype).npu()
 
        get_default_net().mark_output(Tensor("x") / scale, "res")
        engine = get_default_net().build_engine()

        engine.forward({"x": x}, {"res": res})
        golden_out = self.golden(x.cpu(), scale)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(res.cpu(), golden_out, rtol=precision, atol=precision))


if __name__ == '__main__':
    unittest.main()