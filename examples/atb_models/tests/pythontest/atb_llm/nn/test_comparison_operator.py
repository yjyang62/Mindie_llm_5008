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
from ddt import ddt, data

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional import equal


@ddt
class TestGtFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return x > y

    def setUp(self):
        get_default_net().mark_output(Tensor("x") > Tensor('y'), "res")
        self.engine = get_default_net().build_engine()

    @data(torch.float16, torch.float32)
    def test_gt_fp(self, dtype):
        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype).npu()
        y = torch.randn(shape, dtype=dtype).npu()
        res = torch.empty(shape).npu().to(torch.int8)

        self.engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.equal(res.cpu().to(torch.bool), golden_out))

    def test_gt_int(self):
        shape = (1024, 1024)
        dtype = torch.int64
        x = torch.randint(-10, 10, shape, dtype=dtype).npu()
        y = torch.randint(-10, 10, shape, dtype=dtype).npu()
        res = torch.empty(shape).npu().to(torch.int8)

        self.engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.equal(res.cpu().to(torch.bool), golden_out))


@ddt
class TestLtFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return x < y

    def setUp(self):
        get_default_net().mark_output(Tensor("x") < Tensor('y'), "res")
        self.engine = get_default_net().build_engine()

    @data(torch.float16, torch.float32)
    def test_lt_fp(self, dtype):
        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype).npu()
        y = torch.randn(shape, dtype=dtype).npu()
        res = torch.empty(shape).npu().to(torch.int8)

        self.engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.equal(res.cpu().to(torch.bool), golden_out))

    def test_lt_int(self):
        shape = (1024, 1024)
        dtype = torch.int64
        x = torch.randint(-10, 10, shape, dtype=dtype).npu()
        y = torch.randint(-10, 10, shape, dtype=dtype).npu()
        res = torch.empty(shape).npu().to(torch.int8)

        self.engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.equal(res.cpu().to(torch.bool), golden_out))


@ddt
class TestEqFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return torch.eq(x, y)

    def setUp(self):
        get_default_net().mark_output(
            equal(Tensor("x"), Tensor("y")), "res")
        self.engine = get_default_net().build_engine()

    @data(torch.float16, torch.float32)
    def test_eq(self, dtype):
        shape = (1024, 1024)
        x = torch.randn(shape, dtype=dtype).npu()
        y = torch.randn(shape, dtype=dtype).npu()
        res = torch.empty(shape).npu().to(torch.int8)

        self.engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.equal(res.cpu().to(torch.bool), golden_out))


if __name__ == '__main__':
    unittest.main()