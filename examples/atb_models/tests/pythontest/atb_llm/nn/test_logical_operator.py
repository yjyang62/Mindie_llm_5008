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

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional import logical_not, logical_and, logical_or


class TestLogicalNotFunction(unittest.TestCase):
    @staticmethod
    def golden(x):
        return torch.logical_not(x)

    def test_logical_not(self):
        get_default_net().mark_output(logical_not(Tensor("x")), "res")
        engine = get_default_net().build_engine()

        shape = (1024)
        x = (torch.randn(shape, dtype=torch.float16) > 0).to(torch.int8).npu()
        res = torch.empty(shape).npu().to(torch.int8)

        engine.forward({"x": x}, {"res": res})
        golden_out = self.golden(x.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.equal(res.cpu().to(torch.bool), golden_out))


class TestLogicalAndFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return torch.logical_and(x, y)

    def test_logical_and(self):
        get_default_net().mark_output(logical_and(Tensor("x"), Tensor('y')), "res")
        engine = get_default_net().build_engine()

        shape = (1024)
        x = (torch.randn(shape, dtype=torch.float16) > 0).to(torch.int8).npu()
        y = (torch.randn(shape, dtype=torch.float16) > 0).to(torch.int8).npu()
        res = torch.empty(shape).npu().to(torch.int8)

        engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.equal(res.cpu().to(torch.bool), golden_out))


class TestLogicalOrFunction(unittest.TestCase):
    @staticmethod
    def golden(x, y):
        return torch.logical_or(x, y)

    def test_logical_and(self):
        get_default_net().mark_output(logical_or(Tensor("x"), Tensor('y')), "res")
        engine = get_default_net().build_engine()

        shape = (1024)
        x = (torch.randn(shape, dtype=torch.float16) > 0).to(torch.int8).npu()
        y = (torch.randn(shape, dtype=torch.float16) > 0).to(torch.int8).npu()
        res = torch.empty(shape).npu().to(torch.int8)

        engine.forward({"x": x, "y": y}, {"res": res})
        golden_out = self.golden(x.cpu(), y.cpu())

        torch.npu.synchronize()
        self.assertTrue(torch.equal(res.cpu().to(torch.bool), golden_out))


if __name__ == '__main__':
    unittest.main()