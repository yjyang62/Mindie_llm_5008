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

from atb_llm.nn.functional import amax
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor


def golden(x, axis):
    return torch.amax(x, axis)


class TestAmaxFunction(unittest.TestCase):
    def test_amax(self):
        axis = [0]
        x = torch.rand(100, 1024).to(torch.int32).npu()
        golden_out = golden(x, axis)

        out = amax(Tensor("x"), axis)
        get_default_net().mark_output(out, "out")
        amax_engine = get_default_net().build_engine()

        out = torch.empty(golden_out.shape).to(torch.int32).npu()
        inputs = {"x": x}
        outputs = {"out": out}
        amax_engine.forward(inputs, outputs)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(golden_out, out, rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()