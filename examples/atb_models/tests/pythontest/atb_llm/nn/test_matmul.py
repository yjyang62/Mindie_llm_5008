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

from atb_llm.nn.functional import linear
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor


def golden(x, weight):
    return torch.matmul(x, weight)


class TestMatmulFunction(unittest.TestCase):
    def test_matmul(self):
        out = linear(Tensor("x"), Tensor("weight"), None, False, False)
        get_default_net().mark_output(out, "out")
        matmul_engine = get_default_net().build_engine()

        x = torch.rand(1000, 2000).to(torch.float16).npu()
        weight = torch.rand(2000, 3000).to(torch.float16).npu()
        inputs = {"x": x, "weight": weight}
        out = torch.empty(1000, 3000).to(torch.float16).npu()
        outputs = {"out": out}

        matmul_engine.forward(inputs, outputs)
        golden_out = golden(x, weight)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(golden_out, out, rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()