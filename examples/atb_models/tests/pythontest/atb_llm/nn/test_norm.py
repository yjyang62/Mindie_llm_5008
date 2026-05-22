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

from atb_llm.nn.functional import norm
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor


def golden(x):
    return torch.norm(x, p=1, dim=1, keepdim=True)


class TestNormFunction(unittest.TestCase):
    def test_norm(self):
        out = norm(Tensor("x"))
        get_default_net().mark_output(out, "out")
        vector_norm_engine = get_default_net().build_engine()

        tokens_num = 10
        hidden_size = 1000
        x = torch.rand(tokens_num, hidden_size).to(torch.float16).npu()
        out = torch.empty(tokens_num, 1).to(torch.float16).npu()
        inputs = {"x": x}
        outputs = {"out": out}

        vector_norm_engine.forward(inputs, outputs)
        golden_out = golden(x)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(golden_out, out, rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()