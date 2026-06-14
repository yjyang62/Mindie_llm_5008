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


def golden(inputs, multiples):
    input_tensor = inputs[0]
    input_dim = input_tensor.dim()
    shape_prefix = [1] * (len(multiples) - input_dim)
    input_tensor = input_tensor.view(*shape_prefix, *input_tensor.shape)
    return input_tensor.repeat(*multiples)


class TestRepeatFunction(unittest.TestCase):
    def test_repeat(self):
        multiples = [2, 1, 3]
        out = Tensor("input_tensor").repeat(multiples)
        get_default_net().mark_output(out, "out")
        repeat_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).to(torch.float16).npu()
        inputs = {"input_tensor": input_tensor}
        out = torch.empty(2, 100, 3072).to(torch.float16).npu()
        outputs = {"out": out}

        repeat_engine.forward(inputs, outputs)
        golden_out = golden([input_tensor], multiples)

        torch.npu.synchronize()
        assert torch.allclose(golden_out, out, rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
    unittest.main()