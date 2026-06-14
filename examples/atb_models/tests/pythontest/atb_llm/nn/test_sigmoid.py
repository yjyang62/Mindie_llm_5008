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

from atb_llm.nn.functional import activation, ActType
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.network_manager import get_default_net


def golden(inputs):
    input_tensor = inputs[0]
    return torch.sigmoid(input_tensor)


class TestSigmoidFunction(unittest.TestCase):
    def test_sigmoid(self):
        out = activation(Tensor("input_tensor"), ActType.SIGMOID)
        get_default_net().mark_output(out, "out")
        sigmoid_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).to(torch.float16).npu()
        inputs = {"input_tensor": input_tensor}
        out = torch.empty(100, 1024).to(torch.float16).npu()
        outputs = {"out": out}

        sigmoid_engine.forward(inputs, outputs)
        golden_out = golden([input_tensor])

        torch.npu.synchronize()
        assert torch.allclose(golden_out, out, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
    unittest.main()