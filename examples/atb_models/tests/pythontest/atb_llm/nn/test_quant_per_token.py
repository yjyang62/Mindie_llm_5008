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

from atb_llm.nn.quantized import quantize_per_token
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor


def golden(x):
    scale = torch.max(torch.abs(x), dim=-1).values / 127
    scale = scale.unsqueeze(-1)
    y = torch.round(x / scale)
    return y


class TestDynamicQuantFunction(unittest.TestCase):
    def test_dynamic_quant(self):
        y, scale = quantize_per_token(Tensor("x"))
        get_default_net().mark_output(y, "y")
        get_default_net().mark_output(scale, "scale")
        dynamic_quant_engine = get_default_net().build_engine()

        tokens_num = 100
        hidden_size = 1024

        x = torch.rand(tokens_num, hidden_size).to(torch.float16).npu()
        inputs = {"x": x}
        y = torch.empty(tokens_num, hidden_size).to(torch.int8).npu()
        scale = torch.empty(tokens_num).to(torch.float16).npu()
        outputs = {"y": y, "scale": scale}

        dynamic_quant_engine.forward(inputs, outputs)
        golden_y = golden(x)

        torch.npu.synchronize()
        assert torch.allclose(golden_y.to(torch.int8), y, rtol=1, atol=1)


if __name__ == '__main__':
    unittest.main()