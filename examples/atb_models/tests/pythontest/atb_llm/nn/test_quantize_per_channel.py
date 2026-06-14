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
from ddt import ddt, data

import torch
import torch_npu
import numpy as np

from atb_llm.nn.quantized import quantize_per_channel
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


def golden(x, input_scale, input_offset):
    int8_lower_bound = -128
    if x.dtype == torch.bfloat16:
        input_x = x.float().numpy()
        input_scale = input_scale.float().numpy()
        input_offset = input_offset.float().numpy()
    else:
        input_x = x.numpy()
        input_scale = input_scale.numpy()
        input_offset = input_offset.numpy()
    if len(input_offset) == 0:
        out = np.clip((np.round((input_x / input_scale))), int8_lower_bound, 127)
    else:
        out = np.clip((np.round((input_x / input_scale)) + input_offset), int8_lower_bound, 127)

    return torch.from_numpy(out).to(torch.int8)


@ddt
class TestPerChannelQuantFunction(unittest.TestCase):
    @data(torch.float16, torch.bfloat16)
    def test_per_channel_quant(self, dtype):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")
        y = quantize_per_channel(Tensor("x"), Tensor("input_scale"), Tensor("input_offset"))
        get_default_net().mark_output(y, "y")
        per_channel_quant_engine = get_default_net().build_engine()

        tokens_num = 100
        hidden_size = 1024

        x = torch.rand(tokens_num, hidden_size).to(dtype)
        input_scale = torch.rand(hidden_size).to(dtype)
        input_offset = torch.rand(hidden_size).to(torch.int8)
        y = torch.empty(tokens_num, hidden_size).to(torch.int8).npu()

        golden_y = golden(x, input_scale, input_offset)

        inputs = {"x": x.npu(), "input_scale": input_scale.npu(), "input_offset": input_offset.npu()}
        outputs = {"y": y}
        per_channel_quant_engine.forward(inputs, outputs)

        torch.npu.synchronize()

        assert torch.allclose(golden_y, y.cpu(), rtol=1, atol=1)


if __name__ == '__main__':
    unittest.main()