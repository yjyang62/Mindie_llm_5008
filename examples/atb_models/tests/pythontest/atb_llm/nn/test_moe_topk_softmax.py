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
import torch_npu

from atb_llm.nn.functional.moe import moe_topk_softmax
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


def golden(params, inputs):
    tokens_num = params[0]
    topk_num = params[1]
    input_tensor = inputs[0]

    softmax_out = torch.softmax(input_tensor, dim=-1)
    golden_output, golden_expert_idx_out = torch.topk(softmax_out, k=topk_num, dim=-1, sorted=False)
    golden_row_idx_out = torch.arange(tokens_num * topk_num).reshape(topk_num, tokens_num).transpose(1, 0)
    return golden_output, golden_expert_idx_out, golden_row_idx_out


class TestMoeTopkSoftmaxFunction(unittest.TestCase):
    def test_moe_topk_softmax(self):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This operation doesn't support Atlas 300I DUO.")

        tokens_num = 100
        expert_num = 8
        topk_num = 4

        output, expert_idx_out, row_idx_out = moe_topk_softmax(Tensor("input_tensor"), topk_num)
        get_default_net().mark_output(output, "output")
        get_default_net().mark_output(expert_idx_out, "expert_idx_out")
        get_default_net().mark_output(row_idx_out, "row_idx_out")
        topk_softmax_engine = get_default_net().build_engine()

        input_tensor = torch.rand(tokens_num, expert_num).to(torch.float16).npu()
        inputs = {"input_tensor": input_tensor}

        output = torch.empty(tokens_num, topk_num).to(torch.float16).npu()
        expert_idx_out = torch.empty(tokens_num, topk_num).to(torch.int32).npu()
        row_idx_out = torch.empty(tokens_num, topk_num).to(torch.int32).npu()
        outputs = {"output": output, "expert_idx_out": expert_idx_out, "row_idx_out": row_idx_out}

        topk_softmax_engine.forward(inputs, outputs)
        (golden_output, golden_expert_idx_out,
         golden_row_idx_out) = golden([tokens_num, topk_num, expert_num], [input_tensor])

        torch.npu.synchronize()
        assert torch.allclose(golden_output, output, rtol=1e-02, atol=1e-02)
        assert torch.allclose(golden_expert_idx_out.to(torch.int32), expert_idx_out, rtol=1e-02, atol=1e-02)
        assert torch.allclose(golden_row_idx_out.to(torch.int32).npu(), row_idx_out, rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
    unittest.main()