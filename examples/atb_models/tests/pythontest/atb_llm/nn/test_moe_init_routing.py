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

from atb_llm.nn.functional import moe_init_routing
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


def golden(params, inputs):
    tokens_num = params[0]
    topk_num = params[1]
    expert_num = params[2]
    hidden_size = params[3]
    input_tensor = inputs[0]
    expert_idx = inputs[1]

    expert_idx = expert_idx.reshape(tokens_num * topk_num)
    sorted_row_idx = torch.argsort(expert_idx, stable=True)
    golden_expanded_row_idx_out = torch.empty(tokens_num * topk_num).to(torch.int32)
    golden_expanded_row_idx_out[sorted_row_idx] = torch.arange(tokens_num * topk_num).to(torch.int32)

    token_indices = sorted_row_idx // topk_num
    golden_expanded_x_out = torch.gather(input_tensor, 0, token_indices.unsqueeze(1).expand(-1, hidden_size))
    counts = torch.bincount(expert_idx, minlength=expert_num)
    golden_cumsum_out = torch.cumsum(counts, dim=0).to(torch.int32)
    return golden_expanded_x_out, golden_expanded_row_idx_out.npu(), golden_cumsum_out


class TestMoeInitRoutingFunction(unittest.TestCase):
    def test_moe_init_routing(self):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This operation doesn't support Atlas 300I DUO.")

        tokens_num = 4
        topk_num = 3
        expert_num = 6
        hidden_size = 1000

        expanded_x_out, expanded_row_idx_out, cumsum_out = moe_init_routing(
            Tensor("input_tensor"), Tensor("expert_idx"), topk_num, expert_num)
        get_default_net().mark_output(expanded_x_out, "expanded_x_out")
        get_default_net().mark_output(expanded_row_idx_out, "expanded_row_idx_out")
        get_default_net().mark_output(cumsum_out, "cumsum_out")
        init_routing_engine = get_default_net().build_engine()

        input_tensor = torch.rand(tokens_num, hidden_size).to(torch.float16).npu()
        expert_idx = torch.tensor([5, 1, 2, 0, 2, 3, 3, 0, 1, 1, 5, 2]).to(torch.int32).npu()
        expert_idx = expert_idx.reshape(tokens_num, topk_num)
        inputs = {"input_tensor": input_tensor, "expert_idx": expert_idx}

        expanded_x_out = torch.empty(tokens_num * topk_num, hidden_size).to(torch.float16).npu()
        expanded_row_idx_out = torch.empty(tokens_num * topk_num).to(torch.int32).npu()
        cumsum_out = torch.empty(expert_num).to(torch.int32).npu()
        outputs = {"expanded_x_out": expanded_x_out, "expanded_row_idx_out": expanded_row_idx_out,
                   "cumsum_out": cumsum_out}

        init_routing_engine.forward(inputs, outputs)
        (golden_expanded_x_out, golden_expanded_row_idx_out,
         golden_cumsum_out) = golden([tokens_num, topk_num, expert_num, hidden_size], [input_tensor, expert_idx])

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(golden_expanded_x_out, expanded_x_out, rtol=1e-02, atol=1e-02))
        self.assertTrue(torch.allclose(golden_expanded_row_idx_out, expanded_row_idx_out, rtol=1e-02, atol=1e-02))
        self.assertTrue(torch.allclose(golden_cumsum_out, cumsum_out, rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()