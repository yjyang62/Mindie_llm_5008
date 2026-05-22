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

from atb_llm.nn.functional.moe import gating
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor


def golden(params, inputs):
    topk_num = params[0]
    expert_num = params[1]
    topk = inputs[0]

    golden_original_index = torch.argsort(topk, stable=True)
    golden_token_index = golden_original_index // topk_num
    counts = torch.bincount(topk, minlength=expert_num)
    golden_cum_sum = torch.cumsum(counts, dim=0)
    return golden_token_index, golden_cum_sum, golden_original_index


class TestMoeGatingFunction(unittest.TestCase):
    def test_moe_gating(self):
        tokens_num = 4
        topk_num = 2
        expert_num = 6

        token_index, cum_sum, original_index = gating(Tensor("topk"), Tensor("idx_arr"),
                                                      topk_expert_num=topk_num, cum_sum_num=expert_num)
        get_default_net().mark_output(token_index, "token_index")
        get_default_net().mark_output(cum_sum, "cum_sum")
        get_default_net().mark_output(original_index, "original_index")
        gating_engine = get_default_net().build_engine()

        topk = torch.tensor([5, 0, 3, 2, 0, 3, 2, 4]).to(torch.int32).npu()
        idx_arr = torch.arange(tokens_num * topk_num).to(torch.int32).npu()
        inputs = {"topk": topk, "idx_arr": idx_arr}

        token_index = torch.empty(tokens_num * topk_num).to(torch.int32).npu()
        cum_sum = torch.empty(expert_num).to(torch.int32).npu()
        original_index = torch.empty(tokens_num * topk_num).to(torch.int32).npu()
        outputs = {"token_index": token_index, "cum_sum": cum_sum, "original_index": original_index}

        gating_engine.forward(inputs, outputs)
        golden_token_index, golden_cum_sum, golden_original_index = golden([topk_num, expert_num], [topk])

        torch.npu.synchronize()
        assert torch.allclose(golden_token_index.to(torch.int32), token_index, rtol=1e-02, atol=1e-02)
        assert torch.allclose(golden_cum_sum.to(torch.int32), cum_sum, rtol=1e-02, atol=1e-02)
        assert torch.allclose(golden_original_index.to(torch.int32), original_index, rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
    unittest.main()