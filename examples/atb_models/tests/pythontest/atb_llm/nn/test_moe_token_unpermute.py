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

from atb_llm.nn.functional.moe import moe_token_unpermute
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


def golden(params, inputs):
    tokens_num = params[0]
    topk_num = params[1]
    hidden_size = params[2]
    permuted_tokens = inputs[0]
    sorted_indices = inputs[1]
    experts_weights = inputs[2]

    golden_out = torch.zeros(tokens_num, hidden_size).to(torch.float16).npu()
    for i in range(tokens_num):
        for j in range(topk_num):
            index = sorted_indices[i * topk_num + j]
            golden_out[i, :] = golden_out[i, :] + permuted_tokens[index, :] * experts_weights[i, j]
    return golden_out


class TestMoeTokenUnpermuteFunction(unittest.TestCase):
    def test_moe_token_unpermute(self):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This operation doesn't support Atlas 300I DUO.")

        tokens_num = 20
        topk_num = 4
        hidden_size = 1000

        out = moe_token_unpermute(Tensor("permuted_tokens"), Tensor("sorted_indices"), Tensor("experts_weights"))
        get_default_net().mark_output(out, "out")
        unpermute_engine = get_default_net().build_engine()

        permuted_tokens = torch.rand(tokens_num * topk_num, hidden_size).to(torch.float16).npu()
        sorted_indices = torch.arange(tokens_num * topk_num).to(torch.int32).npu()
        experts_weights = torch.rand(tokens_num, topk_num).to(torch.float16).npu()
        inputs = {"permuted_tokens": permuted_tokens, "sorted_indices": sorted_indices,
                  "experts_weights": experts_weights}

        out = torch.empty(tokens_num, hidden_size).to(torch.float16).npu()
        outputs = {"out": out}

        unpermute_engine.forward(inputs, outputs)
        golden_out = golden([tokens_num, topk_num, hidden_size], [permuted_tokens, sorted_indices, experts_weights])

        torch.npu.synchronize()
        assert torch.allclose(golden_out, out, rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
    unittest.main()