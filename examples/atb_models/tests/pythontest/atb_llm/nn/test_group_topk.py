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

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional import group_topk
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


class TestGroupTopkFunction(unittest.TestCase):
    def test_group_topk(self):
        def golden(tensor0, group_num, k):
            group_scores = (
                tensor0.view(tensor0.size(0), group_num, -1).max(dim=-1).values
            )
            group_idx = torch.topk(
                group_scores, k=k, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    tensor0.size(0), group_num, tensor0.size(1) // group_num
                )
                .reshape(tensor0.size(0), -1)
            )
            tmp_scores = tensor0.masked_fill(~score_mask.bool(), 0.0)

            return tmp_scores

        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This operation doesn't support Atlas 300I DUO.")

        group_num = 4
        k = 2

        group_topk(Tensor("input_tensor"), Tensor("index"), group_num, k)
        engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).half().npu()
        index = torch.linspace(0, 1023, 1024).to(torch.int32).npu()

        inputs = {}
        inputs['input_tensor'] = input_tensor
        inputs["index"] = index
        outputs = {}
        golden_out = golden(input_tensor, group_num, k)
        engine.forward(inputs, outputs)
        torch.npu.synchronize()

        self.assertTrue(torch.allclose(input_tensor, golden_out, rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()