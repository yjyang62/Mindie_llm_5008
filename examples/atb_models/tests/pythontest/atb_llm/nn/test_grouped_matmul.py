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

from atb_llm.nn.functional import grouped_matmul
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.network_manager import get_default_net


def golden(x, weight, group_list):
    result = []
    start = 0
    for i in range(group_list.shape[0]):
        end = group_list[i]
        if start == end:
            continue
        result.append(torch.matmul(x[start:end, :], weight[i, :, :]))
        start = end
    return torch.cat(result, dim=0)


class TestGroupedMatmulFunction(unittest.TestCase):
    def test_grouped_matmul(self):
        out = grouped_matmul(Tensor("x"), Tensor("weight"), None, Tensor("group_list"))
        get_default_net().mark_output(out, "out")
        grouped_matmul_engine = get_default_net().build_engine()

        x = torch.rand(20, 1000).to(torch.float16).npu()
        weight = torch.rand(8, 1000, 2000).to(torch.float16).npu()
        group_list = torch.tensor([0, 0, 5, 10, 10, 14, 17, 20]).to(torch.int64).npu()
        inputs = {"x": x, "weight": weight, "group_list": group_list}
        out = torch.empty(20, 2000).to(torch.float16).npu()
        outputs = {"out": out}

        grouped_matmul_engine.forward(inputs, outputs)
        golden_out = golden(x, weight, group_list)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(golden_out, out, rtol=1e-05, atol=1e-05))


if __name__ == '__main__':
    unittest.main()