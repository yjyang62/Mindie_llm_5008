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

from atb_llm.nn.functional import sort
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor


def golden(x):
    return torch.sort(x, descending=True)


class TestSortFunction(unittest.TestCase):
    def test_sort(self):
        num = 10
        out, indices = sort(Tensor("x"), num)
        get_default_net().mark_output(out, "out")
        get_default_net().mark_output(indices, "indices")
        sort_engine = get_default_net().build_engine()

        x = torch.rand(num).to(torch.float16).npu()
        inputs = {"x": x}
        out = torch.empty(num).to(torch.float16).npu()
        indices = torch.empty(num).to(torch.int32).npu()
        outputs = {"out": out, "indices": indices}

        sort_engine.forward(inputs, outputs)
        golden_out, golden_indices = golden(x)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(golden_out, out, rtol=1e-02, atol=1e-02))
        self.assertTrue(torch.allclose(golden_indices.to(torch.int32), indices, rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()