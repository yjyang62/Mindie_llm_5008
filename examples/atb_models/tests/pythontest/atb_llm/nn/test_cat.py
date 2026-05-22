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

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional import cat


class TestCatFunction(unittest.TestCase):
    def test_cat_dim0(self):
        def golden(x, y):
            return torch.cat((x, y), dim=0)

        cat_out = cat([Tensor("x"), Tensor("y")], dim=0)
        get_default_net().mark_output(cat_out, "cat_out")
        cos_engine = get_default_net().build_engine()

        x = torch.rand(100, 1024).half().npu()
        y = torch.rand(100, 1024).half().npu()
        cat_out = torch.empty(200, 1024).half().npu()

        inputs = {}
        inputs["x"] = x
        inputs["y"] = y
        outputs = {"cat_out": cat_out}
        cos_engine.forward(inputs, outputs)

        golden_out = golden(x, y)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(outputs["cat_out"], golden_out, rtol=1e-02, atol=1e-02))

if __name__ == '__main__':
    unittest.main()