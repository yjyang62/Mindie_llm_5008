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
from atb_llm.nn.functional import neg


class TestNegFunction(unittest.TestCase):
    def test_neg(self):
        def golden(x):
            return torch.neg(x)

        neg_out = neg(Tensor("input"))
        get_default_net().mark_output(neg_out, "neg_out")
        neg_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).half().npu()
        neg_out = torch.rand(100, 1024).half().npu()

        inputs = {}
        inputs["input"] = input_tensor
        outputs = {"neg_out": neg_out}
        neg_engine.forward(inputs, outputs)

        golden_out = golden(input_tensor)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(outputs["neg_out"], golden_out, rtol=1e-02, atol=1e-02))

if __name__ == '__main__':
    unittest.main()