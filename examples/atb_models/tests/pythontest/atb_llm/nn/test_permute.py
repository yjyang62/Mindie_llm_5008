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


class TestPermuteFunction(unittest.TestCase):
    def test_permute(self):
        def golden(x, perm):
            return x.permute(perm)

        perm = (0, 2, 1, 3)
        perm_out = Tensor("input").permute(perm)
        get_default_net().mark_output(perm_out, "perm_out")
        neg_engine = get_default_net().build_engine()

        input_tensor = torch.rand(1, 2, 3, 4).half().npu()
        perm_out = torch.empty(1, 3, 2, 4).half().npu()

        inputs = {}
        inputs["input"] = input_tensor
        outputs = {"perm_out": perm_out}
        neg_engine.forward(inputs, outputs)

        golden_out = golden(input_tensor, perm)

        torch.npu.synchronize()
        assert torch.allclose(outputs["perm_out"], golden_out, rtol=1e-02, atol=1e-02)

if __name__ == '__main__':
    unittest.main()