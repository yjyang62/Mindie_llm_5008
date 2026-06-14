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
import atb_llm.nn as nn


class TestRmsnormModule(unittest.TestCase):
    def test_rmsnorm_without_bias_f16(self):
        def golden(x, weight, eps):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
            if torch.any(rms == 0):
                return 0
            x = x / rms
            return weight * x

        eps = 1e-06
        rmsnorm = nn.modules.RmsNorm('weight', eps)
        rmsnorm_out = rmsnorm(Tensor('input'))
        get_default_net().mark_output(rmsnorm_out, "rmsnorm_out")
        cos_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).half().to('npu')
        weight = torch.rand(1024).half().to('npu')
        rmsnorm_out = torch.rand(100, 1024).half().to('npu')

        inputs = {}
        inputs["input"] = input_tensor
        outputs = {"rmsnorm_out": rmsnorm_out}
        cos_engine.set_weights({"weight.weight": weight})
        cos_engine.forward(inputs, outputs)

        golden_out = golden(input_tensor, weight, eps)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(outputs["rmsnorm_out"], golden_out, rtol=1e-02, atol=1e-02))

if __name__ == '__main__':
    unittest.main()