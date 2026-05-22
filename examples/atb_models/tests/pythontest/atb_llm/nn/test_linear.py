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
from ddt import ddt, data
import torch
import torch_npu

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.modules import Linear
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION

NPU = 'npu'
LINEAR_OUT = 'linear_out'


@ddt
class TestLinearModule(unittest.TestCase):
    @data(torch.float16, torch.bfloat16)
    def test_linear_without_bias(self, data_type):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and data_type == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")

        def golden(x, weight):
            return torch.matmul(x, weight)

        linear = Linear('weight', False)
        linear_out = linear(Tensor('input'))
        get_default_net().mark_output(linear_out, LINEAR_OUT)
        linear_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).to(data_type).to(NPU)
        weight = torch.rand(100, 1024).to(data_type).to(NPU)
        linear_out = torch.rand(100, 100).to(data_type).to(NPU)

        inputs = {}
        inputs["input"] = input_tensor
        outputs = {LINEAR_OUT: linear_out}
        linear_engine.set_weights({"weight.weight": weight})
        linear_engine.forward(inputs, outputs)

        golden_out = golden(input_tensor, weight.t())

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(outputs[LINEAR_OUT], golden_out, rtol=1e-02, atol=1e-02))
    
    @data(torch.float16, torch.bfloat16)
    def test_linear_with_bias(self, data_type):
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and data_type == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")

        def golden(x, weight, bias):
            return torch.matmul(x, weight) + bias
        
        linear = Linear('weight', True)
        linear_out = linear(Tensor('input'))
        get_default_net().mark_output(linear_out, LINEAR_OUT)
        linear_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).to(data_type).to(NPU)
        weight = torch.rand(100, 1024).to(data_type).to(NPU)
        bias = torch.rand(1, 100).to(data_type).to(NPU)
        linear_out = torch.rand(100, 100).to(data_type).to(NPU)

        inputs = {}
        inputs["input"] = input_tensor
        outputs = {LINEAR_OUT: linear_out}
        linear_engine.set_weights({"weight.weight": weight, "weight.bias": bias})
        linear_engine.forward(inputs, outputs)

        golden_out = golden(input_tensor, weight.t(), bias)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(outputs[LINEAR_OUT], golden_out, rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()