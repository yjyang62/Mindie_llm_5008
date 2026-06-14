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

import atb_llm.nn as nn
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor


class TestActivationFunction(unittest.TestCase):
    def test_swish(self):
        def golden(x):
            return x * torch.nn.functional.sigmoid(x)

        swish_out_key = "swish_out"
        act_out = nn.functional.activation(input_=Tensor("input"), act_type=nn.functional.ActType.SWISH)
        get_default_net().mark_output(act_out, swish_out_key)
        swish_engine = get_default_net().build_engine()

        act_in = torch.rand(100, 1024).half().npu()
        act_out = torch.rand(100, 1024).half().npu()

        inputs = {}
        inputs["input"] = act_in
        outputs = {swish_out_key: act_out}
        swish_engine.forward(inputs, outputs)

        golden_score = golden(act_in)

        torch.npu.synchronize()
        assert torch.allclose(outputs[swish_out_key], golden_score, rtol=1e-02, atol=1e-02)
    
    def test_swiglu(self):
        def golden(x):
            a, b = x.chunk(2, dim=-1)
            a = a.to(torch.float32)
            b = b.to(torch.float32)
            y = torch.sigmoid(a) * a * b
            y = y.to(torch.float16)
            return y

        swiglu_out_key = "swiglu_out"
        act_out = nn.functional.activation(input_=Tensor("input"), act_type=nn.functional.ActType.SWIGLU)
        get_default_net().mark_output(act_out, swiglu_out_key)
        swish_engine = get_default_net().build_engine()

        act_in = torch.rand(100, 1024).half().npu()
        act_out = torch.rand(100, 512).half().npu()

        inputs = {}
        inputs["input"] = act_in
        outputs = {swiglu_out_key: act_out}
        swish_engine.forward(inputs, outputs)

        golden_score = golden(act_in)

        torch.npu.synchronize()
        assert torch.allclose(outputs[swiglu_out_key], golden_score, rtol=1e-02, atol=1e-02)

if __name__ == '__main__':
    unittest.main()