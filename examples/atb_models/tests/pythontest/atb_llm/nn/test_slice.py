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


class TestSliceFunction(unittest.TestCase):
    def test_slice_dim1(self):
        slice_tensor = Tensor("input")[:, :100]
        get_default_net().mark_output(slice_tensor, "slice_tensor")
        cos_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).half().npu()
        slice_tensor = torch.rand(100, 100).half().npu()

        inputs = {}
        inputs["input"] = input_tensor
        outputs = {"slice_tensor": slice_tensor}
        cos_engine.forward(inputs, outputs)

        golden_out = input_tensor[:, :100]

        torch.npu.synchronize()
        assert torch.allclose(slice_tensor, golden_out, rtol=1e-02, atol=1e-02)

if __name__ == '__main__':
    unittest.main()