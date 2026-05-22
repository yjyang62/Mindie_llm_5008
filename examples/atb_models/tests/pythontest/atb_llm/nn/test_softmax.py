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

from atb_llm.nn.functional import softmax
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor


def golden(inputs, axes):
    input_tensor = inputs[0]
    if len(axes) == 1:
        torch_softmax = torch.nn.Softmax(dim=axes[0])
        return torch_softmax(input_tensor)
    else:
        start_dim = axes[0]
        dim_num = len(axes)
        original_shape = input_tensor.shape
        merged_shape = original_shape[:start_dim] + (-1,) + original_shape[start_dim + dim_num:]
        input_merged = input_tensor.view(merged_shape)
        torch_softmax = torch.nn.Softmax(dim=start_dim)
        return torch_softmax(input_merged).view(original_shape)


class TestSoftmaxFunction(unittest.TestCase):
    def test_softmax_single_dim(self):
        axes = [-1]
        out = softmax(Tensor("input_tensor"), axes)
        get_default_net().mark_output(out, "out")
        softmax_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024).to(torch.float16).npu()
        inputs = {"input_tensor": input_tensor}
        out = torch.empty(100, 1024).to(torch.float16).npu()
        outputs = {"out": out}

        softmax_engine.forward(inputs, outputs)
        golden_out = golden([input_tensor], axes)

        torch.npu.synchronize()
        assert torch.allclose(golden_out, out, rtol=1e-02, atol=1e-02)

    def test_softmax_multi_dim(self):
        axes = [1, 2]
        out = softmax(Tensor("input_tensor"), axes)
        get_default_net().mark_output(out, "out")
        softmax_engine = get_default_net().build_engine()

        input_tensor = torch.rand(100, 1024, 100, 100).to(torch.float16).npu()
        inputs = {"input_tensor": input_tensor}
        out = torch.empty(100, 1024, 100, 100).to(torch.float16).npu()
        outputs = {"out": out}

        softmax_engine.forward(inputs, outputs)
        golden_out = golden([input_tensor], axes)

        torch.npu.synchronize()
        assert torch.allclose(golden_out, out, rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
    unittest.main()