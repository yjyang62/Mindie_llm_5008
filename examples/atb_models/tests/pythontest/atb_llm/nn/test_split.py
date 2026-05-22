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
from atb_llm.nn.functional import split


class TestSplitFunction(unittest.TestCase):
    def test_dim1_num3(self):
        out1, out2, out3 = split(Tensor("tensor"), 3, 1)
        get_default_net().mark_output(out1, "out1")
        get_default_net().mark_output(out2, "out2")
        get_default_net().mark_output(out3, "out3")
        split_engine = get_default_net().build_engine()

        split_tensor = torch.rand(1000, 3000).half().npu()
        out1 = torch.empty(1000, 1000).half().npu()
        out2 = torch.empty(1000, 1000).half().npu()
        out3 = torch.empty(1000, 1000).half().npu()

        inputs = {"tensor": split_tensor}
        outputs = {"out1": out1, "out2": out2, "out3": out3}
        split_engine.forward(inputs, outputs)

        goldens = torch.split(split_tensor, dim=1, split_size_or_sections=1000)

        assert torch.allclose(out1, goldens[0], rtol=1e-02, atol=1e-02)
        assert torch.allclose(out2, goldens[1], rtol=1e-02, atol=1e-02)
        assert torch.allclose(out3, goldens[2], rtol=1e-02, atol=1e-02)


if __name__ == '__main__':
    unittest.main()