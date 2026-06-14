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
from unittest.mock import MagicMock

from atb_llm.layers.mlp.mlp import Mlp 
from atb_llm.nn.tensor import Tensor


class TestMlp(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()

        self.weight_tool = MagicMock()
        self.weight_tool.mapping = MagicMock()
        self.weight_tool.mapping.rank = 0
        self.weight_tool.mapping.mlp_tp = MagicMock()
        self.weight_tool.mapping.mlp_tp.group_size = 1
        self.weight_tool.mapping.mlp_tp.process_group = '-1'

        self.mlp = Mlp(self.config, self.weight_tool, "test", MagicMock())
        self.mlp.gate_up = MagicMock(return_value=[Tensor("test1"), Tensor("test2")])
        self.mlp.down = MagicMock(return_value=Tensor("testd"))

    def test_forward_gate_up_nd(self):
        out = self.mlp(Tensor("inputs"))
        self.assertIsInstance(out, Tensor)

    def test_forward_up_only_nz(self):
        self.mlp.gate_up = MagicMock(return_value=[Tensor("test1")])
        self.mlp.gate_up.__len__.return_value = 1
        self.mlp.need_nz = True
        out = self.mlp(Tensor("inputs"))
        self.assertIsInstance(out, Tensor)


if __name__ == '__main__':
    unittest.main()