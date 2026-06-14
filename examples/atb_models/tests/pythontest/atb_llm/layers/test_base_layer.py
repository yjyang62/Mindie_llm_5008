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

import torch

from atb_llm.layers.base_layer import BaseLayer
from atb_llm.nn.parameter import Parameter


class TestBaseLayer(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()

        self.weight_tool = MagicMock()
        self.weight_tool.mapping = MagicMock()
        self.weight_tool.mapping.rank = 0
        self.weight_tool.mapping.attn_tp = MagicMock()
        self.weight_tool.mapping.attn_tp.group_size = 1

        self.llm_config = MagicMock()
        self.llm_config.llm = MagicMock()
        self.llm_config.llm.weights_options = MagicMock()
        self.llm_config.llm.weights_options.low_cpu_memory_mode = True

        self.base_layer = BaseLayer(config=self.config, file_loader=self.weight_tool)

    def test_load_parameter(self):
        param = Parameter(prefix="fake_prefix", suffix="fake_suffix")
        self.base_layer.weight_loader = MagicMock(return_value=torch.tensor([1, 2]))
        self.base_layer.load_parameter(param, llm_config=self.llm_config)
        self.assertEqual(param.data.device, torch.device("npu:0"))


if __name__ == '__main__':
    unittest.main()