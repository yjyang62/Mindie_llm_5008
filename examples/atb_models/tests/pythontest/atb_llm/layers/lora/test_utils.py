#!/usr/bin/env python
# coding=utf-8
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

from atb_llm.nn.modules import Module
from atb_llm.layers.lora.lora_layers import ColumnParallelLinearWithLoRA, RowParallelLinearWithLoRA
from atb_llm.layers.lora.utils import from_layer, replace_submodule
from atb_llm.layers.linear.linear import ColumnParallelLinear, RowParallelLinear
from atb_llm.models.base.config import BaseConfig, LoraModelConfig
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.utils.quantize.quant_type import LinearTypeV2


class MockModelParam(MagicMock):
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.hf_config = MagicMock()
        self.hf_config.torch_dtype = torch.float16
        self.soc_info = MagicMock()
        self.soc_info.need_nz = False
        self.mapping = MagicMock()
        self.mapping.rank = 1
        self.mapping.world_size = 4
        self.lora_config = LoraModelConfig(max_loras=5, max_lora_rank=128)


class FakeModel(Module):
    def __init__(self, prefixes):
        super().__init__()
        config = BaseConfig(torch_dtype=torch.float16)
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        file_loader = weight_tool_cls()
        file_loader.mapping = MagicMock(spec=Mapping)
        file_loader.mapping.rank = 1
        file_loader.mapping.world_size = 4
        file_loader.get_sharded = MagicMock(return_value=torch.rand(128,128))
        file_loader.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        self.linear_layer_1 = ColumnParallelLinear(config, file_loader, prefixes, bias=False)
        self.linear_layer_2 = RowParallelLinear(config, file_loader, prefixes, bias=False)
        self.soc_info = MagicMock()
        self.soc_info.need_nz = False


class TestLoraUtils(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel(["linear"])
        self.device = torch.device("cpu")
        self.mindie_llm_config = MockModelParam()

    def test_replace_submodule(self):
        for module_name, module in self.model.named_modules(remove_duplicate=False):
            if isinstance(module, RowParallelLinear) or isinstance(module, ColumnParallelLinear):
                replace_submodule(
                    self.model, module_name,
                    from_layer(module, self.mindie_llm_config, self.device))
        self.assertIsInstance(self.model.linear_layer_1, ColumnParallelLinearWithLoRA)
        self.assertIsInstance(self.model.linear_layer_2, RowParallelLinearWithLoRA)


if __name__ == '__main__':
    unittest.main()