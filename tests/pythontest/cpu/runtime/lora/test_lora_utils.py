# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import MagicMock, patch

import random
import torch
import torch.nn as nn

from mindie_llm.runtime.config.mindie_llm_config import LoraModelConfig
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelInfo
from mindie_llm.runtime.lora.utils import from_layer, replace_submodule
from mindie_llm.runtime.layers.linear.linear import (
    ColumnParallelLinear,
    RowParallelLinear
)
from mindie_llm.runtime.lora.lora_layers import (
    ParallelLinearWithLoRA,
    ColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA
)
from tests.pythontest.cpu.runtime.lora.test_lora_layers import (
    FakeColumnParallelLinear,
    FakeMergedColumnParallelLinear,
    FakeQKVParallelLinear,
    FakeRowParallelLinear
)


class FakeModel(nn.Module):
    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def __init__(self, mock_get_parallel_info_manager):
        super().__init__()
        world_size = 2 ** random.randint(0, 2)
        tp_rank = random.randint(0, world_size - 1)
        parallel_info = ParallelInfo()
        parallel_info.group_size = world_size
        parallel_info.rank = tp_rank
        mock_parallel_info_manager = MagicMock()
        mock_parallel_info_manager.rank = parallel_info.rank
        mock_parallel_info_manager.world_size = parallel_info.group_size
        mock_get_parallel_info_manager.return_value = mock_parallel_info_manager
        self.linear_layer_1 = FakeColumnParallelLinear(["linear"], parallel_info)
        self.linear_layer_2 = FakeMergedColumnParallelLinear(["gate", "up"], parallel_info)
        self.linear_layer_3 = FakeQKVParallelLinear(["q", "k", "v"], parallel_info)
        self.linear_layer_4 = FakeRowParallelLinear(["linear"], parallel_info)
        self.soc_info = MagicMock()
        self.soc_info.need_nz = False


class TestLoraUtils(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.device = torch.device("cpu")
        self.lora_model_config = LoraModelConfig(max_loras=5, max_lora_rank=128)
        self.dtype = torch.float16
        self.soc_info = MagicMock()
        self.soc_info.need_nz = False

    @patch.object(ParallelLinearWithLoRA, "weight_format_cast")
    def test_replace_submodule(self, mock_weight_format_cast):
        mock_weight_format_cast.side_effect = lambda x: x
        for module_name, module in self.model.named_modules(remove_duplicate=False):
            if isinstance(module, RowParallelLinear) or isinstance(module, ColumnParallelLinear):
                replace_submodule(
                    self.model, module_name,
                    from_layer(module, self.lora_model_config, self.dtype, self.device))
        self.assertIsInstance(self.model.linear_layer_1, ColumnParallelLinearWithLoRA)
        self.assertIsInstance(self.model.linear_layer_2, ColumnParallelLinearWithLoRA)
        self.assertIsInstance(self.model.linear_layer_3, ColumnParallelLinearWithLoRA)
        self.assertIsInstance(self.model.linear_layer_4, RowParallelLinearWithLoRA)


if __name__ == '__main__':
    unittest.main()