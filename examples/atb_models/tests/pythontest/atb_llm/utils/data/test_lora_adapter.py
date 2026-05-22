#!/usr/bin/env python
# coding=utf-8
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
from torch.nn.utils.rnn import pad_sequence

from atb_llm.utils.data.lora_adapter import ParallelLinearWithLoRAAdaptee, LoraManager
from atb_llm.utils.data.layer_adapter import ColumnParallelLinear
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.quantize.pack_type import TransposeType
from mindie_llm.runtime.config.mindie_llm_config import LoraModelConfig
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelInfo
from mindie_llm.runtime.lora.lora_manager import AdapterInfo
from mindie_llm.runtime.lora.lora_layers import ParallelLinearWithLoRA, ColumnParallelLinearWithLoRA


BASE_ADAPTER_NAME = "base"
ADAPTER1_STR = "adapter1"
ADAPTER2_STR = "adapter2"
ADAPTER3_STR = "adapter3"
ADAPTER4_STR = "adapter4"
ADAPTER5_STR = "adapter5"


class FakeColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, prefix, parallel_info):
        input_size = random.randint(1, 1024)
        output_size = parallel_info.group_size * random.randint(1, 1024)
        weight_dtype = torch.float16
        self.parallel_info = parallel_info
        super().__init__(input_size=input_size, output_size=output_size,
            weight_dtype=weight_dtype, prefix=prefix, parallel_info=parallel_info)


class TestLoraManager(unittest.TestCase):
    def setUp(self):
        self.r = 4 ** random.randint(1, 6)
        self.lora_adapter = {ADAPTER1_STR: "fake_adapter_1_path", ADAPTER2_STR: "fake_adapter_2_path",
                             ADAPTER3_STR: "fake_adapter_3_path", ADAPTER4_STR: "fake_adapter_4_path",
                             ADAPTER5_STR: "fake_adapter_5_path"}
        self.max_loras = len(self.lora_adapter)
        self.max_lora_rank = 16 * self.r
        self.dtype = torch.float16
        self.device = torch.device("cpu")
        self.rank = 1
        self.world_size = 4
        base_model = MagicMock()
        base_model.dtype = self.dtype
        base_model.device = self.device
        base_model.mapping = MagicMock()
        base_model.mapping.rank = self.rank
        base_model.mapping.world_size = self.world_size
        base_model.soc_info = MagicMock()
        base_model.soc_info.need_nz = False
        lora_model_config = LoraModelConfig(max_loras=self.max_loras, max_lora_rank=self.max_lora_rank)
        self.adapter_manager = LoraManager(base_model, lora_model_config)
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = self.rank
        self.mock_parallel_info_manager.world_size = self.world_size
        self.parallel_info = ParallelInfo()
        self.parallel_info.group_size = self.world_size
        self.parallel_info.rank = self.rank

    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def testParallelLinearWithLoRAAdaptee(self, mock_get_parallel_info_manager):
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeColumnParallelLinear(["linear"], self.parallel_info)
        linear_layer.weight.data = torch.randn((64, 64), dtype=self.dtype, device='cpu')
        linear_layer.bias.data = torch.randn(64, dtype=self.dtype, device='cpu')
        linear_layer.quant_method = MagicMock()
        linear_layer.quant_method.get_weights_for_atb_graph.return_value = [torch.tensor([1.0], device='cpu')]
        desc = LinearTypeV2.FLOAT16
        linear_layer.quant_method.get_linear_descs.return_value = desc
        linear_layer.quant_method.get_weight_transpose_type.return_value = TransposeType.NOT_TRANSPOSE
        linear_layer._PLACEHOLDER = torch.tensor([999], dtype=torch.float32, device='cpu')
        lora_module = ColumnParallelLinearWithLoRA(linear_layer)
        lora_module_adaptee = ParallelLinearWithLoRAAdaptee(lora_module)
        weights = lora_module_adaptee.get_weights_for_atb_graph(padding=True)
        self.assertEqual(len(weights), 1)
        with self.assertRaises(ValueError) as cm:
            lora_module_adaptee.get_weights_for_atb_graph(is_swiglu_quant_enabled=True)
        self.assertIn("Cannot set `is_swiglu_quant_enabled` to True", str(cm.exception))
        descs = lora_module_adaptee.get_linear_descs()
        trans = lora_module_adaptee.get_weight_transpose_type()
        self.assertEqual(descs, [LinearTypeV2.FLOAT16])
        self.assertEqual(trans, [TransposeType.NOT_TRANSPOSE])
    
    @patch.object(ParallelLinearWithLoRA, "weight_format_cast")
    @patch.object(LoraManager, "_find_lora_module")
    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_get_adapters_single_adapter(self, mock_get_parallel_info_manager, \
            mock_find_lora_module, mock_weight_format_cast):
        self._update_adapter_ids_registry()
        self.adapter_manager.lora_modules = dict()
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeColumnParallelLinear(["linear"], self.parallel_info)
        n, k = sum(linear_layer.output_partition_sizes), linear_layer.input_size_per_partition
        mock_find_lora_module.return_value = [("linear", linear_layer)]
        mock_weight_format_cast.side_effect = lambda x: x
        self.adapter_manager._create_lora_modules()
        lora_module = self.adapter_manager.lora_modules[list(self.adapter_manager.lora_modules.keys())[0]]
        fake_lora_a_tensor_1 = torch.rand((self.max_lora_rank, k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((n, self.max_lora_rank), dtype=self.dtype).T.contiguous()
        lora_module.lora_a_stacked.data[0].copy_(fake_lora_a_tensor_1)
        lora_module.lora_b_stacked.data[0].copy_(fake_lora_b_tensor_1)
        self.adapter_manager.update_adapter([ADAPTER1_STR])
        adapter_weights = self.adapter_manager.get_adapters([ADAPTER1_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.allclose(adapter_weights[0].cpu(), fake_lora_a_tensor_1))
        self.assertTrue(torch.allclose(adapter_weights[1].cpu(), fake_lora_b_tensor_1))

    @patch.object(ParallelLinearWithLoRA, "weight_format_cast")
    @patch.object(LoraManager, "_find_lora_module")
    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_get_adapters_mixed_adapter(self, mock_get_parallel_info_manager, \
            mock_find_lora_module, mock_weight_format_cast):
        self._update_adapter_ids_registry()
        self.adapter_manager.lora_modules = dict()
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeColumnParallelLinear(["linear"], self.parallel_info)
        n, k = sum(linear_layer.output_partition_sizes), linear_layer.input_size_per_partition
        mock_find_lora_module.return_value = [("linear", linear_layer)]
        mock_weight_format_cast.side_effect = lambda x: x
        self.adapter_manager._create_lora_modules()
        lora_module = self.adapter_manager.lora_modules[list(self.adapter_manager.lora_modules.keys())[0]]
        fake_lora_a_tensor_1 = torch.rand((self.max_lora_rank, k), dtype=self.dtype)
        fake_lora_a_tensor_2 = torch.rand((self.max_lora_rank, k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((n, self.max_lora_rank), dtype=self.dtype).T.contiguous()
        fake_lora_b_tensor_2 = torch.rand((n, self.max_lora_rank), dtype=self.dtype).T.contiguous()
        lora_module.lora_a_stacked.data[0].copy_(fake_lora_a_tensor_1)
        lora_module.lora_b_stacked.data[0].copy_(fake_lora_b_tensor_1)
        lora_module.lora_a_stacked.data[1].copy_(fake_lora_a_tensor_2)
        lora_module.lora_b_stacked.data[1].copy_(fake_lora_b_tensor_2)
        self.adapter_manager.update_adapter([ADAPTER2_STR, ADAPTER1_STR])
        adapter_weights = self.adapter_manager.get_adapters([ADAPTER2_STR, ADAPTER1_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.allclose(
            adapter_weights[0].cpu(),
            pad_sequence([fake_lora_a_tensor_2, fake_lora_a_tensor_1], batch_first=True)
        ))
        self.assertTrue(torch.allclose(
            adapter_weights[1].cpu(),
            pad_sequence([fake_lora_b_tensor_2, fake_lora_b_tensor_1], batch_first=True)
        ))

    @patch.object(ParallelLinearWithLoRA, "weight_format_cast")
    @patch.object(LoraManager, "_find_lora_module")
    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_get_adapters_sorted_adapter(self, mock_get_parallel_info_manager, \
            mock_find_lora_module, mock_weight_format_cast):
        self._update_adapter_ids_registry()
        self.adapter_manager.lora_modules = dict()
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeColumnParallelLinear(["linear"], self.parallel_info)
        n, k = sum(linear_layer.output_partition_sizes), linear_layer.input_size_per_partition
        mock_find_lora_module.return_value = [("linear", linear_layer)]
        mock_weight_format_cast.side_effect = lambda x: x
        self.adapter_manager._create_lora_modules()
        lora_module = self.adapter_manager.lora_modules[list(self.adapter_manager.lora_modules.keys())[0]]
        fake_lora_a = torch.rand((self.max_loras + 1, self.max_lora_rank, k), dtype=self.dtype)
        fake_lora_b = torch.rand((self.max_loras + 1, self.max_lora_rank, n), dtype=self.dtype)
        lora_module.lora_a_stacked.data.copy_(fake_lora_a)
        lora_module.lora_b_stacked.data.copy_(fake_lora_b)
        self.adapter_manager.update_adapter([ADAPTER1_STR, ADAPTER2_STR])
        adapter_weights = self.adapter_manager.get_adapters([ADAPTER1_STR, ADAPTER2_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.allclose(
            adapter_weights[0].cpu(), fake_lora_a))
        self.assertTrue(torch.allclose(
            adapter_weights[1].cpu(), fake_lora_b))

    def _update_adapter_ids_registry(self):
        for adapter_id, adapter_path in self.lora_adapter.items():
            self.adapter_manager.adapter_info_registry[adapter_id] = AdapterInfo(
                idx=len(self.adapter_manager.adapter_info_registry), adapter_path=adapter_path)
        self.adapter_manager.adapter_info_registry[BASE_ADAPTER_NAME] = AdapterInfo(
            idx=len(self.adapter_manager.adapter_info_registry), adapter_path="")
        for i, _ in enumerate(self.adapter_manager.lora_slots_occupied):
            self.adapter_manager.lora_slots_occupied[i] = True


if __name__ == "__main__":
    unittest.main()