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
from unittest.mock import Mock, MagicMock, patch
import random

import torch
from ddt import ddt, data, unpack

from mindie_llm.runtime.config.mindie_llm_config import LoraModelConfig
from mindie_llm.runtime.config.lora_config import LoraConfig
from mindie_llm.runtime.lora.lora_manager import LoraManager, AdapterInfo
import mindie_llm.runtime.lora.lora_manager as lora_manager_module


BASE_ADAPTER_NAME = "base"
ADAPTER1_STR = "adapter1"
ADAPTER2_STR = "adapter2"
ADAPTER3_STR = "adapter3"
ADAPTER4_STR = "adapter4"
ADAPTER5_STR = "adapter5"


@ddt
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
        
    def test_get_lora_slots(self):
        self.assertEqual(self.adapter_manager.lora_slots, len(self.lora_adapter))
        
    @data(("model.layer.0.attention.qkv", "attention.qkv"),
          ("transformers.model.layer.25.mlp.down", "mlp.down"))
    @unpack
    def test_get_last_two_prefix(self, prefix, expect_prefix):
        prefix = self.adapter_manager.get_last_two_prefix(prefix)
        self.assertEqual(prefix, expect_prefix)
        
    def test_get_r(self):
        r = self.adapter_manager._get_r(LoraConfig(r=4, lora_alpha=64), prefix="")
        self.assertEqual(r, 4)
        pattern_key = "model.layer.0.attention.qkv"
        r = self.adapter_manager._get_r(LoraConfig(
            r=4,
            lora_alpha=64,
            rank_pattern={"attention.qkv": 8},
            alpha_pattern={"attention.qkv": 16}),
            prefix=pattern_key)
        self.assertEqual(r, 8)

    def test_get_alpha(self):
        alpha = self.adapter_manager._get_alpha(LoraConfig(r=4, lora_alpha=64), prefix="")
        self.assertEqual(alpha, 64)
        pattern_key = "model.layer.0.attention.qkv"
        alpha = self.adapter_manager._get_alpha(LoraConfig(
            r=4,
            lora_alpha=64,
            rank_pattern={"attention.qkv": 8},
            alpha_pattern={"attention.qkv": 16}),
            prefix=pattern_key)
        self.assertEqual(alpha, 16)

    def test_get_scaling(self):
        scaling = self.adapter_manager._get_scaling(LoraConfig(r=4, lora_alpha=64), prefix="")
        self.assertEqual(scaling, 16)
        pattern_key = "model.layer.0.attention.qkv"
        scaling = self.adapter_manager._get_scaling(LoraConfig(
            r=4,
            lora_alpha=64,
            rank_pattern={"attention.qkv": 8},
            alpha_pattern={"attention.qkv": 16}),
            prefix=pattern_key)
        self.assertEqual(scaling, 2)
        scaling = self.adapter_manager._get_scaling(LoraConfig(r=4, lora_alpha=64, use_rslora=True), prefix="")
        self.assertEqual(scaling, 32)

    @patch.object(LoraManager, "_find_lora_module")
    @patch.object(LoraManager, "load_lora_config")
    def test_update_max_lora_rank(self, mock_load_lora_config, mock_find_lora_module):
        self._clear_adapter_ids_registry()
        mock_load_lora_config.return_value = LoraConfig(r=4, lora_alpha=64, rank_pattern={"attention.qkv": 8},
            alpha_pattern={"attention.qkv": 16})
        self.adapter_manager.lora_model_config.max_lora_rank = 0
        mock_module = Mock()
        mock_module.prefix = ["model.layer.0.attention.qkv"]
        mock_find_lora_module.return_value = [("module", mock_module)]
        self.adapter_manager._update_max_lora_rank({ADAPTER1_STR: "fake_adapter_1_path"})
        self.assertEqual(self.adapter_manager.lora_model_config.max_lora_rank, 8)
        
    @data((None, [BASE_ADAPTER_NAME]), ([None, None], [BASE_ADAPTER_NAME]),
          ([ADAPTER1_STR, ADAPTER1_STR], [ADAPTER1_STR]),
          ([ADAPTER3_STR], [ADAPTER3_STR]), ([ADAPTER1_STR, ADAPTER2_STR], [ADAPTER1_STR, ADAPTER2_STR]))
    @unpack
    def test_preprocess_adapter_ids(self, adapter_ids, expected_adapter_ids):
        self._update_adapter_ids_registry()
        effective_adapter_ids = self.adapter_manager.preprocess_adapter_ids(adapter_ids)
        self.assertEqual(effective_adapter_ids, expected_adapter_ids)
        
    @data(([BASE_ADAPTER_NAME], [ADAPTER1_STR], True), ([BASE_ADAPTER_NAME], [BASE_ADAPTER_NAME], False),
          ([ADAPTER1_STR, BASE_ADAPTER_NAME], [ADAPTER1_STR, ADAPTER2_STR], False),
          ([ADAPTER1_STR, BASE_ADAPTER_NAME], [ADAPTER2_STR, ADAPTER1_STR], True))
    @unpack
    def test_update_adapter_check_return_value(self, previous_adapter_ids, adapter_ids, expected_result):
        self._update_adapter_ids_registry()
        self.adapter_manager.update_adapter(previous_adapter_ids)
        need_update = self.adapter_manager.update_adapter(adapter_ids)
        self.assertEqual(need_update, expected_result)

    @data((None, True), ([None, None], True), ([ADAPTER1_STR, ADAPTER2_STR], True),
         ([ADAPTER2_STR, ADAPTER1_STR], False), ([BASE_ADAPTER_NAME, ADAPTER1_STR], False),
         ([ADAPTER1_STR, ADAPTER3_STR], True))
    @unpack
    def test_check_adapter_ids_is_sorted(self, adapter_ids, expected_result):
        self._update_adapter_ids_registry()
        actual_result = self.adapter_manager.check_adapter_ids_is_sorted(adapter_ids)
        self.assertEqual(actual_result, expected_result)

    def test_sort_adapter_ids(self):
        self._update_adapter_ids_registry()
        candidates = [ADAPTER1_STR, ADAPTER2_STR, ADAPTER3_STR, ADAPTER4_STR, ADAPTER5_STR]
        for i, item in enumerate(candidates):
            self.adapter_manager.adapter_info_registry[item] = AdapterInfo(idx=i, adapter_path="")
        adapter_ids = [ADAPTER2_STR, ADAPTER4_STR, ADAPTER5_STR, ADAPTER5_STR, ADAPTER1_STR]
        sorted_adapter_idx, revert_adapter_idx = self.adapter_manager.sort_adapter_ids(adapter_ids)
        self.assertTrue(sorted_adapter_idx, [4, 0, 1, 2, 3])
        self.assertTrue(revert_adapter_idx, [1, 2, 3, 4, 0])
    
    def _load_dummy_adapter(self):
        for _, module in self.lora_modules.items():
            n, k = module.base_weight_shape
            dim_r = 16 if self.base_model.soc_info.need_nz else 64
            lora_a = torch.zeros([dim_r, k], dtype=module.dtype)
            lora_b = torch.zeros([dim_r, n], dtype=module.dtype)
            module.set_lora(self.max_loras, lora_a, lora_b)
        # register_adapter
        self.adapter_info_registry[BASE_ADAPTER_NAME] = AdapterInfo(
            idx=self.max_loras, adapter_path="", config=LoraConfig(r=1, lora_alpha=1, use_rslora=False))
    
    @patch.object(LoraManager, "_create_lora_modules")
    @patch.object(LoraManager, "_load_adapter")
    def test_preload_adapter(self, mock_load_adapter, mock_create_lora_modules):
        self._clear_adapter_ids_registry()
        mock_load_adapter.return_value = Mock()
        mock_create_lora_modules.return_value = Mock()
        mock_module = Mock()
        mock_module.base_weight_shape = (128, 128)
        mock_module.dtype = torch.float16
        mock_module.set_lora = Mock()
        self.adapter_manager.lora_modules = {"module": mock_module}
        self.adapter_manager.preload_adapter({ADAPTER1_STR: "fake_adapter_1_path"})
        self.assertNotEqual(self.adapter_manager.adapter_info_registry[BASE_ADAPTER_NAME], None)
    
    @patch("safetensors.torch.safe_open")
    @patch.object(LoraManager, "_get_scaling")
    @patch.object(LoraManager, "load_lora_config")
    def test_load_adapter(self, mock_load_lora_config, mock_get_scaling, mock_safetensors_torch_safe_open):
        self._clear_adapter_ids_registry()
        mock_load_lora_config.return_value = LoraConfig(r=4, lora_alpha=64)
        mock_get_scaling.return_value = 0
        lora_manager_module.standardize_path = Mock(side_effect=self._mock_standardize_path)
        lora_manager_module.check_file_safety = Mock()
        mock_file = Mock()
        mock_file.keys.return_value = ["linear.weight"]
        mock_file.get_tensor.side_effect = torch.rand(256, 1024, device=self.device, dtype=self.dtype)
        mock_safetensors_torch_safe_open.return_value.__enter__.return_value = mock_file
        mock_module = Mock()
        mock_module.base_layer_prefixes = ["linear"]
        mock_module.slice_lora_a = Mock(return_value=torch.Tensor([]))
        mock_module.slice_lora_b = Mock(return_value=torch.Tensor([]))
        mock_module.set_lora = Mock()
        self.adapter_manager.lora_modules = {"module": mock_module}
        self.adapter_manager.load_adapter({ADAPTER1_STR: "fake_adapter_1_path"})
        self.assertTrue(self.adapter_manager.lora_slots_occupied[0])

    def test_load_adapter_duplicate(self):
        self._clear_adapter_ids_registry()
        self.adapter_manager.adapter_info_registry = {
            ADAPTER1_STR: AdapterInfo(
                idx=0, adapter_path="fake_adapter_1_path")
        }
        with self.assertRaises(ValueError):
            self.adapter_manager.load_adapter({ADAPTER1_STR: "fake_adapter_1_path"})

    def test_add_adapter_invalid_number(self):
        self._clear_adapter_ids_registry()
        with self.assertRaises(RuntimeError):
            self.adapter_manager.load_adapter(dict())

    def test_add_adapter_invalid_id_length(self):
        self._clear_adapter_ids_registry()
        with self.assertRaises(ValueError):
            self.adapter_manager.load_adapter({"": "path1"})

    def test_add_adapter_id_exists(self):
        self._clear_adapter_ids_registry()
        self.adapter_manager.adapter_info_registry = {
            ADAPTER2_STR: AdapterInfo(idx=0, adapter_path="fake_adapter_2_path"),
            ADAPTER1_STR: AdapterInfo(idx=1, adapter_path="fake_adapter_1_path")
        }
        with self.assertRaises(ValueError):
            self.adapter_manager.load_adapter({ADAPTER1_STR: "path1"})

    def test_add_adapter_slots_full(self):
        self._update_adapter_ids_registry()
        with self.assertRaises(RuntimeError):
            self.adapter_manager.load_adapter({"id1": "path1"})

    def test_unload_adapter_not_found(self):
        self._update_adapter_ids_registry()
        with self.assertRaises(RuntimeError):
            self.adapter_manager.unload_adapter("id1")

    def test_unload_adapter(self):
        self._update_adapter_ids_registry()
        mock_module = Mock()
        mock_module.reset_lora = MagicMock()
        self.adapter_manager.lora_modules = {"module": mock_module}
        self.adapter_manager.unload_adapter(ADAPTER1_STR)
        self.assertNotIn(ADAPTER1_STR, self.adapter_manager.adapter_info_registry)
        self.assertFalse(self.adapter_manager.lora_slots_occupied[0])
    
    def _clear_adapter_ids_registry(self):
        self.adapter_manager.adapter_info_registry.clear()
        self.adapter_manager.lora_slots_occupied = [False for _ in range(self.adapter_manager.lora_slots + 1)]

    def _update_adapter_ids_registry(self):
        for adapter_id, adapter_path in self.lora_adapter.items():
            self.adapter_manager.adapter_info_registry[adapter_id] = AdapterInfo(
                idx=len(self.adapter_manager.adapter_info_registry), adapter_path=adapter_path)
        self.adapter_manager.adapter_info_registry[BASE_ADAPTER_NAME] = AdapterInfo(
            idx=len(self.adapter_manager.adapter_info_registry), adapter_path="")
        for i, _ in enumerate(self.adapter_manager.lora_slots_occupied):
            self.adapter_manager.lora_slots_occupied[i] = True
    
    def _mock_standardize_path(self, path, check_link=True):
        return path


if __name__ == '__main__':
    unittest.main()