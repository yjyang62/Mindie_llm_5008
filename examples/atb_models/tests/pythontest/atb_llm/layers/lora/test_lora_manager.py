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
from unittest.mock import Mock, MagicMock, patch, mock_open
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from ddt import ddt, data, unpack

import atb_llm.layers.lora.lora_manager as lora_manager_module
from atb_llm.layers.lora.lora_manager import LoraManager
from atb_llm.layers.linear.linear import ColumnParallelLinear
from atb_llm.models.base.config import LoraConfig, LoraModelConfig
from atb_llm.models.base.flash_causal_lm_v3 import FlashCausalLMV3
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.adapter_manager import AdapterInfo, AdapterIdsRecord, AdapterIdsType
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.file_utils import MAX_PATH_LENGTH
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


BASE_ADAPTER_NAME = "base"
ADATPER1_STR = "adapter1"
ADATPER2_STR = "adapter2"
ADATPER3_STR = "adapter3"
ADATPER4_STR = "adapter4"
ADATPER5_STR = "adapter5"


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


class MockModelStatus(MagicMock):
    def __init__(self, *args, **kw,) -> None:
        super().__init__(*args, **kw)
        self.position_embedding_type = PositionEmbeddingType.ROPE
        self.head_dim = 64
    
    @classmethod
    def from_config(cls, mindie_llm_config):
        return cls()


class FakeModel(FlashCausalLMV3):
    model_status_cls = MockModelStatus

    def __init__(self, weight_tensor, prefixes):
        mindie_llm_config = MockModelParam()
        weight_loader = MagicMock()
        weight_loader.device = torch.device("cpu")
        weight_loader.mapping = MagicMock(spec=Mapping)
        weight_loader.mapping.rank = 1
        weight_loader.mapping.world_size = 4
        weight_loader.get_sharded = MagicMock(return_value=weight_tensor)
        weight_loader.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        super().__init__(mindie_llm_config, weight_loader)
        self.config = self.mindie_llm_config.hf_config
        self.linear_layer = ColumnParallelLinear(self.config, weight_loader, prefixes, bias=False)

    def forward(self, **kwargs):
        return None


@ddt
class TestLoraManager(unittest.TestCase):
    @patch("atb_llm.models.base.flash_causal_lm_v3.load_atb_speed", MagicMock())
    def setUp(self):
        torch.classes = MockTorchClasses()
        self.dtype = torch.float16
        self.n = 2 * random.randint(1, 512)
        self.k = 2 * random.randint(1, 512)
        self.r = 4 ** random.randint(1, 6)
        self.lora_adapter = {ADATPER1_STR: "fake_adapter_1_path", ADATPER2_STR: "fake_adapter_2_path",
                             ADATPER3_STR: "fake_adapter_3_path", ADATPER4_STR: "fake_adapter_4_path",
                             ADATPER5_STR: "fake_adapter_5_path"}
        self.max_loras = len(self.lora_adapter)
        self.max_lora_rank = 16 * self.r
        base_model = FakeModel(torch.rand((self.n, self.k)), ["linear_module"])
        base_model.mindie_llm_config.lora_config = LoraModelConfig(max_loras=self.max_loras, max_lora_rank=self.max_lora_rank)
        self.adapter_manager = LoraManager(base_model)

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

    @data((None, [BASE_ADAPTER_NAME]), ([None, None], [BASE_ADAPTER_NAME]),
          ([ADATPER1_STR, ADATPER1_STR], [ADATPER1_STR]),
          ([ADATPER3_STR], [ADATPER3_STR]), ([ADATPER1_STR, ADATPER2_STR], [ADATPER1_STR, ADATPER2_STR]))
    @unpack
    def test_preprocess_adatper_ids(self, adapter_ids, expected_adapter_ids):
        self._update_adapter_ids_registry()
        effective_adapter_ids = self.adapter_manager.preprocess_adapter_ids(adapter_ids)
        self.assertEqual(effective_adapter_ids, expected_adapter_ids)

    @data(([BASE_ADAPTER_NAME], AdapterIdsType.SINGLE), ([BASE_ADAPTER_NAME, ADATPER1_STR], AdapterIdsType.MIXED),
          ([ADATPER1_STR, ADATPER2_STR], AdapterIdsType.SORTED))
    @unpack
    def test_update_adapter_check_modification(self, adapter_ids, adapter_ids_type):
        self._update_adapter_ids_registry()
        self.adapter_manager.set_current_adapter_ids_status(adapter_ids)
        self.adapter_manager.check_adapter_weights_update()
        self.assertEqual(self.adapter_manager.previous_adapter_ids, AdapterIdsRecord(adapter_ids_type, adapter_ids))

    @data(([BASE_ADAPTER_NAME], [ADATPER1_STR], True), ([BASE_ADAPTER_NAME], [BASE_ADAPTER_NAME], False),
          ([ADATPER1_STR, BASE_ADAPTER_NAME], [ADATPER1_STR, ADATPER2_STR], False),
          ([ADATPER1_STR, BASE_ADAPTER_NAME], [ADATPER2_STR, ADATPER1_STR], True))
    @unpack
    def test_update_adapter_check_return_value(self, previous_adapter_ids, adapter_ids, expected_result):
        self._update_adapter_ids_registry()
        self.adapter_manager.set_current_adapter_ids_status(previous_adapter_ids)
        self.adapter_manager.check_adapter_weights_update()
        self.adapter_manager.set_current_adapter_ids_status(adapter_ids)
        need_update = self.adapter_manager.check_adapter_weights_update()
        self.assertEqual(need_update, expected_result)

    @data((None, True), ([None, None], True), ([ADATPER1_STR, ADATPER2_STR], True),
         ([ADATPER2_STR, ADATPER1_STR], False), ([BASE_ADAPTER_NAME, ADATPER1_STR], False),
         ([ADATPER1_STR, ADATPER3_STR], True))
    @unpack
    def test_check_adapter_ids_is_sorted(self, adapter_ids, expected_result):
        self._update_adapter_ids_registry()
        actual_result = self.adapter_manager.check_adapter_ids_is_sorted(adapter_ids)
        self.assertEqual(actual_result, expected_result)

    def test_sort_adapter_ids(self):
        self._update_adapter_ids_registry()
        candidates = [ADATPER1_STR, ADATPER2_STR, ADATPER3_STR, ADATPER4_STR, ADATPER5_STR]
        for i, item in enumerate(candidates):
            self.adapter_manager.adapter_info_registry[item] = AdapterInfo(idx=i, adapter_path="")
        adapter_ids = [ADATPER2_STR, ADATPER4_STR, ADATPER5_STR, ADATPER5_STR, ADATPER1_STR]
        sorted_adapter_idx, revert_adapter_idx = self.adapter_manager.sort_adapter_ids(adapter_ids)
        self.assertTrue(sorted_adapter_idx, [4, 0, 1, 2, 3])
        self.assertTrue(revert_adapter_idx, [1, 2, 3, 4, 0])

    @patch("atb_llm.models.base.flash_causal_lm_v3.load_atb_speed", MagicMock())
    @patch("atb_llm.layers.lora.lora_manager.SafetensorFileLoader")
    @patch("os.fdopen", new_callable=mock_open, read_data='{"r": 2, "lora_alpha": 8}')
    @patch("os.open", return_value=3)
    def test_preload_adapter(self, mock_open_func, mock_fdopen, mock_loader):
        lora_manager_module.standardize_path = Mock(side_effect=self._mock_standardize_path)
        lora_manager_module.check_file_safety = Mock()
        lora_manager_module.safe_open = Mock(side_effect=self._mock_safe_open)
        mock_module = Mock()
        mock_module.base_layer_prefixes = ["linear"]
        mock_module.base_weight_shape = (self.n, self.k)
        mock_module.dtype = self.dtype
        mock_module.load_lora_a = Mock(return_value=torch.Tensor([]))
        mock_module.load_lora_b = Mock(return_value=torch.Tensor([]))
        mock_module.set_lora = MagicMock()
        lora_manager_module.replace_submodule = Mock(return_value=mock_module)
        self.adapter_manager.lora_slots_occupied[0] = False
        self.adapter_manager.adapter_info_registry = dict()
        self.adapter_manager.base_model = FakeModel(torch.rand((self.n, self.k)), ["linear_module"])
        mock_loader_obj = mock_loader.return_value
        mock_loader_obj.release_file_handler = MagicMock()
        self.adapter_manager.preload_adapter({ADATPER1_STR: "fake_adapter_1_path"})
        self.assertEqual(len(self.adapter_manager.lora_modules), 1)
        self.assertTrue(self.adapter_manager.lora_slots_occupied[0])
        self.assertEqual(
            self.adapter_manager.adapter_info_registry[ADATPER1_STR],
            AdapterInfo(idx=0, adapter_path="fake_adapter_1_path", config=LoraConfig(r=2, lora_alpha=8))
        )
        self.assertEqual(
            self.adapter_manager.adapter_info_registry[BASE_ADAPTER_NAME],
            AdapterInfo(idx=self.max_loras, adapter_path="", config=LoraConfig(r=1, lora_alpha=1, use_rslora=False))
        )

    def test_load_adapter_duplicate(self):
        self.adapter_manager.adapter_info_registry = {
            ADATPER1_STR: AdapterInfo(
                idx=0, adapter_path="fake_adapter_1_path")
        }
        with self.assertRaises(ValueError):
            self.adapter_manager.load_adapter({ADATPER1_STR: "fake_adapter_1_path"})
    
    def test_add_adapter_invalid_number(self):
        with self.assertRaises(RuntimeError):
            self.adapter_manager.load_adapter(dict())

    def test_add_adapter_invalid_id_length(self):
        with self.assertRaises(ValueError):
            self.adapter_manager.load_adapter({"": "path1"})

    def test_add_adapter_id_exists(self):
        self.adapter_manager.adapter_info_registry = {
            ADATPER2_STR: AdapterInfo(idx=0, adapter_path="fake_adapter_2_path"),
            ADATPER1_STR: AdapterInfo(idx=1, adapter_path="fake_adapter_1_path")
        }
        with self.assertRaises(ValueError):
            self.adapter_manager.load_adapter({ADATPER1_STR: "path1"})

    def test_add_adapter_slots_full(self):
        self._update_adapter_ids_registry()
        with self.assertRaises(RuntimeError):
            self.adapter_manager.load_adapter({"id1": "path1"})

    @patch("atb_llm.layers.lora.lora_manager.SafetensorFileLoader")
    @patch("os.fdopen", new_callable=mock_open, read_data='{"r": 2, "lora_alpha": 8}')
    @patch("os.open", return_value=3)
    def test_load_adapter(self, mock_open_func, mock_fdopen, mock_loader):
        lora_manager_module.standardize_path = Mock(side_effect=self._mock_standardize_path)
        lora_manager_module.check_file_safety = Mock()
        lora_manager_module.safe_open = Mock(side_effect=self._mock_safe_open)
        self.adapter_manager.lora_slots_occupied[0] = False
        self.adapter_manager.adapter_info_registry = dict()
        mock_module = Mock()
        mock_module.base_layer_prefixes = ["linear"]
        mock_module.load_lora_a = Mock(return_value=torch.Tensor([]))
        mock_module.load_lora_b = Mock(return_value=torch.Tensor([]))
        mock_module.set_lora = MagicMock()
        self.adapter_manager.lora_modules = {"module": mock_module}
        mock_loader_obj = mock_loader.return_value
        mock_loader_obj.release_file_handler = MagicMock()
        self.adapter_manager.load_adapter({ADATPER1_STR: "fake_adapter_1_path"})
        self.assertTrue(self.adapter_manager.lora_slots_occupied[0])
        self.assertEqual(
            self.adapter_manager.adapter_info_registry[ADATPER1_STR],
            AdapterInfo(idx=0, adapter_path="fake_adapter_1_path", config=LoraConfig(r=2, lora_alpha=8))
        )

    def test_unload_adapter_not_found(self):
        self._update_adapter_ids_registry()
        with self.assertRaises(RuntimeError):
            self.adapter_manager.unload_adapter("id1")

    @patch("atb_llm.models.base.flash_causal_lm_v3.load_atb_speed", MagicMock())
    def test_unload_adapter(self):
        self._update_adapter_ids_registry()
        self.adapter_manager.lora_modules = dict()
        self.adapter_manager.base_model = FakeModel(torch.rand((self.n, self.k)), ["linear_module"])
        mock_module = Mock()
        mock_module.reset_lora = MagicMock()
        self.adapter_manager.lora_modules = {"module": mock_module}
        self.adapter_manager.unload_adapter(ADATPER1_STR)
        self.assertNotIn(ADATPER1_STR, self.adapter_manager.adapter_info_registry)
        self.assertFalse(self.adapter_manager.lora_slots_occupied[0])

    @patch("atb_llm.models.base.flash_causal_lm_v3.load_atb_speed", MagicMock())
    def test_get_adapters_single_adapter(self):
        self._update_adapter_ids_registry()
        self.adapter_manager.lora_modules = dict()
        self.adapter_manager.base_model = FakeModel(torch.rand((self.n, self.k)), ["linear_module"])
        self.adapter_manager.base_model.mindie_llm_config.lora_config = \
            LoraModelConfig(max_loras=self.max_loras, max_lora_rank=self.max_lora_rank)
        self.adapter_manager._create_lora_modules()
        lora_module = self.adapter_manager.lora_modules[list(self.adapter_manager.lora_modules.keys())[0]]
        fake_lora_a_tensor_1 = torch.rand((self.max_lora_rank, self.k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((self.n, self.max_lora_rank), dtype=self.dtype).T.contiguous()
        lora_module.lora_a_stacked[0].copy_(fake_lora_a_tensor_1)
        lora_module.lora_b_stacked[0].copy_(fake_lora_b_tensor_1)
        self.adapter_manager.set_current_adapter_ids_status([ADATPER1_STR])
        self.adapter_manager.check_adapter_weights_update()
        adapter_weights = self.adapter_manager.get_adapters([ADATPER1_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.allclose(adapter_weights["linear_module.lora_A.weight"].cpu(), fake_lora_a_tensor_1))
        self.assertTrue(torch.allclose(adapter_weights["linear_module.lora_B.weight"].cpu(), fake_lora_b_tensor_1))

    @patch("atb_llm.models.base.flash_causal_lm_v3.load_atb_speed", MagicMock())
    def test_get_adapters_mixed_adapter(self):
        self._update_adapter_ids_registry()
        self.adapter_manager.lora_modules = dict()
        self.adapter_manager.base_model = FakeModel(torch.rand((self.n, self.k)), ["linear_module"])
        self.adapter_manager.base_model.mindie_llm_config.lora_config = \
            LoraModelConfig(max_loras=self.max_loras, max_lora_rank=self.max_lora_rank)
        self.adapter_manager._create_lora_modules()
        lora_module = self.adapter_manager.lora_modules[list(self.adapter_manager.lora_modules.keys())[0]]
        fake_lora_a_tensor_1 = torch.rand((self.max_lora_rank, self.k), dtype=self.dtype)
        fake_lora_a_tensor_2 = torch.rand((self.max_lora_rank, self.k), dtype=self.dtype)
        fake_lora_b_tensor_1 = torch.rand((self.n, self.max_lora_rank), dtype=self.dtype).T.contiguous()
        fake_lora_b_tensor_2 = torch.rand((self.n, self.max_lora_rank), dtype=self.dtype).T.contiguous()
        lora_module.lora_a_stacked[0].copy_(fake_lora_a_tensor_1)
        lora_module.lora_b_stacked[0].copy_(fake_lora_b_tensor_1)
        lora_module.lora_a_stacked[1].copy_(fake_lora_a_tensor_2)
        lora_module.lora_b_stacked[1].copy_(fake_lora_b_tensor_2)
        self.adapter_manager.set_current_adapter_ids_status([ADATPER2_STR, ADATPER1_STR])
        self.adapter_manager.check_adapter_weights_update()
        adapter_weights = self.adapter_manager.get_adapters([ADATPER2_STR, ADATPER1_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.allclose(
            adapter_weights["linear_module.lora_A.weight"].cpu(),
            pad_sequence([fake_lora_a_tensor_2, fake_lora_a_tensor_1], batch_first=True)
        ))
        self.assertTrue(torch.allclose(
            adapter_weights["linear_module.lora_B.weight"].cpu(),
            pad_sequence([fake_lora_b_tensor_2, fake_lora_b_tensor_1], batch_first=True)
        ))

    @patch("atb_llm.models.base.flash_causal_lm_v3.load_atb_speed", MagicMock())
    def test_get_adapters_sorted_adapter(self):
        self._update_adapter_ids_registry()
        self.adapter_manager.lora_modules = dict()
        self.adapter_manager.base_model = FakeModel(torch.rand((self.n, self.k)), ["linear_module"])
        self.adapter_manager.base_model.mindie_llm_config.lora_config = \
            LoraModelConfig(max_loras=self.max_loras, max_lora_rank=self.max_lora_rank)
        self.adapter_manager._create_lora_modules()
        lora_module = self.adapter_manager.lora_modules[list(self.adapter_manager.lora_modules.keys())[0]]
        fake_lora_a = torch.rand((self.max_loras + 1, self.max_lora_rank, self.k), dtype=self.dtype)
        fake_lora_b = torch.rand((self.max_loras + 1, self.max_lora_rank, self.n), dtype=self.dtype)
        lora_module.lora_a_stacked.copy_(fake_lora_a)
        lora_module.lora_b_stacked.copy_(fake_lora_b)
        self.adapter_manager.set_current_adapter_ids_status([ADATPER1_STR, ADATPER2_STR])
        self.adapter_manager.check_adapter_weights_update()
        adapter_weights = self.adapter_manager.get_adapters([ADATPER1_STR, ADATPER2_STR])
        self.assertEqual(len(adapter_weights), 2)
        self.assertTrue(torch.allclose(
            adapter_weights["linear_module.lora_A.weight"].cpu(), fake_lora_a))
        self.assertTrue(torch.allclose(
            adapter_weights["linear_module.lora_B.weight"].cpu(), fake_lora_b))
    
    def _update_adapter_ids_registry(self):
        for adapter_id, adapter_path in self.lora_adapter.items():
            self.adapter_manager.adapter_info_registry[adapter_id] = AdapterInfo(
                idx=len(self.adapter_manager.adapter_info_registry), adapter_path=adapter_path)
        self.adapter_manager.adapter_info_registry[BASE_ADAPTER_NAME] = AdapterInfo(
            idx=len(self.adapter_manager.adapter_info_registry), adapter_path="")
        for i, _ in enumerate(self.adapter_manager.lora_slots_occupied):
            self.adapter_manager.lora_slots_occupied[i] = True

    def _mock_standardize_path(self, path, max_path_length=MAX_PATH_LENGTH, check_link=True):
        return path

    def _mock_safe_open(self, file_path: str, mode='r',
        encoding=None, permission_mode=0o600, is_exist_ok=True, **kwargs):
        return os.fdopen(os.open(file_path, os.O_RDONLY), "r", encoding=encoding)


if __name__ == '__main__':
    unittest.main()