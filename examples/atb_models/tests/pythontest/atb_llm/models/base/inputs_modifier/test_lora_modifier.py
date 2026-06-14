# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import random
import unittest
from unittest.mock import Mock, MagicMock

import torch

from atb_llm.models.base.config import BaseConfig
from atb_llm.utils.weights import Weights
from atb_llm.models.base.inputs_modifier.lora_modifier import LoraModifier
from atb_llm.utils.dist import initialize_distributed
from atb_llm.utils.adapter_manager import AdapterInfo
import atb_llm.utils.weights as weights_module


ADAPTER_1 = "adapter1"
ADAPTER_2 = "adapter2"
FAKE_PATH_1 = "fake_path_1"
FAKE_PATH_2 = "fake_path_2"
BASE = "base"
SORTED = "sorted"
IDX_1 = 0
IDX_2 = 2
NPU = "npu"


class MockFlashCausalLm:
    def __init__(self, **kwargs):
        self.mapping = MagicMock()
        self.mapping.rank = 0

        for key, value in kwargs.items():
            setattr(self, key, value)


class TestLoraModifier(unittest.TestCase):
    def setUp(self):
        world_size = 2 * random.randint(1, 4)
        rank = random.randint(0, world_size - 1)
        npu_id = 0
        self.process_group, self.device = initialize_distributed(rank, npu_id, world_size)
        self.model_name_or_path = "test_model_path"
        self.dtype = torch.float16

        mock_weight_files = Mock(return_value=[])
        weights_module.weight_files = mock_weight_files

        self.weights = Weights(
            self.model_name_or_path, self.device, self.dtype,
            process_group=self.process_group,
        )

        self.config = BaseConfig()

        self.adapter_info_registry = {
            ADAPTER_1: AdapterInfo(idx=0, adapter_path=FAKE_PATH_1),
            ADAPTER_2: AdapterInfo(idx=1, adapter_path=FAKE_PATH_2),
            BASE: AdapterInfo(idx=2, adapter_path=""),
            SORTED: AdapterInfo(idx=3, adapter_path=""),
        }

    def test_init_with_no_adapter(self):
        lora_modifier = LoraModifier(self.weights, MockFlashCausalLm(), lora_adapter=None, lora_model_config=None)
        self.assertFalse(lora_modifier.active)
        self.assertIsNone(lora_modifier.adapter_manager)

    def test_init_with_adapter(self):
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(), lora_adapter={ADAPTER_1: FAKE_PATH_1}, lora_model_config=None)
        self.assertTrue(lora_modifier.active)
        self.assertIsNotNone(lora_modifier.adapter_manager)

    def test_use_multi_adapter(self):
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(),
            lora_adapter={ADAPTER_1: FAKE_PATH_1, ADAPTER_2: FAKE_PATH_2}, lora_model_config=None)
        lora_modifier.adapter_manager.adapter_info_registry = self.adapter_info_registry
        lora_modifier.adapter_manager.update_adapter([ADAPTER_1, ADAPTER_2])
        self.assertTrue(lora_modifier.use_multi_adapters())
        self.assertFalse(lora_modifier.use_single_adapter())
        self.assertFalse(lora_modifier.use_no_adapter())
        lora_modifier.adapter_manager.update_adapter([ADAPTER_2, ADAPTER_1])
        self.assertTrue(lora_modifier.use_multi_adapters())
        self.assertFalse(lora_modifier.use_single_adapter())
        self.assertFalse(lora_modifier.use_no_adapter())

    def test_use_single_adapter(self):
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(),
            lora_adapter={ADAPTER_1: FAKE_PATH_1, ADAPTER_2: FAKE_PATH_2}, lora_model_config=None)
        lora_modifier.adapter_manager.update_adapter([ADAPTER_1])
        self.assertFalse(lora_modifier.use_multi_adapters())
        self.assertTrue(lora_modifier.use_single_adapter())
        self.assertFalse(lora_modifier.use_no_adapter())

    def test_use_no_adapter(self):
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(),
            lora_adapter={ADAPTER_1: FAKE_PATH_1, ADAPTER_2: FAKE_PATH_2}, lora_model_config=None)
        lora_modifier.adapter_manager.update_adapter([BASE])
        self.assertFalse(lora_modifier.use_multi_adapters())
        self.assertFalse(lora_modifier.use_single_adapter())
        self.assertTrue(lora_modifier.use_no_adapter())
    
    def test_modify_inputs_inactive(self):
        lora_modifier = LoraModifier(self.weights, MockFlashCausalLm(), lora_adapter=None, lora_model_config=None)
        engine_inputs = []
        lora_modifier.modify_inputs(engine_inputs, [BASE], torch.tensor([]), True)
        self.assertListEqual(engine_inputs, [])

    def test_modify_inputs_no_update(self):
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(), lora_adapter={ADAPTER_1: FAKE_PATH_1}, lora_model_config=None)
        engine_inputs = []
        lora_modifier.modify_inputs(engine_inputs, [None], torch.tensor([2, 1, 4]), True)
        self.assertListEqual(engine_inputs, [])
    
    def test_modify_inputs_need_update(self):
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(), lora_adapter={ADAPTER_1: FAKE_PATH_1}, lora_model_config=None)
        lora_modifier.adapter_manager.adapter_info_registry = {
            ADAPTER_1: AdapterInfo(idx=0, adapter_path="fake_path"),
            BASE: AdapterInfo(idx=2, adapter_path=""),
            SORTED: AdapterInfo(idx=3, adapter_path=""),
        }
        fake_weights = [torch.Tensor([1024, 1024]).npu()]
        fake_group_size = torch.Tensor([2, 3, 7]).npu()
        lora_modifier.adapter_manager.get_adapters = Mock(return_value=fake_weights)
        lora_modifier._calculate_adapter_group_size = Mock(return_value=fake_group_size)
        engine_inputs = [torch.Tensor([1])]
        lora_modifier.modify_inputs(engine_inputs, [ADAPTER_1], torch.tensor([2, 1, 4]).npu(), True)
        self.assertListEqual(lora_modifier.adapter_weights, fake_weights)
        self.assertEqual(len(engine_inputs), 3)
        self.assertTrue(torch.equal(engine_inputs[0], torch.Tensor([1])))
        self.assertTrue(torch.equal(engine_inputs[1], fake_group_size))
        self.assertTrue(torch.equal(engine_inputs[2], fake_weights[0]))
        args, kwargs = lora_modifier._calculate_adapter_group_size.call_args
        self.assertEqual(args[0], [ADAPTER_1])
        self.assertTrue(torch.equal(args[1], torch.tensor([2, 1, 4]).npu()))
        self.assertTrue(kwargs.get("is_prefill"))

        lora_modifier.modify_inputs(engine_inputs, [ADAPTER_1], torch.tensor([2, 1, 4]).npu(), False)
        args, kwargs = lora_modifier._calculate_adapter_group_size.call_args
        self.assertTrue(torch.equal(args[1], torch.tensor([1, 1, 1], dtype=torch.int64).npu()))

    def test_calculate_adapter_group_size_single_adapter(self):
        fake_input_lengths = torch.tensor([2]).npu()
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(), lora_adapter={ADAPTER_1: FAKE_PATH_1}, lora_model_config=None)
        group_size = lora_modifier._calculate_adapter_group_size([ADAPTER_1], fake_input_lengths, True)
        self.assertTrue(torch.equal(group_size, torch.zeros(1, dtype=self.dtype, device=NPU)))

    def test_calculate_adapter_group_size_mixed(self):
        fake_input_lengths = torch.tensor([4, 2]).npu()
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(),
            lora_adapter={ADAPTER_1: FAKE_PATH_1, ADAPTER_2: FAKE_PATH_2}, lora_model_config=None)
        lora_modifier.adapter_manager.adapter_info_registry = self.adapter_info_registry
        adapter_ids = [ADAPTER_2, ADAPTER_1]
        lora_modifier.adapter_manager.update_adapter(adapter_ids)
        group_size = lora_modifier._calculate_adapter_group_size(
            adapter_ids, fake_input_lengths, True)
        self.assertTrue(torch.equal(group_size, torch.tensor([4, 6], dtype=torch.int64, device=NPU)))

        group_size = lora_modifier._calculate_adapter_group_size(
            adapter_ids, torch.tensor([1, 2], dtype=self.dtype, device=NPU), False)
        self.assertTrue(torch.equal(group_size, torch.tensor([1, 2], dtype=torch.int64, device=NPU)))

    def test_calculate_adapter_group_size_sorted(self):
        fake_input_lengths = torch.tensor([4, 2]).npu()
        lora_modifier = LoraModifier(
            self.weights, MockFlashCausalLm(),
            lora_adapter={ADAPTER_1: FAKE_PATH_1, ADAPTER_2: FAKE_PATH_2}, lora_model_config=None)
        lora_modifier.adapter_manager.adapter_info_registry = self.adapter_info_registry
        adapter_ids = [ADAPTER_1, BASE]
        lora_modifier.adapter_manager.update_adapter(adapter_ids)
        group_size = lora_modifier._calculate_adapter_group_size(
            adapter_ids, fake_input_lengths, True)
        self.assertTrue(torch.equal(group_size, torch.tensor([4, 4, 6], dtype=torch.int64, device=NPU)))

        group_size = lora_modifier._calculate_adapter_group_size(
            adapter_ids, torch.tensor([1, 2], dtype=self.dtype, device=NPU), False)
        self.assertTrue(torch.equal(group_size, torch.tensor([1, 1, 3], dtype=torch.int64, device=NPU)))


if __name__ == '__main__':
    unittest.main()