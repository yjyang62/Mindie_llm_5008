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
from ddt import ddt, data, unpack

from atb_llm.models.base.flash_causal_lm_v3 import FlashCausalLMV3, EngineWrapper
from atb_llm.models.base.feature_decorator.v3.multilora_decorator import MultiLoraDecorator
from atb_llm.models.base.config import LoraModelConfig
from atb_llm.utils.adapter_manager import AdapterInfo


BASE_ADAPTER_NAME = "base"
ADATPER1_STR = "adapter1"
ADATPER2_STR = "adapter2"


@ddt
class TestMultiLoraDecorator(unittest.TestCase):
    def setUp(self) -> None:
        self.base_model = MagicMock(spec=FlashCausalLMV3)
        self.base_model.mindie_llm_config = MagicMock()
        self.base_model.mindie_llm_config.lora_config = LoraModelConfig(max_loras=2, max_lora_rank=8)
        self.base_model.torch_device = torch.device("cpu")
        self.base_model.torch_dtype = torch.float16
        self.base_model.weight_loader = MagicMock()
        self.base_model.weight_loader.mapping = MagicMock()
        self.lora_adapters = {ADATPER1_STR: "fake_lora_path1", ADATPER2_STR: "fake_lora_path2"}
        self.decorator = MultiLoraDecorator(self.base_model)
        self._update_adapter_ids_registry()

    def test_init_without_lora_model_config(self):
        base_model = MagicMock(spec=FlashCausalLMV3)
        base_model.torch_device = torch.device("cpu")
        base_model.torch_dtype = torch.float16
        base_model.mindie_llm_config = MagicMock()
        base_model.mindie_llm_config.lora_config = None
        decorator = MultiLoraDecorator(base_model)
        self.assertEqual(decorator.feature_name, "multi_lora")
        self.assertFalse(decorator.is_enabled)

    def test_is_stackable(self):
        self.assertTrue(self.decorator.is_stackable(["prefill"]))
        self.assertFalse(self.decorator.is_stackable(["prefill", "other_mock"]))

    def test_generate_engine_wrappers(self):
        engine_wrappers = [
            EngineWrapper(["prefill"], {"input_ids"}, {}),
            EngineWrapper(["decode"], {"input_ids"}, {}),
            EngineWrapper(["prefill", "other_mock"], {"input_ids"}, {})
        ]
        self.decorator.expand_engine_wrapper_collections(engine_wrappers)
        self.assertEqual(len(engine_wrappers), 5)

    @data((None, False), ([None, None], False), ([ADATPER1_STR], False), ([BASE_ADAPTER_NAME, ADATPER1_STR], True),
          ([ADATPER1_STR, ADATPER1_STR], False), ([ADATPER1_STR, ADATPER2_STR], True))
    @unpack
    def test_is_activated(self, adapter_ids, expected_result):
        input_metadata = {"adapter_ids": adapter_ids}
        self.assertEqual(self.decorator.is_activated(input_metadata), expected_result)

    def test_modify_inputs(self):
        decorator = MultiLoraDecorator(self.base_model)
        decorator.adapter_manager = MagicMock()
        decorator.adapter_manager.lora_slots = 2
        decorator.adapter_manager.preprocess_adapter_ids.return_value = ["adapter1", "adapter2"]
        decorator.adapter_manager.check_adapter_weights_update.return_value = True
        decorator.adapter_manager.get_adapters.return_value = MagicMock()
        mock_engine_wrapper1 = MagicMock()
        mock_engine_wrapper1.feature_list = ["other_mock", "multi_lora"]
        mock_engine_wrapper2 = MagicMock()
        mock_engine_wrapper2.feature_list = ["other_mock"]
        self.base_model.get_engine_wrappers = MagicMock()
        self.base_model.get_engine_wrappers.return_value = [mock_engine_wrapper1, mock_engine_wrapper2]
        decorator.modify_inputs({}, {}, {}, input_metadata={"input_lengths": torch.tensor([4, 5], dtype=torch.int64),
                                                            "adapter_ids": ["adapter1", "adapter2"]})
        decorator.adapter_manager.preprocess_adapter_ids.assert_called_once_with(["adapter1", "adapter2"])
        decorator.adapter_manager.check_adapter_weights_update.assert_called_once()
        decorator.adapter_manager.get_adapters.assert_called_once_with(["adapter1", "adapter2"])
        mock_engine_wrapper1.set_weights.assert_called_once()
        mock_engine_wrapper2.set_weights.assert_not_called()

    def test_calculate_adapter_group_size_mixed(self):
        fake_input_lengths = torch.tensor([4, 2], dtype=torch.int64)
        decorator = MultiLoraDecorator(self.base_model)
        decorator.adapter_manager.max_loras = 2
        self._update_adapter_ids_registry()
        adapter_ids = [ADATPER2_STR, ADATPER1_STR]
        decorator.adapter_manager.set_current_adapter_ids_status(adapter_ids)
        decorator.adapter_manager.check_adapter_weights_update()
        group_size = decorator._calculate_adapter_group_size(
            adapter_ids, fake_input_lengths, True)
        self.assertTrue(torch.equal(group_size, torch.tensor([4, 6], dtype=torch.int64)))

    def test_calculate_adapter_group_size_sorted(self):
        fake_input_lengths = torch.tensor([4, 2], dtype=torch.int64)
        decorator = MultiLoraDecorator(self.base_model)
        decorator.adapter_manager.max_loras = 2
        self._update_adapter_ids_registry()
        adapter_ids = [ADATPER1_STR, BASE_ADAPTER_NAME]
        decorator.adapter_manager.set_current_adapter_ids_status(adapter_ids)
        decorator.adapter_manager.check_adapter_weights_update()
        group_size = decorator._calculate_adapter_group_size(
            adapter_ids, fake_input_lengths, True)
        self.assertTrue(torch.equal(group_size, torch.tensor([4, 4, 6], dtype=torch.int64)))

        group_size = decorator._calculate_adapter_group_size(
            adapter_ids, torch.tensor([1, 2], dtype=torch.int64), False)
        self.assertTrue(torch.equal(group_size, torch.tensor([1, 1, 3], dtype=torch.int64)))

    def _update_adapter_ids_registry(self):
        for adapter_id, adapter_path in self.lora_adapters.items():
            self.decorator.adapter_manager.adapter_info_registry[adapter_id] = AdapterInfo(
                idx=len(self.decorator.adapter_manager.adapter_info_registry), adapter_path=adapter_path)
        self.decorator.adapter_manager.adapter_info_registry[BASE_ADAPTER_NAME] = AdapterInfo(
            idx=len(self.decorator.adapter_manager.adapter_info_registry), adapter_path="")


if __name__ == "__main__":
    unittest.main()