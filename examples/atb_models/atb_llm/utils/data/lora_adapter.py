# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List
from abc import ABC, abstractmethod
import torch

from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.quantize.quant_type import LinearTypeV2, QuantType
from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.runtime.lora.lora_layers import ParallelLinearWithLoRA
from mindie_llm.runtime.lora.lora_manager import (
    LoraManager as LoraManagerAdaptee,
    AdapterIdsType
)
from mindie_llm.runtime.lora.utils import replace_submodule


class ParallelLinearWithLoRAAdaptee(CustomLayer):
    def __init__(self, lora_module: ParallelLinearWithLoRA):
        super().__init__()
        self.lora_module = lora_module
    
    def get_weights_for_atb_graph(
        self, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> List[torch.Tensor]:
        return self.lora_module.base_layer.get_weights_for_atb_graph(padding, is_swiglu_quant_enabled, quant_type)

    def get_linear_descs(self) -> List[LinearTypeV2]:
        return self.lora_module.base_layer.get_linear_descs()

    def get_weight_transpose_type(self) -> List[TransposeType]:
        return self.lora_module.base_layer.get_weight_transpose_type()


class LoraManager(LoraManagerAdaptee):
    _PLACEHOLDER = torch.tensor([1], dtype=torch.get_default_dtype(), device='npu')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_adapters(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        """Retrieve adapter weights based on current configuration.
        Args:
            adapter_ids: List of adapter IDs.
        Returns:
            List of adapter weight tensors.
        """
        if self.previous_adapter_ids.record_type == AdapterIdsType.SINGLE:
            adapter_weights = self._get_single_adapter(adapter_ids)
        elif self.previous_adapter_ids.record_type == AdapterIdsType.SORTED:
            adapter_weights = self._get_sorted_adapter()
        else:
            adapter_weights = self._get_mixed_adapter(adapter_ids)
        return adapter_weights

    def wrap_lora_module_for_atb_graph(self):
        for module_name, lora_module in self.lora_modules.items():
            replace_submodule(self.base_model, module_name, ParallelLinearWithLoRAAdaptee(lora_module))

    def _get_single_adapter(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        adapter_info = self.adapter_info_registry.get(adapter_ids[0])
        if adapter_info is not None and adapter_info.weights is not None:
            return adapter_info.weights
        idx = adapter_info.idx
        adapter_weights = []
        for _, module in self.lora_modules.items():
            adapter_weights.append(module.lora_a_stacked.data[idx])
            adapter_weights.append(module.lora_b_stacked.data[idx])
            # add placeholder
            for _ in range(module.pack_num - 1):
                adapter_weights.extend([self._PLACEHOLDER, self._PLACEHOLDER])  # lora_a, lora_b
        adapter_info.weights = adapter_weights
        return adapter_info.weights

    def _get_sorted_adapter(self) -> List[torch.Tensor]:
        if self.stacked_adapter_weights is not None:
            return self.stacked_adapter_weights
        adapter_weights = []
        for _, module in self.lora_modules.items():
            adapter_weights.append(module.lora_a_stacked.data)
            adapter_weights.append(module.lora_b_stacked.data)
            # add placeholder
            for _ in range(module.pack_num - 1):
                adapter_weights.extend([self._PLACEHOLDER, self._PLACEHOLDER])  # lora_a, lora_b
        self.stacked_adapter_weights = adapter_weights
        return self.stacked_adapter_weights

    def _get_mixed_adapter(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        adapter_weights = []
        for _, module in self.lora_modules.items():
            lora_a = []
            lora_b = []
            for adapter_id in adapter_ids:
                idx = self.adapter_info_registry.get(adapter_id).idx
                lora_a.append(module.lora_a_stacked.data[idx].unsqueeze(0))
                lora_b.append(module.lora_b_stacked.data[idx].unsqueeze(0))
            lora_a = torch.cat(lora_a, dim=0)
            lora_b = torch.cat(lora_b, dim=0)
            adapter_weights.append(lora_a)
            adapter_weights.append(lora_b)
            # add placeholder
            for _ in range(module.pack_num - 1):
                adapter_weights.extend([self._PLACEHOLDER, self._PLACEHOLDER])  # lora_a, lora_b
        return adapter_weights
