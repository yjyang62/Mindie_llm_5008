# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List
import torch

from atb_llm.layers.lora.lora_manager import LoraManager
from atb_llm.models.base.flash_causal_lm_v3 import FlashCausalLMV3, EngineWrapper
from atb_llm.utils.adapter_manager import AdapterIdsType
from .base_decorator import FeatureDecorator


class MultiLoraDecorator(FeatureDecorator):
    def __init__(self, model: FlashCausalLMV3) -> None:
        super().__init__(model)
        self.feature_stackable_whitelist = [
            "prefill",
            "decode"
        ]
        self.need_additional_engine = True
        self.device = self.model.torch_device
        self.dtype = self.model.torch_dtype
        self.adapter_manager = None
        self.adapter_weights = []
        self.lora_adapter_config = getattr(self.model.mindie_llm_config, "lora_config", None)
        if self.is_enabled:
            self.adapter_manager = LoraManager(self.model)
            if not hasattr(self.model, "adapter_manager"):
                setattr(self.model, "adapter_manager", self.adapter_manager)
        self.adapter_weights = None

    @property
    def feature_name(self):
        return "multi_lora"

    @property
    def is_enabled(self):
        return self.lora_adapter_config is not None

    def generate_engine_wrapper(self, engine_wrapper):
        feature_list = engine_wrapper.feature_list.copy()
        feature_list.append(self.feature_name)
        input_keys = engine_wrapper.input_keys.copy()
        input_keys.add("group_list")
        args = engine_wrapper.args.copy()
        args.update({"enable_lora": True})
        return EngineWrapper(feature_list, input_keys, args)

    def modify_inputs(self, engine_inputs, engine_outputs, engine_runtime_params, input_metadata):
        adapter_ids = input_metadata.get("adapter_ids", None)
        effective_adapter_ids = self.adapter_manager.preprocess_adapter_ids(adapter_ids)
        need_update_weights = self.adapter_manager.check_adapter_weights_update()
        if need_update_weights:
            self.adapter_weights = self.adapter_manager.get_adapters(effective_adapter_ids)
            for engine_wrapper in self.model.get_engine_wrappers():
                if "multi_lora" in engine_wrapper.feature_list:
                    engine_wrapper.set_weights(self.adapter_weights)
        is_prefill = input_metadata.get("is_prefill", False)
        input_lengths = input_metadata.get("input_lengths", None)
        if is_prefill:
            actual_input_tokens = input_lengths
        else:
            actual_input_tokens = torch.ones_like(input_lengths, device=self.device, dtype=torch.int64)
        group_list = self._calculate_adapter_group_size(
            effective_adapter_ids, actual_input_tokens, is_prefill=is_prefill)
        engine_inputs.update({"group_list": group_list})

    def is_activated(self, input_metadata):
        adapter_ids = input_metadata.get("adapter_ids")
        effective_adapter_ids = self.adapter_manager.preprocess_adapter_ids(adapter_ids)
        self.adapter_manager.set_current_adapter_ids_status(effective_adapter_ids)
        return self.adapter_manager is not None \
            and self.adapter_manager.current_adapter_ids_cache is not None \
            and self.adapter_manager.current_adapter_ids_cache.record_type != AdapterIdsType.SINGLE

    def _calculate_adapter_group_size(
            self, adapter_ids: None | List[str | None],
            input_lengths: torch.Tensor, is_prefill: bool = False
    ) -> torch.Tensor:
        """
        Calculate the group size based on the adapter IDs and input lengths.

        group size is used when multiple lora adapters are enabled in one batch.
        It means the cumulative sum of the actual sequence lengths for each 
        matrix passing into the group matmul operation.
        It has different calculation method depends on whether `adapter_ids` is sorted and the inference phase.

        Args:
            adapter_ids (None | List[str | None]): The list of adapter IDs.
            input_lengths (torch.Tensor): The lengths of the inputs.
            is_prefill (bool, optional): Whether to prefill the inputs. Defaults to False.

        Returns:
            torch.Tensor: The calculated adapter group size.
        """
        if self.adapter_manager.previous_adapter_ids.record_type == AdapterIdsType.MIXED:
            if is_prefill:
                cum_group_size = torch.cumsum(input_lengths, dim=0, dtype=torch.int64)
            else:
                cum_group_size = torch.arange(1, input_lengths.shape[0] + 1, 
                                              dtype=torch.int64, device=self.device)
        else:
            adapters_count = self.adapter_manager.lora_slots + 1 # include base
            adapter_indexes = []
            for adapter_id in adapter_ids:
                adapter_indexes.append(self.adapter_manager.adapter_info_registry.get(adapter_id).idx)
            labels = torch.tensor(adapter_indexes, device=self.device, dtype=torch.int64)
            unique_labels = torch.arange(0, adapters_count, dtype=torch.int64, device=self.device)
            group = torch.zeros_like(unique_labels).scatter_add_(0, labels, input_lengths.to(torch.int64))
            cum_group_size = torch.cumsum(group, dim=0, dtype=torch.int64)
        return cum_group_size