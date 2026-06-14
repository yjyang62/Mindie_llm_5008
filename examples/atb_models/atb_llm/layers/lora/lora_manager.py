# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import re
import json
import math
from typing import List, Union

import torch

from atb_llm.layers.linear.linear import BaseLinear
from atb_llm.models.base.config import LoraConfig
from atb_llm.models.base.flash_causal_lm_v3 import FlashCausalLMV3
from atb_llm.utils.log import print_log, logger
from atb_llm.utils.adapter_manager import AdapterInfo, AdapterIdsRecord, AdapterIdsType
from atb_llm.utils.file_utils import standardize_path, check_file_safety, safe_open
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.utils.argument_utils import MAX_KEY_LENGTH
from atb_llm.utils.singleton import Singleton
from .utils import from_layer, replace_submodule


BASE_ADAPTER_NAME = "base"
LORA_CONFIG_NAME = "adapter_config.json"
ERROR_INVALID_LORA_ID = f"[INVALID LORA ID], Lora names should be no more than {MAX_KEY_LENGTH} characters long" \
                        + "and at least one character long."


def format_lora_a_key(base_weight_prefix):
    return f"base_model.model.{base_weight_prefix}.lora_A"


def format_lora_b_key(base_weight_prefix):
    return f"base_model.model.{base_weight_prefix}.lora_B"


class LoraManager(Singleton):
    """A class to manage LoRA(Low-Rank Adaptation) models and operations.

    Attributes:
        base_model: Reference to the base model being adapted.
        lora_model_config(LoraModelConfig): Configuration settings for the LoRA adapters.
        max_loras: Maximum LoRA weights on NPU
        lora_dtype(torch.dtype): Data type used for LoRA operations.
        device: The computing device used for the operation.
        mapping: Device mapping information for distributed systems.
        stacked_adapter_weights: Adapter weights cache.
        previous_adapter_ids: Cache of previously used adapter IDs.
        lora_modules: Dict of recording all of LoRA modules.
        adapter_info_registry: Registry tracking loaded adapter information.
        lora_slots_occupied: Boolean list tracking occupied LoRA slots.
    """
    def __init__(self, model: FlashCausalLMV3):
        """Initialize LoRA manager with configuration and operational parameters.
        Args:
            model: Base Model
        """
        self.base_model = model
        self.lora_model_config = self.base_model.mindie_llm_config.lora_config
        self.max_loras = self.lora_model_config.max_loras
        self.lora_dtype = self.base_model.torch_dtype
        self.device = self.base_model.torch_device
        self.mapping = self.base_model.weight_loader.mapping

        self.stacked_adapter_weights = None
        self.previous_adapter_ids = None
        self.format_lora_a_key = format_lora_a_key
        self.format_lora_b_key = format_lora_b_key
        self.lm_head_prefix = "lm_head"
        self.lora_modules = dict()
        self.adapter_info_registry = dict()
        self.lora_slots_occupied = [False for _ in range(self.lora_slots + 1)] # include base
        self.lora_slots_occupied[-1] = True # index self.lora_slots is preserved for base

    @property
    def lora_slots(self) -> int:
        """Get the maximum number of available LoRA slots"""
        return self.max_loras

    @staticmethod
    def load_lora_config(adapter_path: str):
        """Load LoRA configuration from LoRA weight"""
        adapter_config_path = standardize_path(os.path.join(adapter_path, LORA_CONFIG_NAME), check_link=True)
        check_file_safety(adapter_config_path)
        lora_config_in_dict = None
        with safe_open(adapter_config_path, mode="r", encoding="utf-8") as f:
            lora_config_in_dict = json.load(f)
        lora_config = LoraConfig.from_dict(lora_config_in_dict)
        return lora_config

    @staticmethod
    def get_last_two_prefix(prefix: str) -> str:
        """Extract the last two components from a module prefix string.
        Example:
            For `model.layer.0.attention.qkv`, returns 'attention.qkv'
        Args:
            prefix: Full module prefix string
        Returns:
            Last two components of path joined by dot
        """
        match = re.search(r'([^\.]+\.[^\.]+)$', prefix)
        if match:
            return match.group(1) # The first parenthesized subgroup.
        return ""

    def preload_adapter(self, lora_adapter: dict):
        """Load static LoRAs adapters into avaliable slots
        Args:
            lora_adapter: Dictionary of {adapter_id: adapter path} pairs
        Raises:
            ValueError: For invalid adapter IDs
            RuntimeError: when no slots available
        """
        # Update max_lora_rank if max_lora_rank is set as 0
        if lora_adapter is not None and self.lora_model_config.max_lora_rank == 0:
            self._update_max_lora_rank(lora_adapter)
        # Create lora modules
        self._create_lora_modules()
        # Load load_adpaters
        if lora_adapter is not None:
            for adapter_id, adapter_path in lora_adapter.items():
                if not 0 < len(adapter_id) <= MAX_KEY_LENGTH:
                    raise ValueError(ERROR_INVALID_LORA_ID)
                idx = self._find_avaliable_slots()
                self._load_adapter(adapter_id, adapter_path, idx)
        # Load dummy adapter
        self._load_dummy_adapter()

    def load_adapter(self, lora_adapter: dict):
        """Add a single LoRA adapter to the model
        Args:
            lora_adapter: Dictionary of {adapter_id: adapter path} pairs
        Returns:
            Result of add LoRA
        """
        if len(lora_adapter) != 1:
            raise RuntimeError(f"[INVALID LORA ID], Invalid number of adapters: expected 1, got {len(lora_adapter)}")
        adapter_id, adapter_path = next(iter(lora_adapter.items()))
        if not 0 < len(adapter_id) <= MAX_KEY_LENGTH:
            raise ValueError(ERROR_INVALID_LORA_ID)
        if adapter_id in self.adapter_info_registry:
            raise ValueError(f"[DUPLICATED LORA ID], This adapter_id {adapter_id} is already added before.")
        idx = self._find_avaliable_slots()
        self._load_adapter(adapter_id, adapter_path, idx)
        print_log(self.mapping.rank, logger.info,
                  f"Success: LoRA adapter '{adapter_id}' added successfully.", need_filter=True)

    def unload_adapter(self, adapter_id: str):
        """Remove a LoRA adapter from the model
        Args:
            adapter_id: the adapter to be removed.
        Returns:
            Result of remove LoRA
        """
        if adapter_id not in self.adapter_info_registry:
            raise RuntimeError(f"[INVALID LORA ID], LoRA adapter '{adapter_id}' not found")
        if adapter_id == BASE_ADAPTER_NAME:
            raise RuntimeError(f"[INVALID LORA ID], LoRA adapter '{BASE_ADAPTER_NAME}' cannot be unload")
        idx = self.adapter_info_registry.get(adapter_id).idx
        for _, module in self.lora_modules.items():
            module.reset_lora(idx)
        del self.adapter_info_registry[adapter_id]
        self.lora_slots_occupied[idx] = False
        print_log(self.mapping.rank, logger.info,
                  f"Success: LoRA adapter '{adapter_id}' removed successfully.",
                  need_filter=True)

    def check_adapter_ids_is_sorted(self, adapter_ids: None | List[str | None]) -> bool:
        """Verify if adapter IDs are sorted by their slot indices
        Args: 
            adapter_ids: List of adapter IDs to check
        Returns:
            True if IDs are in sorted slot order, False otherwise
        """
        if adapter_ids is None:
            adapter_ids = [None]
        adapter_idx = -1
        for adapter_id in adapter_ids:
            if adapter_id is None or adapter_id not in self.adapter_info_registry.keys():
                print_log(self.mapping.rank, logger.warning,
                    f"Adapter {adapter_id} is not found, using base model instead",
                    need_filter=True)
                cur_adapter_idx = self.adapter_info_registry.get(BASE_ADAPTER_NAME).idx
            else:
                cur_adapter_idx = self.adapter_info_registry.get(adapter_id).idx
            if cur_adapter_idx < adapter_idx:
                return False
            adapter_idx = cur_adapter_idx
        return True

    def sort_adapter_ids(self, adapter_ids: List[str]) -> Union[List[int], List[int]]:
        """Sort adapters by their slot indices and return mapping indices.
        Args:
            adapter_ids: List of adapter IDs to sort
        Returns:
            Tuple containing:
            - sorted_indices: Original indices sorted by slot order
            - revert_indices: Indices to revert to original order
        """
        sorted_adapter_ids_by_idx = sorted(enumerate(adapter_ids),
            key=lambda idx_adapter_id: self.adapter_info_registry.get(idx_adapter_id[1]).idx)
        sorted_adapter_idx = [idx for idx, _ in sorted_adapter_ids_by_idx]
        revert_adapter_idx = [idx for idx, _ in sorted(enumerate(sorted_adapter_idx), key=lambda group: group[1])]
        return sorted_adapter_idx, revert_adapter_idx

    def preprocess_adapter_ids(
            self, adapter_ids: None | List[str | None],
            model_name=None, model_id=None) -> List[str]:
        """Normalize and validate adapter IDs for processing
        Args:
            adapter_ids: Input list of adapter IDs
            model_name: Optional base model name alias
            model_id: Optional base model ID alias
        Returns:
            List of validated and normalized adapter IDs
        """
        if adapter_ids is None:
            adapter_ids = [None]

        effective_adapter_ids = []
        for adapter_id in adapter_ids:
            if adapter_id == model_name or adapter_id == model_id:
                effective_adapter_ids.append(BASE_ADAPTER_NAME)
            elif adapter_id and adapter_id not in self.adapter_info_registry.keys():
                print_log(self.mapping.rank, logger.warning,
                    f"Adapter `{adapter_id}` is not found, using base model instead",
                    need_filter=True)
                effective_adapter_ids.append(BASE_ADAPTER_NAME)
            elif adapter_id is None:
                effective_adapter_ids.append(BASE_ADAPTER_NAME)
            else:
                effective_adapter_ids.append(adapter_id)

        unique_adapter_ids = set(effective_adapter_ids)
        if len(unique_adapter_ids) == 1:
            effective_adapter_ids = list(unique_adapter_ids)
        return effective_adapter_ids

    def set_current_adapter_ids_status(self, adapter_ids: None | List[str | None]):
        """Update the current active adapter configuration
        Args:
            adapter_ids: List of adapter IDs to activate
        """
        if len(adapter_ids) == 1:
            self.current_adapter_ids_cache = AdapterIdsRecord(AdapterIdsType.SINGLE, adapter_ids)
        elif self.check_adapter_ids_is_sorted(adapter_ids):
            self.current_adapter_ids_cache = AdapterIdsRecord(AdapterIdsType.SORTED, adapter_ids)
        else:
            self.current_adapter_ids_cache = AdapterIdsRecord(AdapterIdsType.MIXED, adapter_ids)

    def check_adapter_weights_update(self) -> bool:
        """Check whether the adapter weights need to be updated.
        Returns:
            True if the adapter weights need to be updated, False otherwise
        """
        if self.previous_adapter_ids is None:
            self.previous_adapter_ids = self.current_adapter_ids_cache
            return True
        if self.previous_adapter_ids == self.current_adapter_ids_cache:
            return False
        if self.previous_adapter_ids.record_type == self.current_adapter_ids_cache.record_type == AdapterIdsType.SORTED:
            return False
        self.previous_adapter_ids = self.current_adapter_ids_cache
        return True

    def get_adapters(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        """Retrieve adapter weights based on current configuration
        Args:
            adapter_ids: List of adapter IDs to retrieve
        Returns:
            List of adapter weight tensors
        """
        if self.previous_adapter_ids.record_type == AdapterIdsType.SINGLE:
            adapter_weights = self._get_single_adapter(adapter_ids)
        elif self.previous_adapter_ids.record_type == AdapterIdsType.SORTED:
            adapter_weights = self._get_sorted_adapter()
        else:
            adapter_weights = self._get_mixed_adapter(adapter_ids)
        return adapter_weights

    def _update_max_lora_rank(self, lora_adapter):
        max_lora_rank = self.lora_model_config.max_lora_rank
        for adapter_path in lora_adapter.values():
            lora_cofig = self.load_lora_config(adapter_path)
            for _, linear_layer in self._find_lora_module():
                for prefix in linear_layer.prefixes:
                    max_lora_rank = max(max_lora_rank, self._get_r(lora_cofig, prefix))
        self.lora_model_config.max_lora_rank = max_lora_rank

    def _find_avaliable_slots(self) -> int:
        if False not in self.lora_slots_occupied:
            raise RuntimeError("[LORA MEMORY ERROR], No empty LoRA slots. " \
                               + "Please check the max_loras config or remove unused lora adapters.")
        index = self.lora_slots_occupied.index(False)
        return index

    def _find_lora_module(self):
        for module_name, module in self.base_model.named_modules(remove_duplicate=False):
            if "lm_head" in module_name:
                continue
            if isinstance(module, BaseLinear):
                yield module_name, module
            else:
                continue

    def _get_r(self, lora_config: LoraConfig, prefix: str) -> int:
        lora_r = lora_config.r
        if lora_config.rank_pattern is not None:
            r_prefix = self.get_last_two_prefix(prefix)
            lora_r = lora_config.rank_pattern.get(r_prefix, lora_r)
        if lora_r > self.lora_model_config.max_lora_rank > 0:
            raise ValueError("[INVALID LORA RANK], LoRA rank is greater than the max_lora_rank config.")
        return lora_r

    def _get_alpha(self, lora_config: LoraConfig, prefix: str) -> int:
        lora_alpha = lora_config.lora_alpha
        if lora_config.alpha_pattern is not None:
            alpha_prefix = self.get_last_two_prefix(prefix)
            lora_alpha = lora_config.alpha_pattern.get(alpha_prefix, lora_alpha)
        return lora_alpha

    def _get_scaling(self, lora_config: LoraConfig, prefix: str) -> int:
        lora_alpha = self._get_alpha(lora_config, prefix)
        lora_r = self._get_r(lora_config, prefix)
        if lora_config.use_rslora:
            scaling = lora_alpha / math.sqrt(lora_r)
        else:
            scaling = lora_alpha / lora_r
        return scaling

    def _create_lora_modules(self):
        for module_name, module in self._find_lora_module():
            new_module = replace_submodule(
                self.base_model, module_name,
                from_layer(module, self.base_model.mindie_llm_config, self.device))
            self.lora_modules[module_name] = new_module

    def _validate_adapter_id(self, adapter_id: None | str):
        if adapter_id is None:
            adapter_id = BASE_ADAPTER_NAME
        if adapter_id == BASE_ADAPTER_NAME:
            raise ValueError(f"[INVALID LORA ID], Adapter ID `{BASE_ADAPTER_NAME}` is reserved.")
        if adapter_id in self.adapter_info_registry:
            raise ValueError(f"[INVALID LORA ID], Adapter ID `{adapter_id} is already added before.")

    def _load_adapter(self, adapter_id: None | str, adapter_path: str, index: int):
        self._validate_adapter_id(adapter_id)
        lora_config = self.load_lora_config(adapter_path)
        # load weights
        weight = SafetensorFileLoader(adapter_path, self.device, self.mapping)
        for _, module in self.lora_modules.items():
            linear_layer_prefixes = module.base_layer_prefixes
            lora_a_prefixes = [self.format_lora_a_key(prefix) for prefix in linear_layer_prefixes]
            lora_b_prefixes = [self.format_lora_b_key(prefix) for prefix in linear_layer_prefixes]
            r_list = []
            alpha_list = []
            scale_list = []
            for prefix in linear_layer_prefixes:
                r_list.append(self._get_r(lora_config, prefix))
                alpha_list.append(self._get_alpha(lora_config, prefix))
                scale_list.append(self._get_scaling(lora_config, prefix))
            try:
                lora_a = module.load_lora_a(weight, lora_a_prefixes)
                lora_b = module.load_lora_b(weight, lora_b_prefixes, scale_list)
            except ValueError as _:
                print_log(weight.mapping.rank, logger.debug,
                    f"Tensor name {','.join(lora_a_prefixes + lora_b_prefixes)} not found, using all zero tensor instead",
                    need_filter=True)
                n, k = module.base_weight_shape
                lora_a = torch.zeros((1, k)).to(self.lora_dtype)
                lora_b = torch.zeros((1, n)).to(self.lora_dtype)
            module.set_lora(index, lora_a, lora_b)
        weight.release_file_handler()
        # register_adapter
        self.adapter_info_registry[adapter_id] = AdapterInfo(
            idx=index, adapter_path=adapter_path, config=lora_config)
        self.lora_slots_occupied[index] = True

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

    def _get_single_adapter(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        adapter_info = self.adapter_info_registry.get(adapter_ids[0])
        if adapter_info is not None and adapter_info.weights is not None:
            return adapter_info.weights
        idx = adapter_info.idx
        adapter_weights = dict()
        for _, module in self.lora_modules.items():
            adapter_weights.update({
                module.lora_a_stacked.name: module.lora_a_stacked.data[idx],
                module.lora_b_stacked.name: module.lora_b_stacked.data[idx]
            })
        adapter_info.weights = adapter_weights
        return adapter_info.weights

    def _get_sorted_adapter(self) -> List[torch.Tensor]:
        if self.stacked_adapter_weights is not None:
            return self.stacked_adapter_weights
        adapter_weights = dict()
        for _, module in self.lora_modules.items():
            adapter_weights.update({
                module.lora_a_stacked.name: module.lora_a_stacked.data,
                module.lora_b_stacked.name: module.lora_b_stacked.data
            })
        self.stacked_adapter_weights = adapter_weights
        return self.stacked_adapter_weights

    def _get_mixed_adapter(self, adapter_ids: List[str]) -> List[torch.Tensor]:
        adapter_weights = dict()
        for _, module in self.lora_modules.items():
            lora_a = []
            lora_b = []
            for adapter_id in adapter_ids:
                idx = self.adapter_info_registry.get(adapter_id).idx
                lora_a.append(module.lora_a_stacked.data[idx].unsqueeze(0))
                lora_b.append(module.lora_b_stacked.data[idx].unsqueeze(0))
            lora_a = torch.cat(lora_a, dim=0)
            lora_b = torch.cat(lora_b, dim=0)
            adapter_weights.update({
                module.lora_a_stacked.name: lora_a,
                module.lora_b_stacked.name: lora_b
            })
        return adapter_weights
