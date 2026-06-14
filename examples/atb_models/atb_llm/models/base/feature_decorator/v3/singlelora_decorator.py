# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from atb_llm.layers.lora.lora_manager import LoraManager
from atb_llm.models.base.flash_causal_lm_v3 import EngineWrapper
from atb_llm.utils.adapter_manager import AdapterIdsType
from .base_decorator import FeatureDecorator


class SingleLoraDecorator(FeatureDecorator):
    def __init__(self, model) -> None:
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
        return "single_lora"

    @property
    def is_enabled(self):
        return self.lora_adapter_config is not None

    def generate_engine_wrapper(self, engine_wrapper):
        feature_list = engine_wrapper.feature_list.copy()
        feature_list.append(self.feature_name)
        input_keys = engine_wrapper.input_keys.copy()
        args = engine_wrapper.args.copy()
        args.update({"enable_lora": True})
        return EngineWrapper(feature_list, input_keys, args)

    def modify_inputs(self, engine_inputs, engine_outputs, engine_runtime_params, input_metadata):
        adapter_ids = input_metadata.get("adapter_ids")
        effective_adapter_ids = self.adapter_manager.preprocess_adapter_ids(adapter_ids)
        if effective_adapter_ids == ["base"]:
            return
        need_update_weights = self.adapter_manager.check_adapter_weights_update()
        if need_update_weights:
            self.adapter_weights = self.adapter_manager.get_adapters(effective_adapter_ids)
            for engine_wrapper in self.model.get_engine_wrappers():
                if "single_lora" in engine_wrapper.feature_list:
                    engine_wrapper.set_weights(self.adapter_weights)

    def is_activated(self, input_metadata):
        adapter_ids = input_metadata.get("adapter_ids", None)
        effective_adapter_ids = self.adapter_manager.preprocess_adapter_ids(adapter_ids)
        self.adapter_manager.set_current_adapter_ids_status(effective_adapter_ids)
        return self.adapter_manager is not None \
            and self.adapter_manager.current_adapter_ids_cache is not None \
            and self.adapter_manager.current_adapter_ids_cache.record_type == AdapterIdsType.SINGLE \
            and self.adapter_manager.current_adapter_ids_cache.adapter_ids != ["base"]