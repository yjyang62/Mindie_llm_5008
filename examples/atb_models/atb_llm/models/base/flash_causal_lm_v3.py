# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar

import torch

from atb_llm.models.base.mindie_llm_config import ModelStatus, MindIELLMConfig
from atb_llm.utils.initial import load_atb_speed
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.models.base.model_input_managers import AttentionMaskGenerator, PositionEmbeddingGenerator, KVCacheUpdater
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.parameter import Parameter


class FlashCausalLMV3(torch.nn.Module, ABC):
    model_status_cls = ModelStatus

    def __init__(self, mindie_llm_config: MindIELLMConfig, weight_loader: SafetensorFileLoader, **kwargs) -> None:
        super().__init__()
        load_atb_speed()

        self.mindie_llm_config = mindie_llm_config
        self.weight_loader = weight_loader
        self.torch_dtype = mindie_llm_config.hf_config.torch_dtype
        self.torch_device = weight_loader.device

        self.model_status = self.model_status_cls.from_config(self.mindie_llm_config)
        self.soc_info = NPUSocInfo()
        self.model = None
        self.lm_head = None

        self.attn_mask_generator = AttentionMaskGenerator(mindie_llm_config, self.model_status, self.torch_device)
        self.pos_embed_generator = PositionEmbeddingGenerator(mindie_llm_config, self.model_status, self.torch_device)
        self.kv_cache_updater = KVCacheUpdater(mindie_llm_config)

        self.device_weights_dict = {}

    @abstractmethod
    def forward(self, **kwargs):
        pass

    def _get_device_weights(self):
        weight_nz = self.soc_info.need_nz or self.model_status.enable_matmul_nz
        for _, parameter in self.model.named_parameters():
            if isinstance(parameter, Parameter):
                parameter.weight_format_cast(weight_nz)
            self.device_weights_dict.update({parameter.name: parameter.data})

        if isinstance(self.lm_head.weight, Parameter):
            self.lm_head.weight.weight_format_cast(weight_nz)
        self.device_weights_dict.update({self.lm_head.weight.name: self.lm_head.weight})

    def _prepare_default_inputs(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[dict, dict, dict]:
        """
        Prepare default inputs and runtime params for engine execution.

        Args:
            See arguments for `forward` for more details.
        
        Outputs:
            dict: named `engine_inputs`, which maps from input key to input tensor.
            dict: named `engine_outputs`, which maps from output key to empty tensor.
            dict: named `engine_runtime_params`: which maps from param key to param tensor.
        """
        engine_inputs = {}
        engine_outputs = {}
        engine_runtime_params = {}
        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                           dtype=torch.int64, device=input_ids.device)
        
        if is_prefill or self.pos_embed_generator.cosine_table is None or self.pos_embed_generator.sine_table is None:
            self.pos_embed_generator.generate_position_embedding(self.mindie_llm_config.hf_config.max_position_embeddings)
        
        attn_mask = self.attn_mask_generator.generate_mask(
            kwargs.get('attn_mask'),
            is_prefill,
            max_seq_len=max_seq_len,
            position_ids=position_ids
        )

        input_lengths = input_lengths.to(torch.int32)
        engine_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids.to(torch.int64),
            "cosine_table": self.pos_embed_generator.cosine_table,
            "sine_table": self.pos_embed_generator.sine_table,
            "slots_mapping": slots.to(torch.int32),
            "seq_len": input_lengths,
        }
        if is_prefill:
            engine_inputs.update({
                "attention_mask": attn_mask,
                "lm_head_indices": lm_head_indices.to(torch.int64),
            })
        else:
            engine_inputs.update({
                "block_table": block_tables.to(torch.int32)
            })

        batch_size = lm_head_indices.shape[0] if is_prefill else input_ids.shape[0]
        vocab_size = self.mindie_llm_config.hf_config.vocab_size
        engine_outputs = {
            "model_out": torch.empty(batch_size, vocab_size, dtype=self.torch_dtype, device=input_ids.device)
        }

        engine_runtime_params = {
            "seq_len": input_lengths.cpu()
        }

        return engine_inputs, engine_outputs, engine_runtime_params
    
    def _update_model_inputs(self, input_metadata) -> tuple[dict, dict, dict]:
        return {}, {}, {}

    def _get_model_outputs(self, engine_outputs):
        torch.npu.synchronize()
        return engine_outputs["model_out"]


T = TypeVar("T", bound=FlashCausalLMV3)


def torch_to_mindie_graph(*feature_decorator_cls_list):
    feature_decorator_cls_list = list(feature_decorator_cls_list)

    def class_decorator(cls: Type[T]):
        class EngineFromNetwork(cls):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                
                self._feature_dict = {}
                for feature_decorator_cls in feature_decorator_cls_list:
                    feature_decorator = feature_decorator_cls(self)
                    if feature_decorator.is_enabled:
                        self._feature_dict[feature_decorator.feature_name] = feature_decorator
                
                self._feature_index_map = {}
                self._engine_wrappers = []
                self._ready_for_execute = False
            
            def get_engine_wrappers(self):
                return [engine_wrapper for engine_wrapper in self._engine_wrappers if engine_wrapper is not None]
            
            def forward(self, **kwargs):
                input_metadata = {**kwargs}
                
                if not self._ready_for_execute:
                    self._get_device_weights()
                    input_keys = self._get_input_keys(input_metadata)
                    self._build_engines(input_keys)
                    self._ready_for_execute = True

                self.kv_cache_updater.update_kv_cache(input_metadata.get("kv_cache", None), self.get_engine_wrappers())
                engine_inputs, engine_outputs, engine_runtime_params = self._prepare_default_inputs(**input_metadata)
                feature_list = self._update_feature_inputs(engine_inputs, engine_outputs, engine_runtime_params, input_metadata)
                for target, source in zip([engine_inputs, engine_outputs, engine_runtime_params], self._update_model_inputs(input_metadata)):
                    target.update(source)
                self._engine_wrappers[self._get_engine_index(feature_list)].execute(engine_inputs, engine_outputs, engine_runtime_params)
                return self._get_model_outputs(engine_outputs)

            def _get_input_keys(self, input_metadata):
                input_keys = {"input_ids", "position_ids", "slots_mapping", "seq_len", 
                             "cosine_table", "sine_table", "attention_mask", "lm_head_indices", "block_table"}
                fake_input_dict = {key: None for key in input_keys}
                input_metadata_prefill = {**input_metadata}.update({"is_prefill": True})
                input_metadata_decode = {**input_metadata}.update({"is_prefill": False})
                fake_input_dict.update(self._update_model_inputs(input_metadata_prefill)[0])
                fake_input_dict.update(self._update_model_inputs(input_metadata_decode)[0])
                return set(fake_input_dict)
            
            def _build_engines(self, input_keys):
                self._feature_index_map = {"prefill": 0}
                index = 1
                for feature_name, feature_decorator in self._feature_dict.items():
                    if feature_decorator.need_additional_engine:
                        self._feature_index_map[feature_name] = index
                        index += 1
                
                engine_list_len = 2 ** index
                self._engine_wrappers = [None for i in range(engine_list_len)]
                engine_wrappers = self._init_engine_wrappers(input_keys)
                
                for engine_wrapper in engine_wrappers:
                    feature_list = engine_wrapper.feature_list
                    self._build(engine_wrapper)
                    self._engine_wrappers[self._get_engine_index(feature_list)] = engine_wrapper
            
            def _build(self, engine_wrapper):
                input_tensors = {}
                for key in engine_wrapper.input_keys:
                    input_tensors[key] = Tensor(key)
                k_caches = []
                v_caches = []
                for i in range(self.mindie_llm_config.hf_config.num_hidden_layers):
                    k_caches.append(Tensor(f"layer_{i}_k_cache"))
                    v_caches.append(Tensor(f"layer_{i}_v_cache"))
                input_tensors["k_caches"] = k_caches
                input_tensors["v_caches"] = v_caches

                outputs = super().forward(**input_tensors, **engine_wrapper.args)

                for output_key, output in outputs.items():
                    get_default_net().mark_output(
                        output, output_key
                    )

                is_prefill = engine_wrapper.args.get("is_prefill", False)
                del_fpass_keys = [
                    "SwigluWeightPackQuantPerTokenPass", "SwigluWeightPackQuantPerChannelPass",
                    "SwigluWeightNoPackQuantPerTokenPass", "SwigluWeightNoPackQuantPerChannelPass",
                    "AddNormWithBiasQuantPerTensorPass"
                ]

                if not is_prefill:
                    del_fpass_keys.append("MatmulAllReducePass")

                engine = get_default_net().build_engine(del_fpass_keys=del_fpass_keys)
                engine_wrapper.engine = engine
                engine_wrapper.set_weights(self.device_weights_dict)

            def _init_engine_wrappers(self, input_keys):
                base_prefill_engine_wrapper = EngineWrapper(
                    feature_list=["prefill"],
                    input_keys=input_keys,
                    args={"is_prefill": True})
                base_decode_engine_wrapper = EngineWrapper(
                    feature_list=["decode"],
                    input_keys=input_keys,
                    args={"is_prefill": False})
                engine_wrappers = [base_prefill_engine_wrapper, base_decode_engine_wrapper]

                for _, feature_decorator in self._feature_dict.items():
                    feature_decorator.expand_engine_wrapper_collections(engine_wrappers)
                return engine_wrappers

            def _get_engine_index(self, feature_list):
                """Derive engine wrapper's index from engine wrapper's feature list"""
                result = 0
                for feature_name in feature_list:
                    if feature_name == "decode":
                        continue
                    elif feature_name in self._feature_index_map:
                        index = self._feature_index_map[feature_name]
                        result |= (1 << index)
                    else:
                        raise ValueError(f"Feature name {feature_name} is not registed")
                return result
            
            def _update_feature_inputs(self, engine_inputs: dict, engine_outputs: dict,
                      engine_runtime_params: dict, input_metadata: dict):
                """Update feature inputs."""
                current_engine_features = ["prefill"] if input_metadata["is_prefill"] else ["decode"]
                for feature_decorator in self._feature_dict.values():
                    if feature_decorator.is_activated(input_metadata) and \
                    feature_decorator.is_stackable(current_engine_features):
                        if feature_decorator.need_additional_engine:
                            current_engine_features.append(feature_decorator.feature_name)
                        feature_decorator.modify_inputs(engine_inputs, engine_outputs,
                                                        engine_runtime_params, input_metadata)
                return current_engine_features

        return EngineFromNetwork
    return class_decorator


class EngineWrapper:
    def __init__(self, feature_list, input_keys, args) -> None:
        self.feature_list: list[str] = feature_list
        self.input_keys: set[str] = input_keys
        self.args: dict = args

        self.engine = None

    def set_weights(self, weight_dict: dict):
        self.engine.set_weights(weight_dict)
    
    def set_kv_caches(self, cache):
        self.engine.set_kv_caches(cache)

    def execute(self, engine_inputs, engine_outputs, engine_runtime_params, **kwargs):
        try:
            self.engine.forward(engine_inputs, engine_outputs, engine_runtime_params)
        except KeyError as e:
            raise RuntimeError("Engine execution error. "
                               "Enable log: export ASCEND_GLOBAL_LOG_LEVEL=3, "
                               "export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find the first error. "
                               "For more details, see the MindIE official document.") from e