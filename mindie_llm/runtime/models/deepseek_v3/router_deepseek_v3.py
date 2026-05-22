# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
import os

from mindie_llm.runtime.models.base.router import BaseRouter
from mindie_llm.runtime.utils.helpers.safety.hf import safe_get_tokenizer_from_pretrained
from mindie_llm.runtime.models.deepseek_v3.input_builder_deepseek_v3 import DeepseekV3InputBuilder
from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.utils.helpers.parameter_validators import (
    IntParameterValidator, FloatParameterValidator, BooleanParameterValidator, RangeParamaterValidator, 
    DictionaryParameterValidator, Field
)


@dataclass
class DeepseekV3Router(BaseRouter):
    """
    Router class for DeepSeek V3 model configuration and initialization.
    
    This class handles the specific configuration and setup for DeepSeek V3 models,
    including model type conversion, configuration validation, and component initialization.
    """
    def __post_init__(self):
        """
        Post-initialization method to handle model type conversion and configuration updates.
        
        Converts model type from various aliases (deepseek_v3, deepseek_v2, deepseekv3) 
        to a standardized format and updates the configuration accordingly.
        """
        super().__post_init__()
        standard_model_type = "deepseek_v3"
        # models_dict model type convert
        if (self._model_type in [standard_model_type, "deepseek_v2", "deepseekv3"] and
            isinstance(self.load_config.models_dict, dict)):
            try:
                self.load_config.models_dict.update({
                    standard_model_type: self.load_config.models_dict.pop(self._model_type)
                })
                self._model_type = standard_model_type
                self._model_type_cap = ''.join(part.capitalize() for part in self._model_type.split('_'))
            except KeyError as e:
                message = "The 'models' field does not contain {self._model_type}. Please check."
                logger.warning(f'{message}, exception info: {e}')
        
        self.llm_config.update(self.load_config.models_dict, allow_new_keys=True, current_path='models')
        self.llm_config.merge_models_config(self._model_type)

    @classmethod
    def get_llm_config_validators(cls):
        """
        Override the base class method to add DeepSeek V3 specific configuration validators.
        
        Returns a dictionary of validators for DeepSeek V3 model configuration parameters.
        
        Returns:
            dict: Configuration validators for DeepSeek V3 model
        """
        hccl_buffsize_env = os.getenv("HCCL_BUFFSIZE")
        hccl_buffsize_env = int(hccl_buffsize_env) if hccl_buffsize_env is not None else 512
        llm_config_validators = super().get_llm_config_validators()
        deepseekv3_config_validators = DictionaryParameterValidator({
            "eplb": DictionaryParameterValidator({
                "level": RangeParamaterValidator(range_list=[0, 1, 2, 3]),
                "num_redundant_experts": IntParameterValidator(Field(ge=0, le=512)),
                "aggregate_threshold": IntParameterValidator(Field(ge=1)),  # Minimum threshold: 1 (statistical period)
                "buffer_expert_layer_num": IntParameterValidator(Field(ge=1)),  # 1 ≤ value < number of MoE layers
                "num_expert_update_ready_countdown": IntParameterValidator(Field(gt=0)),  # 0 < value < threshold / 2
            }),
            "ep_level": RangeParamaterValidator(range_list=[1, 2]),
            "parallel_options": DictionaryParameterValidator({
                "hccl_moe_ep_buffer": IntParameterValidator(Field(ge=max(512, hccl_buffsize_env))),
            }),
            "enable_dispatch_combine_v2": BooleanParameterValidator(),
            "communication_backend": DictionaryParameterValidator({
                "decode": RangeParamaterValidator(range_list=["lccl", "hccl"]),
                "prefill": RangeParamaterValidator(range_list=["lccl", "hccl"]),
            }),
            "enable_gmmswigluquant": BooleanParameterValidator(),
            "enable_oproj_prefetch": BooleanParameterValidator(),
            "enable_mlapo_prefetch": BooleanParameterValidator(),
            "num_dangling_shared_experts": IntParameterValidator(Field(gt=-1), special_values=[-1]),
            "enable_swiglu_quant_for_shared_experts": BooleanParameterValidator(),
            "enable_init_routing_cutoff": BooleanParameterValidator(),
            "topk_scaling_factor": FloatParameterValidator(Field(ge=0.0, le=1.0)),
            "mlp_full_tp": BooleanParameterValidator(),
            "h3p": DictionaryParameterValidator({
                "enable_qkvdown_dp": BooleanParameterValidator(),
                "enable_gating_dp": BooleanParameterValidator(),
                "enable_shared_expert_dp": BooleanParameterValidator(),
                "enable_shared_expert_overlap": BooleanParameterValidator(),
            })
        })

        llm_config_validators["models"]["deepseekv2"] = deepseekv3_config_validators
        return llm_config_validators

    def _get_tokenizer(self):
        """
        Creates and returns the tokenizer for DeepSeek V3 model.
        
        Sets the pad token to be the same as the EOS token.
        
        Returns:
            Tokenizer: Tokenizer for DeepSeek V3 model
        """
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.load_config.tokenizer_path,
            padding_side="left",
            trust_remote_code=False,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _get_input_builder(self):
        """
        Creates and returns the input builder for DeepSeek V3 model.
        
        Returns:
            DeepseekV3InputBuilder: Input builder for DeepSeek V3 model
        """
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            kwargs["max_length"] = self.config.max_position_embeddings
        return DeepseekV3InputBuilder(self.tokenizer, **kwargs)
    
    def _get_tool_calls_parser(self):
        """
        Returns the tool call parser identifier for DeepSeek V3.
        
        Returns:
            str: Identifier for the tool call parser ("deepseek_v3")
        """
        return "deepseek_v3"
