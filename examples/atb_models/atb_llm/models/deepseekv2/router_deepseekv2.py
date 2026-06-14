# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

from ..base.router import BaseRouter
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from .flash_causal_deepseekv2 import DeepseekV2Config
from ..base.reasoning_parser import CommonReasoningParser
from .input_builder_deepseekv2 import Deepseekv2InputBuilder

from ...utils.log import logger
from ...utils.log.error_code import ErrorCode

from ...utils.parameter_validators import (
    IntParameterValidator, FloatParameterValidator, BooleanParameterValidator, RangeParamaterValidator, 
    DictionaryParameterValidator, Field
)


@dataclass
class Deepseekv2Router(BaseRouter):
    @classmethod
    def get_llm_config_validators(cls):
        '''
        override this method to add validators for deepseekv2 config
        '''
        hccl_buffsize_env = os.getenv("HCCL_BUFFSIZE")
        hccl_buffsize_env = int(hccl_buffsize_env) if hccl_buffsize_env is not None else 512
        llm_config_validators = super().get_llm_config_validators()
        deepseekv2_config_validators = DictionaryParameterValidator({
            "eplb": DictionaryParameterValidator({
                "level": RangeParamaterValidator(range_list=[0, 1, 2, 3]),
                "num_redundant_experts": IntParameterValidator(Field(ge=0, le=512)),
                "aggregate_threshold": IntParameterValidator(Field(ge=1)),  # 最小门限值1，作为统计周期
                "buffer_expert_layer_num": IntParameterValidator(Field(ge=1)),  # 大于等于1，小于moe层数
                "num_expert_update_ready_countdown": IntParameterValidator(Field(gt=0)),  # 检测周期，大于0，小于门限值一半
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

        llm_config_validators["models"]["deepseekv2"] = deepseekv2_config_validators
        return llm_config_validators

    def get_config(self):
        config = DeepseekV2Config.from_dict(self.config_dict)
        self.check_config_deepseekv2(config)
        if config.quantization_config.fa_quant_type is not None:
            self.llm_config.llm.kv_cache_options.enable_nz = True
                
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            padding_side="left",
            trust_remote_code=False,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def get_reasoning_parser(self):
        """initialize reasoning parser"""
        def check_token_exists(token):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id == self.tokenizer.unk_token_id:
                    return None
                return token_id
            except Exception:
                return None
        
        start_reasoning_token_id = check_token_exists("<think>")
        end_reasoning_token_id = check_token_exists("</think>")
        if start_reasoning_token_id is not None and end_reasoning_token_id is not None:
            self.config.is_reasoning_model = True
            self.config.reasoning_config.start_reasoning_token_id = start_reasoning_token_id
            self.config.reasoning_config.end_reasoning_token_id = end_reasoning_token_id
        else:
            self.config.is_reasoning_model = False

        return CommonReasoningParser(self.config)

    def get_input_builder(self):
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        return Deepseekv2InputBuilder(self.tokenizer, **kwargs)

    def check_config_deepseekv2(self, config):
        super().check_config(config)
        attribute_ranges = {
            "num_experts_per_tok": (1, 256),
            "n_shared_experts": (0, 256),
            "first_k_dense_replace": (0, 61),
            "n_routed_experts ": (2, 256),
            "q_lora_rank": (1, 1536),
            "qk_nope_head_dim": (1, 128),
            "qk_rope_head_dim": (1, 64),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                msg = f"config.{attr} = {value}, must be between {min_val} and {max_val}"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)

        if getattr(config, "num_experts_per_tok") > getattr(config, "n_routed_experts"):
            msg = f"config.num_experts_per_tok should be less than config.n_routed_experts, " \
                  f"but {config.num_experts_per_tok=}, {config.n_routed_experts=}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if getattr(config, "first_k_dense_replace") > getattr(config, "num_hidden_layers"):
            msg = f"config.first_k_dense_replace should be less than config.num_hidden_layers, " \
                  f"but {config.first_k_dense_replace=}, {config.num_hidden_layers=}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if config.topk_method not in ["greedy", "group_limited_greedy", "noaux_tc"]:
            msg = "`topk_method`'s type field must be one of ['greedy', 'group_limited_greedy', 'noaux_tc'], " \
                  f"got {config.topk_method}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        # 校验topk参数是否匹配
        if config.topk_method == "greedy" and config.topk_group != config.n_group and config.n_group != 1:
            msg = f"`topk_method is `greedy`, please set `topk_group=1` and `n_group=1`, " \
                  f"got topk_group={config.topk_group}, n_group={config.n_group}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
    
    def get_tool_call_parser(self):
        return "deepseek_v3"