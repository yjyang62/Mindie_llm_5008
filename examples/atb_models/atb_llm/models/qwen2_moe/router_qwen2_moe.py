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
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from ..base.router import BaseRouter
from .configuration_qwen2_moe import Qwen2MoeConfig
from ...utils.parameter_validators import (RangeParamaterValidator, DictionaryParameterValidator)
from ...utils.log import logger
from ...utils.log.error_code import ErrorCode


@dataclass
class Qwen2moeRouter(BaseRouter):
    @property
    def config(self):
        config = Qwen2MoeConfig.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code
        )
        self.check_config(config)
        return config

    @classmethod
    def get_llm_config_validators(cls):
        '''
        override this method to add validators for qwen_moe config
        '''
        llm_config_validators = super().get_llm_config_validators()
        qwen_moe_config_validators = DictionaryParameterValidator({
            "eplb": DictionaryParameterValidator({
                "level": RangeParamaterValidator(range_list=[0, 1, 2, 3])
            }),
            "ep_level": RangeParamaterValidator(range_list=[1, 2])
        })

        llm_config_validators["models"]["qwen_moe"] = qwen_moe_config_validators
        return llm_config_validators

    def get_tokenizer(self):
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=self.trust_remote_code
        )

    def check_config(self, config):
        """The validation of values in config."""
        super().check_config(config)
        attribute_ranges = {
            'decoder_sparse_step': (1, 1024),
            'moe_intermediate_size': (1, 2147483647),
            'shared_expert_intermediate_size': (0, 2147483647),
            'num_experts_per_tok': (1, 128),
            'num_experts': (1, 10240),
            'mlp_only_layers': (1, 1024)
        }
        if hasattr(config, "head_dim"):
            attribute_ranges['head_dim'] = (1, 1024)

        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            
            value = getattr(config, attr)
            values = [value] if not isinstance(value, list) else value
            
            for val in values:
                if not (min_val <= val <= max_val):
                    msg = f"config.{attr} = {val}, must be between {min_val} and {max_val}"
                    logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                    raise ValueError(msg)