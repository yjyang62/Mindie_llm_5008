# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass

from .config_kimi_k2 import KimiK2Config
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from ..deepseekv2.router_deepseekv2 import Deepseekv2Router
from ...utils.log import logger
from ...utils.log.error_code import ErrorCode
from .input_builder_kimi_k2 import KimiK2InputBuilder


@dataclass
class Kimik2Router(Deepseekv2Router):

    def __post_init__(self):
        # kimi k2 继承 deepseekv2
        super().__post_init__()
        self.model_type = "deepseekv2"
        self.model_type_cap = self.model_type.capitalize()

    def check_config_kimi_k2(self, config):
        super().check_config(config)
        attribute_ranges = {
            "num_experts_per_tok": (1, 384),
            "n_shared_experts": (0, 384),
            "first_k_dense_replace": (0, 61),
            "n_routed_experts ": (2, 384),
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

    def get_config(self):
        config = KimiK2Config.from_dict(self.config_dict)
        self.check_config_kimi_k2(config)
        if config.quantization_config.fa_quant_type is not None:
            self.llm_config.llm.kv_cache_options.enable_nz = True
        return config

    def get_tokenizer(self):
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            trust_remote_code=self.trust_remote_code
        )

        original_decode = tokenizer.decode
        
        def custom_decode(token_ids, skip_special_tokens=False, **kwargs):
            if not isinstance(token_ids, list):
                token_ids = token_ids.tolist()
            return original_decode(token_ids, **kwargs)

        tokenizer.decode = custom_decode
        return tokenizer

    def get_input_builder(self):
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        return KimiK2InputBuilder(self.tokenizer, **kwargs)