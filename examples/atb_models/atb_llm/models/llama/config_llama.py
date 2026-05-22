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
from typing import Optional

from ...utils.log import message_filter, logger
from ..base.config import BaseConfig


@dataclass
class LlamaConfig(BaseConfig):
    model_type: str = "llama"
    alibi_bias_max: float = 8.0
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    pe_type: str = "ROPE"
    rms_norm_eps: float = 1e-6
    skip_word_embedding: bool = False
    use_cache: bool = True

    num_key_value_heads: Optional[int] = None
    rope_given_inv_feq_str: Optional[str] = None
    rope_keep_local_base_windows: Optional[str] = None
    rope_mscale: Optional[int] = None
    rope_vanilla_theta: Optional[float] = None
    pdmix: Optional[bool] = False

    def __init__(self, **kwargs):
        self.attribute_map = {'max_sequence_length': 'max_position_embeddings', 'epsilon': 'rms_norm_eps'}
        super().__init__(**kwargs)
        if 'model_type' not in kwargs:
            self.model_type = 'llama'
        if 'tie_word_embeddings' not in kwargs:
            self.tie_word_embeddings = False
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if kwargs.get("quantization_config", None) is not None:
            self.pdmix = kwargs["quantization_config"].get("pdmix", False)
        if self.pdmix and self.quantize == "w8a8":
            logger.warning("After 2026/06/30, the existing quantization weight will be degraded. " \
                "To generate the latest version, please upgrade MindStudio ModelSlim.")
            self.quantize = "w8a8_pdmix"


    def validate(self):
        super().validate()
        self.__check_alibi_bias_max()
        self.__check_pe_type()
        self.__check_rope_given_inv_feq_str()
        self.__check_rope_keep_local_base_windows()
        self.__check_rope_mscale()
        self.__check_rope_vanilla_theta()

    def __check_alibi_bias_max(self):
        if not isinstance(self.alibi_bias_max, float):
            error_msg = "`alibi_bias_max`'s factor field must be a float"
            error_msg = message_filter(error_msg)
            raise ValueError(error_msg)
        if self.alibi_bias_max <= 0 or self.alibi_bias_max > 2147483647:
            error_msg = "`alibi_bias_max`'s factor field must be a float within range (0, 2147483647]" \
                        f" got {self.alibi_bias_max}"
            error_msg = message_filter(error_msg)
            raise ValueError(error_msg)
    
    def __check_pe_type(self):
        if self.pe_type not in ["ROPE", "ALIBI"]:
            error_msg = "`pe_type`'s type field must be one of ['ROPE', 'ALIBI']"
            error_msg = message_filter(error_msg)
            raise ValueError(error_msg)

    def __check_rope_given_inv_feq_str(self):
        if self.rope_given_inv_feq_str is not None:
            try:
                [float(digit) for digit in self.rope_given_inv_feq_str.split(',')]
            except Exception:
                error_msg = "`rope_given_inv_feq_str` must be a string separated by float numbers"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg) from Exception

    def __check_rope_keep_local_base_windows(self):
        if self.rope_keep_local_base_windows is not None:
            try:
                [int(digit) for digit in self.rope_keep_local_base_windows.split(',')]
            except Exception:
                error_msg = "`rope_keep_local_base_windows` must be a string separated by int numbers"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg) from Exception
    
    def __check_rope_mscale(self):
        if self.rope_mscale is not None:
            if not isinstance(self.rope_mscale, int):
                error_msg = "`rope_mscale`'s factor field must be a int"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)
            if self.rope_mscale < 1 or self.rope_mscale > 2147483647:
                error_msg = "`rope_mscale`'s factor field must be a int within range [1, 2147483647], " \
                            f"got {self.rope_mscale}"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)
    
    def __check_rope_vanilla_theta(self):
        if self.rope_vanilla_theta is not None:
            if not isinstance(self.rope_vanilla_theta, float):
                error_msg = "`rope_vanilla_theta`'s factor field must be a float"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)
            if self.rope_vanilla_theta < 1 or self.rope_vanilla_theta > 2147483647:
                error_msg = "`rope_vanilla_theta`'s factor field must be a float within range [1, 2147483647], " \
                            f"got {self.rope_vanilla_theta}"
                error_msg = message_filter(error_msg)
                raise ValueError(error_msg)