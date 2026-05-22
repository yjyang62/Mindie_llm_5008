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

from ...utils.log import logger
from ..base.config import BaseConfig


@dataclass
class Qwen2Config(BaseConfig):
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    model_type = "qwen2"
    pdmix: bool = False
    attention_bias: bool = True
    use_qk_norm: bool = False

    def __init__(self, **kwargs):
        self.attribute_map = {'max_sequence_length': 'max_position_embeddings', 'epsilon': 'rms_norm_eps'}
        super().__init__(**kwargs)
        if "tie_word_embeddings" in kwargs:
            self.tie_word_embeddings = kwargs.get("tie_word_embeddings")
        if "num_key_value_heads" not in kwargs:
            self.num_key_value_heads = self.num_attention_heads
        if kwargs.get("rope_scaling", None) is not None:
            self.max_position_embeddings = 131072
        if kwargs.get("quantization_config", None) is not None:
            self.pdmix = kwargs["quantization_config"].get("pdmix", False)
        if self.pdmix and self.quantize == "w8a8":
            logger.warning("After 2026/06/30, the existing quantization weight will be degraded. " \
                "To generate the latest version, please upgrade MindStudio ModelSlim.")
            self.quantize = "w8a8_pdmix"
