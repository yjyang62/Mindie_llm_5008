# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass, field

from atb_llm.utils.configuration_utils import LLMConfig
from atb_llm.models.base.config import BaseConfig, LoraModelConfig
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.op_backend import OpBackend
from atb_llm.utils.quantize.pack_type import QuantType
from atb_llm.nn.parameter import Parameter


def update_kv_head_num(num_kv_heads, tp_group_size):
    if num_kv_heads < tp_group_size:
        repeat_times = tp_group_size // num_kv_heads
    else:
        repeat_times = 1
    num_kv_heads = (num_kv_heads * repeat_times + tp_group_size - 1) // tp_group_size
    return num_kv_heads


def update_position_embedding_type(hf_config):
    pe_type = hf_config.pe_type
    if pe_type == "ROPE":
        position_embedding_type = PositionEmbeddingType.ROPE
    elif pe_type == "ALIBI":
        position_embedding_type = PositionEmbeddingType.ALIBI
        hf_config.rope_type = None
    return position_embedding_type


def update_matmul_nz(soc_info: NPUSocInfo, quantize: QuantType | None):
    enable_matmul_nz = soc_info.soc_version in [223, 225] and \
                       (quantize is None or quantize in [QuantType.FLOAT, QuantType.W8A8,
                                                         QuantType.W8A8_DYNAMIC, QuantType.W8A8_PDMIX])
    soc_info.matmul_nd_nz = enable_matmul_nz
    Parameter.soc_info = soc_info
    return enable_matmul_nz


@dataclass
class ModelStatus:
    enable_matmul_nz: bool
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    position_embedding_type: int

    attn_decode_backend: OpBackend
    enable_rope_quant_kvcache: bool = False
    enable_swiglu_quant: bool = False
    enable_lcoc: bool = False
    skip_word_embedding: bool = False

    def __post_init__(self):
        self.validate()

    @classmethod
    def from_config(cls, mindie_llm_config):
        hf_config = mindie_llm_config.hf_config
        mapping = mindie_llm_config.mapping
    
        head_dim = hf_config.head_dim if hasattr(hf_config, "head_dim") else \
            hf_config.hidden_size // hf_config.num_attention_heads

        num_attention_heads = (hf_config.num_attention_heads + mapping.attn_tp.group_size - 1) \
            // mapping.attn_tp.group_size

        num_key_value_heads = update_kv_head_num(hf_config.num_key_value_heads, mapping.attn_tp.group_size)

        num_hidden_layers = len(mapping.pp_layers(hf_config.num_hidden_layers)) if mapping.has_pp() \
            else hf_config.num_hidden_layers
        
        position_embedding_type = update_position_embedding_type(hf_config)

        enable_matmul_nz = update_matmul_nz(mindie_llm_config.soc_info, hf_config.quantize)
        attn_decode_backend = OpBackend.ACLNN if hf_config.quantization_config.kv_quant_type is not None \
                              else OpBackend.ATB,

        model_status = cls(
            enable_matmul_nz=enable_matmul_nz,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            position_embedding_type=position_embedding_type,
            attn_decode_backend=attn_decode_backend
        )

        return model_status
    
    def validate(self):
        pass


@dataclass
class MindIELLMConfig:
    hf_config: BaseConfig
    llm_config: LLMConfig
    mapping: Mapping
    lora_config: LoraModelConfig = field(init=False)

    def __post_init__(self):
        self.soc_info = NPUSocInfo()
        self._modify_hf_config()
        self._cross_validate()

    def _modify_hf_config(self):
        if getattr(self.hf_config, "num_key_value_heads") is None:
            setattr(self.hf_config, "num_key_value_heads", self.hf_config.num_attention_heads)
        if self.hf_config.rope_scaling.rope_type is None:
            self.hf_config.rope_scaling.rope_type = self.hf_config.rope_scaling.type
        if not hasattr(self.hf_config, "rope_theta"):
            setattr(self.hf_config, "rope_theta", 10000.0)
        if not hasattr(self.hf_config, "alibi_bias_max"):
            setattr(self.hf_config, "alibi_bias_max", 8.0)
        if not hasattr(self.hf_config, "pe_type"):
            setattr(self.hf_config, "pe_type", "ROPE")
        if not hasattr(self.hf_config, "epsilon"):
            setattr(self.hf_config, "epsilon", self.hf_config.rms_norm_eps)
    
    def _cross_validate(self):
        pass