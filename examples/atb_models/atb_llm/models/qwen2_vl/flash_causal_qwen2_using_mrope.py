# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Optional, List, Tuple
from collections import OrderedDict
from enum import Enum
import torch
from torch import nn
from torch.functional import F
from ..base.flash_causal_lm import DistributedType, LwdLayerStatus
from ..qwen2.flash_causal_qwen2 import FlashQwen2ForCausalLM
from ...utils.log import logger
MROPE_SECTION = [16, 24, 24]
MROPE_SECTION_STR = "mrope_section"
POSITION_IDS_THW_LIST_STR = "position_ids_thw_list"
COS_LIST_STR = "cos_list"
SIN_LIST_STR = "sin_list"


class TensorEmbeddingWithoutChecking(nn.Module):
    def __init__(self, prefix: str, weights):
        super().__init__()
        weight = weights.get_whole_tensor(f"{prefix}.weight", dim=0)

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.embedding(input_tensor, self.weight)
        return out

    def get_weights(self, prefix):
        weight_dict = OrderedDict()
        weight_dict[f"{prefix}.weight"] = self.weight.data
        return weight_dict


class FlashQwen2UsingMROPEForCausalLM(FlashQwen2ForCausalLM):
    def __init__(self, config, weights, **kwargs):
        kwargs = {**kwargs}
        kwargs.setdefault("skip_word_embedding", True)
        kwargs.setdefault("transformer_wte_parallel", False)
        super().__init__(config, weights, **kwargs)
        model_prefix = kwargs.get("model_prefix", "model")
        self.transformer.wte = TensorEmbeddingWithoutChecking(
            prefix=f"{model_prefix}.embed_tokens", weights=weights
        )
        for p in self.transformer.wte.parameters():
            p.requires_grad = False

    def update_thw_cos_sin(self, position_ids_thw, mrope_section):
        normal_cos = self.rotary_embedding.get_cos_cached_total()
        normal_sin = self.rotary_embedding.get_sin_cached_total()
        cos = normal_cos[position_ids_thw].clone()
        sin = normal_sin[position_ids_thw].clone()
        mrope_section = mrope_section * 2
        cos_thw = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
        sin_thw = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
        return cos_thw, sin_thw

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the model.'

        Args:
            input_ids (torch.Tensor): The input ids tensor.
            position_ids (torch.Tensor): The position ids tensor.
            is_prefill (bool): Whether the inference mode is prefill.
            kv_cache (List[Tuple[torch.Tensor, torch.Tensor]]): Key-value cache.
            block_tables (torch.Tensor): Input block tables.
            slots (torch.Tensor): Input slots.
            input_lengths (torch.Tensor): Input lengths.
            max_seq_len (torch): Maximum sequence length.
            lm_head_indices (torch.Tensor, optional): LM head indices. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            torch.Tensor: Output logits.
        """
        if not self.weight_initialized:
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()
        self.init_kvcache(kv_cache)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                                    block_tables, slots, input_lengths, max_seq_len,
                                                                    lm_head_indices, **kwargs)
        position_ids_thw_list = kwargs.pop(POSITION_IDS_THW_LIST_STR, None)
        is_not_long_prefill = is_prefill and (not self.long_seq_enable)
        is_exist_position_ids_thw_list = position_ids_thw_list is not None and position_ids_thw_list[0] is not None
        if is_not_long_prefill and is_exist_position_ids_thw_list:
            mrope_section = kwargs.pop(MROPE_SECTION_STR, MROPE_SECTION)
            cos_list, sin_list = [], []
            for position_ids_thw in position_ids_thw_list:
                cos_thw, sin_thw = self.update_thw_cos_sin(position_ids_thw, mrope_section)
                cos_list.append(cos_thw)
                sin_list.append(sin_thw)
            acl_inputs[2] = torch.cat(cos_list, dim=0)
            acl_inputs[3] = torch.cat(sin_list, dim=0)
            new_position_ids = torch.arange(
                input_lengths.sum(), dtype=position_ids.dtype, device=position_ids.device
            )
            acl_inputs[1] = new_position_ids
            del position_ids_thw_list
            kwargs.update({COS_LIST_STR: acl_inputs[2]})
            kwargs.update({SIN_LIST_STR: acl_inputs[3]})

        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill, **kwargs)
        return logits