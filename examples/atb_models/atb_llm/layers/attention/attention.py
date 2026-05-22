# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math

from ... import nn
from ...nn.functional import split, reshape_and_cache
from ...utils.loader.safetensor_file_loader import SafetensorFileLoader
from ...layers import InferenceMode
from ...models.base.config import BaseConfig
from ...models.base.mindie_llm_config import ModelStatus
from ...nn.distributed import distributed as dist


class Attention(nn.Module):
    def __init__(
            self, config: BaseConfig, file_loader: SafetensorFileLoader, prefix: str,
            config_metadata: ModelStatus, **kwargs):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.mapping = file_loader.mapping

        self.head_size = config_metadata.head_dim
        self.num_heads_per_rank = config_metadata.num_attention_heads
        self.num_key_value_heads_per_rank = config_metadata.num_key_value_heads

        self.qkv = None
        self.dense = None
        self.q_norm = None
        self.k_norm = None

    def forward(self, inputs, cos_table, sin_table, k_cache, v_cache, slot_mapping=None, mask=None, seq_lens=None,
                block_table=None, token_offset=None, layer_ids=None, is_prefill: bool = True, **kwargs):
        """
        Args:
            kwargs:
                enable_lora: exists when LoRA is enabled
                group_list: exists when multi-LoRA is activated
        """
        if token_offset is None and layer_ids is None:
            return self.forward_pa(inputs, cos_table, sin_table, k_cache, v_cache, slot_mapping=slot_mapping,
                                   mask=mask, seq_lens=seq_lens, block_table=block_table,
                                   is_prefill=is_prefill, **kwargs)
        else:
            raise NotImplementedError("The method for non-None token_offset and layer_ids is not implemented yet.")

    def forward_pa(self, inputs, cos_table, sin_table, k_cache, v_cache, slot_mapping=None, mask=None,
                    seq_lens=None, block_table=None, is_prefill: bool = True, **kwargs):
        inference_mode = InferenceMode.PREFILL if is_prefill else InferenceMode.DECODE
        if len(self.qkv) == 1:
            qkv_out = self.qkv(inputs, inference_mode=inference_mode, **kwargs)[0]
            q_size = self.head_size * self.num_heads_per_rank
            kv_size = self.head_size * self.num_key_value_heads_per_rank
            q, k, v = split(qkv_out, dim=1, split_size_or_sections=[q_size, kv_size, kv_size])
        else:
            q, k, v = self.qkv(inputs, inference_mode=inference_mode, **kwargs)

        if self.q_norm is not None:
            q = q.reshape(lambda org_shape: [org_shape[0], self.num_heads_per_rank, self.head_size])
            q = self.q_norm(q)
            q = q.reshape(lambda org_shape: [org_shape[0], self.head_size * self.num_heads_per_rank])
        if self.k_norm is not None:
            k = k.reshape(lambda org_shape: [org_shape[0], self.num_key_value_heads_per_rank, self.head_size])
            k = self.k_norm(k)
            k = k.reshape(lambda org_shape: [org_shape[0], self.head_size * self.num_key_value_heads_per_rank])

        q_out, k_out = nn.functional.rope(q, k, cos_table, sin_table, seq_lens)

        q_out_ = q_out.reshape(lambda org_shape: [org_shape[0], self.num_heads_per_rank, self.head_size])
        k_out_ = k_out.reshape(lambda org_shape: [org_shape[0], self.num_key_value_heads_per_rank, self.head_size])
        v_ = v.reshape(lambda org_shape: [org_shape[0], self.num_key_value_heads_per_rank, self.head_size])

        reshape_and_cache(k_out_, v_, k_cache, v_cache, slot_mapping)

        args = {
            "q": q_out_,
            "head_num": self.num_heads_per_rank, 
            "kv_head_num": self.num_key_value_heads_per_rank, 
            "kv_lens": seq_lens, 
            "qk_scale": 1.0 / math.sqrt(self.head_size),
        }
        if is_prefill:
            args.update({
                "k": k_out_,
                "v": v_,
                "mask": mask,
                "mask_type": nn.functional.MaskType.NORM,
                "is_triu_mask": True
            })
        else:
            args.update({
                "k_cache": k_cache, 
                "v_cache": v_cache, 
                "block_table": block_table
            })
        attn_score = nn.functional.paged_attention(**args)

        attn_score_ = attn_score.reshape(lambda org_shape: [org_shape[0], org_shape[1] * org_shape[2]])

        attn_out = self.dense(attn_score_, inference_mode=inference_mode, **kwargs)
        attn_out_ = dist.all_reduce(attn_out, process_group=self.mapping.attn_tp.process_group)
        return attn_out_
