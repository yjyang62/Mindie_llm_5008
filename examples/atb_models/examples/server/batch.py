# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import List, Optional
import torch
import numpy as np
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger
from atb_llm.utils.mapping import Mapping
from .request import Request


class Mtp:
    input_ids: torch.Tensor | None
    position_ids: torch.Tensor | None
    is_prefill: torch.bool
    block_tables: torch.Tensor | None

    slots: torch.Tensor | None 
    dp_rank_ids: torch.Tensor | None
    input_lengths: torch.Tensor | None
    max_seq_len: int
    lm_head_indices: torch.Tensor | None
    final_hidden_state_mtp: torch.Tensor | None
    batch_input_ids_mtp: torch.Tensor | None


    def __init__(self):
        self.input_ids = None
        self.position_ids = None
        self.is_prefill = True
        self.block_tables = None
        self.slots = None
        self.dp_rank_ids = None
        self.input_lengths = None
        self.max_seq_len = 0
        self.lm_head_indices = None
        self.final_hidden_state_mtp = None
        self.accepted_lens = None
        self.batch_input_ids_mtp = None


class Batch:
    req_ids: List[int]
    req_list: List[Request]
    batch_num: int

    cu_seqlen_prefill: torch.Tensor
    batch_input_ids: torch.Tensor
    batch_adapter_ids: list
    batch_position_ids: torch.Tensor

    batch_block_tables: torch.Tensor
    batch_slots_tables: torch.Tensor
    batch_slot_indices: torch.Tensor

    batch_dp_rank_ids: torch.Tensor

    context_length: torch.Tensor
    max_s: int
    lm_head_indices: torch.Tensor

    dep_inputs: Optional[list] = None
    token_size_per_dp_group: Optional[torch.Tensor] = None
    shard_effective_token_indices: Optional[torch.Tensor] = None
    max_dp_batch_size: int = None

    mapping: Mapping
    batch_sp_rank_ids: torch.Tensor
    context_length_sp: torch.Tensor = None
    lm_head_indices_sp: torch.Tensor = None
    post_lmhead_gather_indices: torch.Tensor

    batch_cp_rank_ids: torch.Tensor
    context_length_cp: torch.Tensor  # Record the global sequence length of the request.

    def __init__(self, req_list: List[Request], **kwargs):
        self.req_list = req_list
        self.batch_num = len(req_list)
        
        self.mapping = kwargs.get("mapping", None)

        self.mtp = Mtp()
        self.dep_inputs = None
        self.token_size_per_dp_group = None
        self.shard_effective_token_indices = None
        self.req_ids = [req.req_id for req in req_list]
        input_ids_list = []
        input_ids_list_mtp = []
        adapter_ids_list = []
        position_ids_list = []
        slot_indices_list = []
        dp_rank_list = []
        sp_rank_list = []
        cp_rank_list = []
        context_length_list = []
        context_length_list_sp = []
        context_length_list_cp = []
        multi_context_length_list = []
        self.max_s = 0
        slot_offset = 0
        self.multi_context_length = None
        self.max_dp_batch_size = None

        # for llama3.2
        cross_atten_slot_indices_list = []
        cross_attn_mask_list = []
        cross_attn_context_length_list = []
        cross_attn_full_row_mask_list = []
        multi_modal_list = []
        num_vision_tokens = 0
        for req in self.req_list:
            context_length = req.input_ids.size(0)
            context_length_sp = req.input_length_sp
            input_ids_list.append(req.input_ids)
            input_ids_list_mtp.append(torch.roll(req.input_ids, shifts=-1, dims=0))
            adapter_ids_list.append(req.adapter_id)
            position_ids = torch.arange(context_length, dtype=torch.long)

            if self.mapping is not None and self.mapping.has_attn_inner_sp():
                slot_indices = torch.arange(context_length_sp, dtype=torch.long) + slot_offset
            else:
                slot_indices = torch.arange(context_length, dtype=torch.long) + slot_offset
            slot_indices_list.append(slot_indices)
            context_length_list.append(context_length)
            context_length_list_sp.append(context_length_sp)
            context_length_list_cp.append(context_length * \
                self.mapping.attn_cp.group_size if self.mapping is not None else 1)
            self.max_s = max(self.max_s, context_length)
            if getattr(req, 'cross_attention_mask', None) is not None:   # for llama3.2
                cross_atten_slot_indices_list.append(
                    torch.arange(req.image_context_length, dtype=torch.long) + slot_offset
                )
                cross_attn_mask_list.append(req.cross_attention_mask)
                cross_attn_context_length_list.extend([req.image_context_length] * context_length)
                cross_attn_full_row_mask_list.append(req.full_text_row_masked_out_mask)
                multi_modal_list.append(req.multi_modal_inputs)
                num_vision_tokens = req.num_vision_tokens
            slot_offset += req.need_slots
            dp_rank_list.append(req.dp_rank)
            sp_rank_list.append(req.sp_rank)
            cp_rank_list.append(req.cp_rank)
            if getattr(req, "position_ids", None) is not None:
                position_ids = req.position_ids
            position_ids_list.append(position_ids)
            if getattr(req, "context_length", None) is not None:
                multi_context_length_list.append(req.context_length)
        self.min_s = min(context_length_list)
        if multi_context_length_list:
            self.multi_context_length = torch.tensor(multi_context_length_list, dtype=torch.int64)
        self.cu_seqlen_prefill = torch.tensor([1])
        self.batch_input_ids = torch.concat(input_ids_list, dim=0)
        self.mtp.batch_input_ids_mtp = torch.concat(input_ids_list_mtp, dim=0)
        self.batch_adapter_ids = adapter_ids_list
        self.batch_position_ids = torch.concat(position_ids_list, dim=0)
        self.batch_block_tables: None | torch.Tensor = None
        self.batch_slots_tables: None | torch.Tensor = None
        self.batch_slot_indices = torch.concat(slot_indices_list, dim=0)
        self.batch_dp_rank_ids = torch.tensor(dp_rank_list)
        self.batch_sp_rank_ids = torch.tensor(sp_rank_list)
        self.batch_cp_rank_ids = torch.tensor(cp_rank_list)
        self.context_length = torch.tensor(context_length_list, dtype=torch.int64)
        self.lm_head_indices = torch.cumsum(self.context_length, dim=0) - 1
        self.post_lmhead_gather_indices = torch.tensor([1])

        self.context_length_sp = torch.tensor(context_length_list_sp, dtype=torch.int32)
        self.lm_head_indices_sp = torch.cumsum(self.context_length_sp, dim=0) - 1
        self.context_length_cp = torch.tensor(context_length_list_cp, dtype=torch.int32)
        self.step = 0
        self.cp_cache_rank = -1
        self.sp_cache_rank = -1

        if cross_attn_mask_list: # for llama3.2
            self.batch_cross_attn_slot_indices = torch.concat(cross_atten_slot_indices_list, dim=0)

            self.batch_cross_attn_context_length = torch.tensor(cross_attn_context_length_list, dtype=torch.int64)
            self.batch_cross_attn_full_row_mask = torch.concat(cross_attn_full_row_mask_list, dim=0)
            cross_attention_mask = torch.concat(cross_attn_mask_list, dim=0)

            dtype = self.batch_cross_attn_full_row_mask.dtype
            cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=-1)
            inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
            cross_attention_mask = inverted_cross_attn_mask.masked_fill(
                inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
            )
            self.batch_cross_attn_mask = cross_attention_mask
            self.batch_multi_modal = multi_modal_list
        else:
            self.batch_cross_attn_mask = None
        
        self.after_compress_seqlens = None
        self.pffset_index = None
        self.seq_len = None
        self.global_block_int_offsets_per_head = None
        self.wins = None
        self.ra_offset = None
        self.in_reshape_seqlen = None
        self.batch_max_blocks_per_layer = None
        self.mod_mask = None

    @classmethod
    def concatenate(cls, batches: List["Batch"]):
        req_ids = []
        req_list = []
        batch_num = 0
        input_ids_list = [batch.batch_input_ids for batch in batches]

        adapter_ids_list = []
        for batch in batches:
            for adapter in batch.batch_adapter_ids:
                adapter_ids_list.append(adapter)
        position_ids_list = [batch.batch_position_ids for batch in batches]
        block_tables_list = []
        slots_tables_list = [batch.batch_slots_tables for batch in batches]
        slot_indices_list = []
        batch_input_ids_mtps = [batch.mtp.batch_input_ids_mtp for batch in batches]
        accepted_lenss = [batch.mtp.accepted_lens for batch in batches]
        dp_rank_list = [batch.batch_dp_rank_ids for batch in batches]
        sp_rank_list = [batch.batch_sp_rank_ids for batch in batches]
        cp_rank_list = [batch.batch_cp_rank_ids for batch in batches]
        context_length_list = [batch.context_length for batch in batches]
        context_length_list_sp = [batch.context_length_sp for batch in batches]
        context_length_list_cp = [batch.context_length_cp for batch in batches]
        max_s = 0
        if ENV.omni_attention_enable:
            for batch in batches:
                req_ids.extend(batch.req_ids)
                req_list.extend(batch.req_list)
                batch_num += batch.batch_num
                
            batches[0].min_s = min([tmp.min_s for tmp in batches])

            block_tables_list = [tmp.batch_block_tables for tmp in batches]
            batches[0].allocate_index = torch.concat([tmp.allocate_index for tmp in batches])
            batches[0].after_compress_seqlens = torch.concat([tmp.after_compress_seqlens for tmp in batches], dim=1)
            batches[0].global_block_int_offsets_per_head = torch.concat(
                [tmp.global_block_int_offsets_per_head for tmp in batches],
                dim=1
            )
            batches[0].seq_len = torch.concat([tmp.seq_len for tmp in batches], dim=1)
            batches[0].pffset_index = torch.concat([tmp.pffset_index for tmp in batches], dim=1)
            batches[0].ra_offset = torch.concat([tmp.ra_offset for tmp in batches], dim=1)
            batches[0].batch_max_blocks_per_layer = sum([tmp.batch_max_blocks_per_layer for tmp in batches])
            batches[0].wins = torch.concat([tmp.wins for tmp in batches])
            batches[0].in_reshape_seqlen = torch.concat([tmp.in_reshape_seqlen for tmp in batches])
        else:
            max_block = 0
            for batch in batches:
                req_ids.extend(batch.req_ids)
                req_list.extend(batch.req_list)
                batch_num += batch.batch_num
                max_s = max(max_s, batch.max_s)
                max_block = max(max_block, batch.batch_block_tables.size(1))

            slot_offset = 0
            for batch in batches:
                cur_block = batch.batch_block_tables.size(1)
                if cur_block < max_block:
                    zero = torch.zeros(batch.batch_num, max_block - cur_block, dtype=torch.long)
                    batch.batch_block_tables = torch.concat([batch.batch_block_tables, zero], dim=-1)
                block_tables_list.append(batch.batch_block_tables)
                slot_indices_list.append(batch.batch_slot_indices + slot_offset)
                slot_offset += batch.batch_slots_tables.size(0)
        
        if not ENV.omni_attention_enable: # 修复Omni特性无法ENABLE_PD_SPLIT
            batch_slots_tables = torch.concat(slots_tables_list, dim=0)
            batch_slot_indices = torch.concat(slot_indices_list, dim=0)
        if ENV.deepseek_mtp:
            batches[0].mtp.batch_input_ids_mtp = torch.concat(batch_input_ids_mtps, dim=0)
            batches[0].mtp.accepted_lens = torch.concat(accepted_lenss, dim=0)

        batches[0].req_ids = req_ids
        batches[0].req_list = req_list
        batches[0].batch_num = batch_num
        batches[0].batch_input_ids = torch.concat(input_ids_list, dim=0)
        batches[0].batch_adapter_ids = adapter_ids_list
        batches[0].batch_position_ids = torch.concat(position_ids_list, dim=0)
        batches[0].batch_block_tables = torch.concat(block_tables_list, dim=0)
        batches[0].batch_slots_tables = torch.concat(slots_tables_list, dim=0)
        batches[0].batch_slot_indices = torch.concat(slot_indices_list, dim=0)
        batches[0].batch_dp_rank_ids = torch.concat(dp_rank_list, dim=0)
        batches[0].batch_sp_rank_ids = torch.concat(sp_rank_list, dim=0)
        batches[0].batch_cp_rank_ids = torch.concat(cp_rank_list, dim=0)
        batches[0].context_length = torch.concat(context_length_list, dim=0)
        batches[0].context_length_sp = torch.concat(context_length_list_sp, dim=0)
        batches[0].context_length_cp = torch.concat(context_length_list_cp, dim=0)
        batches[0].max_s = max_s
        if ENV.omni_attention_enable:
            batches[0].batch_block_tables = torch.concat(block_tables_list, dim=1)
            batches[0].batch_slots_tables = torch.concat(slots_tables_list, dim=1)
        else:
            batches[0].batch_block_tables = torch.concat(block_tables_list, dim=0)
            batches[0].batch_slots_tables = batch_slots_tables
            batches[0].batch_slot_indices = batch_slot_indices
        if batches[0].multi_context_length is not None:
            multi_context_length_list = [batch.multi_context_length for batch in batches]
            batches[0].multi_context_length = torch.concat(multi_context_length_list, dim=0)

        if batches[0].batch_cross_attn_mask is not None: # for llama3.2
            batches[0].batch_cross_attn_mask = torch.concat(
                [batch.batch_cross_attn_mask for batch in batches], dim=0
            )
            batches[0].batch_cross_attn_context_length = torch.concat(
                [batch.batch_cross_attn_context_length for batch in batches], dim=0
            )
            batches[0].batch_cross_attn_full_row_mask = torch.concat(
                [batch.batch_cross_attn_full_row_mask for batch in batches], dim=0
            )

        while len(batches) > 1:
            del batches[1]

    def get_omni_input_tensor(self, cache_manager):
        # reshape 
        forward_kwargs = {
            "in_attn_seqlens": self.after_compress_seqlens.view(cache_manager.num_layers, -1),
            "wins": self.wins.view(-1),
            "pffset_index": self.pffset_index.view(cache_manager.num_layers, -1),
            "ra_offset": self.ra_offset.view(cache_manager.num_layers, -1, cache_manager.block_size),
            "in_reshape_seqlen": self.in_reshape_seqlen,
            "batch_block_tables": self.batch_block_tables.view(
                cache_manager.num_layers, self.batch_num * cache_manager.num_kv_heads, -1
            ),
            "batch_slots_tables": self.batch_slots_tables.view(cache_manager.num_layers, -1)
        }
        return forward_kwargs
    
    def transpose_input_tensor(self):
        self.after_compress_seqlens = self.after_compress_seqlens.transpose(0, 1)
        self.pffset_index = self.pffset_index.transpose(0, 1)
        self.batch_block_tables = self.batch_block_tables.transpose(0, 1)
        self.batch_slots_tables = self.batch_slots_tables.transpose(0, 1)
        self.seq_len = self.seq_len.transpose(0, 1)
        self.global_block_int_offsets_per_head = self.global_block_int_offsets_per_head.transpose(0, 1)

    def contiguous_input_tensor(self):
        self.after_compress_seqlens = self.after_compress_seqlens.contiguous()
        self.wins = self.wins.contiguous()
        self.pffset_index = self.pffset_index.contiguous()
        self.ra_offset = self.ra_offset.contiguous()
        self.in_reshape_seqlen = self.in_reshape_seqlen.contiguous()
        self.batch_block_tables = self.batch_block_tables.contiguous()
        self.batch_slots_tables = self.batch_slots_tables.contiguous()
        self.seq_len = self.seq_len.contiguous()
        self.global_block_int_offsets_per_head = self.global_block_int_offsets_per_head.contiguous()

    def omni_decoder_prepare(self, cache_manager, shift_windows=True):
        if self.batch_num > cache_manager.batch_size:
            raise ValueError(f"Current Batch {self.batch_num} is larger than the max BS {cache_manager.batch_size}")

        # 1. For Slots
        if shift_windows:
            if self.min_s >= cache_manager.first_sink:
                # all sample token lengths both large than first sink
                self.seq_len -= cache_manager.first_sink - 1
                self.seq_len %= cache_manager.mod_mask
                self.seq_len += cache_manager.first_sink
            else:
                self.seq_len += 1 # length after compress -1 for slot index
                recircle_batch_mask = self.context_length > cache_manager.first_sink
                self.seq_len[:, recircle_batch_mask] = (
                    self.seq_len[:, recircle_batch_mask] - cache_manager.first_sink
                ) % cache_manager.mod_mask + cache_manager.first_sink
        else:
            self.seq_len += 1 # length after compress -1 for slot index

        self.batch_slots_tables = self.global_block_int_offsets_per_head + self.seq_len

    
        # 2. seqlens For PageAttention, to get the index of blocktables
        self.after_compress_seqlens += 1

        # 3. new kv to kv_cache
        self.in_reshape_seqlen.fill_(1)

        # 4. compress start index to -1 means no need to compress
        self.pffset_index.fill_(-1)

    def filter(self, postprocessor, cache_manager):
        # global GLOBAL_HIDDEN_STATES
        if self.batch_num == 0:
            logger.error("batch.batch_num is 0")
            raise AssertionError
        if ENV.omni_attention_enable:
            self.omni_decoder_prepare(cache_manager, shift_windows=ENV.omni_shift_windows_enable)
        
        finish_num = 0
        finish_list = []
        finish_id_list = []
        finished_block_tables = {}
        for i, req in enumerate(self.req_list):
            if (postprocessor.stopping_criteria(req.out_token_list)) or \
                    len(req.out_token_list) >= postprocessor.max_new_tokens:
                if ENV.omni_attention_enable:
                    cache_manager.free_omni(self.allocate_index[i])
                else:
                    if finished_block_tables.get(req.dp_rank) is None:
                        finished_block_tables[req.dp_rank] = [req.block_tables]
                    else:
                        finished_block_tables[req.dp_rank].append(req.block_tables)
                finish_num += 1
                finish_list.append(i)
                finish_id_list.append(self.req_ids[i])

        if finish_num == 0:
            return 0

        batch_mask = torch.ones(self.batch_num, dtype=torch.int64)
        batch_mask[finish_list] = 0
        remain_batch = batch_mask.nonzero().flatten()

        self.batch_num -= finish_num
        if self.batch_num == 0:
            cache_manager.free_all()
            return finish_num

        for dp_rank in finished_block_tables.keys():
            finished_block_tables[dp_rank] = np.concatenate(finished_block_tables.get(dp_rank))
        cache_manager.free(finished_block_tables)

        if ENV.deepseek_mtp:
            self.batch_input_ids = self.batch_input_ids[remain_batch]
            stop_req_ids = np.array(finish_id_list, dtype=np.int32)
            cache_manager.hidden_state_cache.clear_cache(stop_req_ids)
            self.mtp.accepted_lens = self.mtp.accepted_lens[remain_batch]
            remain_batch_mtp = batch_mask.bool()
            remain_batch_mtp = remain_batch_mtp.view(-1, 1).repeat(1, ENV.deepseek_mtp + 1).view(-1)
            self.mtp.batch_input_ids_mtp = self.mtp.batch_input_ids_mtp[remain_batch_mtp]
        else:
            self.batch_input_ids = self.batch_input_ids[remain_batch]
        self.batch_position_ids = self.batch_position_ids[remain_batch]
        self.batch_dp_rank_ids = self.batch_dp_rank_ids[remain_batch]
        self.batch_sp_rank_ids = self.batch_sp_rank_ids[remain_batch]
        self.batch_cp_rank_ids = self.batch_cp_rank_ids[remain_batch]
        if ENV.omni_attention_enable:
            self.batch_block_tables = self.batch_block_tables[:, remain_batch]
        else:
            self.batch_block_tables = self.batch_block_tables[remain_batch]
        
        if self.global_block_int_offsets_per_head is not None: # For Omni attention
            self.global_block_int_offsets_per_head = self.global_block_int_offsets_per_head[:, remain_batch]
        if self.batch_slots_tables is not None:
            if ENV.omni_attention_enable:  # For Omni attention:
                self.batch_slots_tables = self.batch_slots_tables[:, remain_batch]

        if self.seq_len is not None: # For Omni attention
            self.seq_len = self.seq_len[:, remain_batch]

        if self.pffset_index is not None: # For Omni attention
            self.pffset_index = self.pffset_index[:, remain_batch]

        if self.after_compress_seqlens is not None: # For Omni attention
            self.after_compress_seqlens = self.after_compress_seqlens[:, remain_batch]

        if self.ra_offset is not None: # For Omni attention
            block_dim = self.ra_offset.size(1) // batch_mask.size(0)
            ra_offset_remain_mask = batch_mask.unsqueeze(1).repeat(1, block_dim)
            ra_offset_remain_mask = ra_offset_remain_mask.flatten().bool()
            self.ra_offset = self.ra_offset[:, ra_offset_remain_mask]
            if self.batch_max_blocks_per_layer is not None:
                self.batch_max_blocks_per_layer = int(
                    self.batch_max_blocks_per_layer / batch_mask.size(0) * self.batch_num
                )

        if self.wins is not None: # For Omni attention
            self.wins = self.wins[remain_batch]
        
        if self.in_reshape_seqlen is not None: # For Omni attention
            self.in_reshape_seqlen = self.in_reshape_seqlen[remain_batch]
        context_length = self.context_length[remain_batch]
        context_length_sp = self.context_length_sp[remain_batch]
        context_length_cp = self.context_length_cp[remain_batch]
        self.max_s = int(context_length.max())

        if self.batch_cross_attn_mask is not None: # for llama3.2
            self.batch_cross_attn_mask = self.batch_cross_attn_mask[remain_batch]
            self.batch_cross_attn_context_length = \
                self.batch_cross_attn_context_length[remain_batch]
            self.batch_cross_attn_full_row_mask = \
                self.batch_cross_attn_full_row_mask[remain_batch]

        req_ids = []
        req_list = []
        adapter_ids_list = []
        slots_tables_list = []
        slot_indices_list = []

        slot_offset = 0
        for i, req in enumerate(self.req_list):
            if i in finish_list:
                continue

            req_ids.append(req.req_id)
            req_list.append(req)
            adapter_ids_list.append(req.adapter_id)
            if not ENV.omni_attention_enable:
                slots_tables_list.append(req.slot_tables)
                if self.mapping is not None and self.mapping.has_attn_inner_sp():
                    slot_indices_list.append(int(self.context_length_sp[i]) - 1 + slot_offset)
                else:
                    slot_indices_list.append(int(self.context_length[i]) - 1 + slot_offset)
                slot_offset += req.need_slots

        self.req_ids = req_ids
        self.req_list = req_list
        self.batch_adapter_ids = adapter_ids_list
        if not ENV.omni_attention_enable:
            self.batch_slots_tables = torch.concat(slots_tables_list, dim=0)
            self.batch_slot_indices = torch.tensor(slot_indices_list, dtype=torch.long)
        self.context_length = context_length
        self.context_length_sp = context_length_sp
        self.context_length_cp = context_length_cp
        if ENV.omni_attention_enable:
            self.contiguous_input_tensor()
        if self.multi_context_length is not None:
            self.multi_context_length = self.multi_context_length[remain_batch]

        return finish_num