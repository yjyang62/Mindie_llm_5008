# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import List
from enum import Enum

import pandas as pd
import numpy as np
import torch

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.moe_utils import ExpertParallelDegree
from atb_llm.utils import file_utils
from atb_llm.utils.eplb_expert_data_collect import EplbExpertDataCollect
from atb_llm.utils.moe_utils import EPLBType, save_eplb_data

from .batch import Batch
from .request import Request


MTP = int(ENV.deepseek_mtp)


class DapStreamId(int, Enum):
    BASE_STREAM = 0
    DAP_STREAM = 1
    STREAM_NUM = 2


def next_token_chooser(logits: torch.Tensor):
    return torch.argmax(logits, dim=-1)


def is_pp_activated(model):
    if model.mapping is not None and model.mapping.has_pp():
        return True
    return False


decode_lm_head_indices_with_dp = None
lm_head_indices_with_dp = None


def partition_data_lm_head(model,
        dp_rank_ids: torch.Tensor,
        input_lengths: torch.Tensor):
    dp_rank_ids_numpy = dp_rank_ids.numpy()
    token_size_per_request = np.ones_like(input_lengths) * (MTP + 1)
    lm_head_indices_with_dp_ = np.ones_like(input_lengths)
    skip_dummy_data = np.zeros_like(input_lengths)
    total_num_request = 0

    for i in range(model.mapping.attn_dp.group_size):
        cur_shard_input_lengths = token_size_per_request[dp_rank_ids_numpy == i]
        cur_num_request = cur_shard_input_lengths.shape[0]
        if cur_num_request == 0:
            skip_dummy_data[total_num_request:] += 1
        else:
            lm_head_indices_with_dp_[total_num_request:total_num_request + cur_num_request] = cur_shard_input_lengths
        total_num_request += cur_num_request
    lm_head_indices_with_dp_ = lm_head_indices_with_dp_.cumsum(0) - 1 + skip_dummy_data

    sorted_indices = np.argsort(dp_rank_ids_numpy, kind='stable')
    reverse_indices = np.argsort(sorted_indices, kind='stable')
    lm_head_indices_with_dp_ = lm_head_indices_with_dp_[reverse_indices]
    lm_head_indices_with_dp_ = torch.from_numpy(lm_head_indices_with_dp_).int()

    return lm_head_indices_with_dp_


def partition_data(model,
        dp_rank_ids: torch.Tensor,
        dp_rank_ids_per_token: torch.Tensor,
        dp_rank_ids_per_token_sp: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        num_blocks: int,
        block_size: int,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        input_lengths_sp: torch.Tensor):
    global decode_lm_head_indices_with_dp, lm_head_indices_with_dp
    cur_dp_rank_id_per_token_mask = dp_rank_ids_per_token == model.mapping.attn_dp.rank
    cur_dp_rank_id_per_token_mask_sp = dp_rank_ids_per_token_sp == model.mapping.attn_dp.rank \
        if model.mapping.has_attn_inner_sp() else None
    cur_dp_rank_id_mask = dp_rank_ids == model.mapping.attn_dp.rank
    shard_input_ids = input_ids[cur_dp_rank_id_per_token_mask]
    is_input_empty = False
    if shard_input_ids.numel() == 0:
        # dp组未分配数据，构造陪跑数据
        is_input_empty = True
        shard_input_ids = torch.tensor([1], dtype=torch.int64)
        shard_position_ids = torch.tensor([0], dtype=torch.int32)
        is_prefill_with_dp = is_prefill
        shard_block_tables = torch.tensor([[num_blocks - 1]], dtype=torch.int32)
        shard_slots = torch.tensor([(num_blocks - 1) * block_size], dtype=torch.int32)
        shard_input_lengths = torch.tensor([1], dtype=torch.int32)
        shard_input_lengths_sp = torch.tensor([1], dtype=torch.int32)
        shard_max_seq_len = 1
        q_lens = [1]
    else:
        # 获取当前dp组的数据
        shard_position_ids = position_ids[cur_dp_rank_id_per_token_mask]
        is_prefill_with_dp = is_prefill
        shard_block_tables = block_tables[cur_dp_rank_id_mask]
        shard_slots = slots[cur_dp_rank_id_per_token_mask_sp \
            if model.mapping.has_attn_inner_sp() else cur_dp_rank_id_per_token_mask]
        shard_input_lengths = input_lengths[cur_dp_rank_id_mask]
        shard_input_lengths_sp = input_lengths_sp[cur_dp_rank_id_mask] if model.mapping.has_attn_inner_sp() else None
        shard_max_seq_len = max(shard_input_lengths.tolist())
        q_lens = [MTP + 1] * shard_block_tables.size(0)
    # dp数据合并在模型侧完成，lm_head_indices需包含所有dp组，同时去除陪跑数据
    dp_rank_ids_numpy = dp_rank_ids.numpy()
    token_size_per_request = input_lengths.numpy() if is_prefill else np.ones_like(input_lengths) * (MTP + 1)
    lm_head_indices_with_dp = np.ones_like(input_lengths)
    skip_dummy_data = np.zeros_like(input_lengths)
    total_num_request = 0

    for i in range(model.mapping.attn_dp.group_size):
        cur_shard_input_lengths = token_size_per_request[dp_rank_ids_numpy == i]
        cur_num_request = cur_shard_input_lengths.shape[0]
        if cur_num_request == 0:
            skip_dummy_data[total_num_request:] += 1
        else:
            lm_head_indices_with_dp[total_num_request:total_num_request + cur_num_request] = cur_shard_input_lengths
        total_num_request += cur_num_request
    lm_head_indices_with_dp = lm_head_indices_with_dp.cumsum(0) - 1 + skip_dummy_data

    sorted_indices = np.argsort(dp_rank_ids_numpy, kind='stable')
    reverse_indices = np.argsort(sorted_indices, kind='stable')
    lm_head_indices_with_dp = lm_head_indices_with_dp[reverse_indices]
    lm_head_indices_with_dp = torch.from_numpy(lm_head_indices_with_dp).int()

    positional_args = (
        shard_input_ids, shard_position_ids, is_prefill_with_dp,
        shard_block_tables, shard_slots, shard_input_lengths, shard_input_lengths_sp, shard_max_seq_len, 
        lm_head_indices_with_dp, q_lens, is_input_empty
    )

    return positional_args


def gather_dp_data(model, dp_rank_ids_per_token):
    # attn dp + mlp tp场景下，构造额外输入用于tp前收集所有dp组的输入，dp前进行数据切分
    token_size_per_dp_group = torch.bincount(dp_rank_ids_per_token, minlength=model.mapping.attn_dp.group_size)
    token_size_per_dp_group = torch.where(token_size_per_dp_group == 0, 1, token_size_per_dp_group)

    # 用于dp前数据切分：从所有请求按dp组排列后的token index中，选取当前dp组的token index，包含陪跑数据
    end_indices = torch.cumsum(token_size_per_dp_group, dim=0)
    start_indices = end_indices - token_size_per_dp_group
    shard_effective_token_indices = torch.arange(
        start_indices[model.mapping.attn_dp.rank], end_indices[model.mapping.attn_dp.rank],
        dtype=torch.int64
    )

    max_token_size_per_dp_group = token_size_per_dp_group.max().item()
    skip_padding_token_indices = torch.arange(
        model.mapping.attn_dp.group_size * max_token_size_per_dp_group, dtype=torch.int64
    ).view(model.mapping.attn_dp.group_size, max_token_size_per_dp_group)
    token_offset_per_dp_group = torch.arange(
        0, model.mapping.attn_dp.group_size * max_token_size_per_dp_group, step=max_token_size_per_dp_group,
        dtype=torch.int64).unsqueeze(1)
    token_index_with_padding = skip_padding_token_indices - token_offset_per_dp_group
    padding_mask = token_index_with_padding >= token_size_per_dp_group.unsqueeze(1)

    # 用于tp前数据汇总：包含padding token的token index（每个dp组的请求统一padding到所有请求的最大的输入长度，padding token index使用0表示）
    token_index_with_padding = token_index_with_padding[model.mapping.attn_dp.rank]
    token_index_with_padding = torch.where(padding_mask[model.mapping.attn_dp.rank], 0, token_index_with_padding)

    # 用于跳过padding token的token index
    skip_padding_token_indices = skip_padding_token_indices[~padding_mask]

    moe_skip_padding_token_indices = torch.arange(
        token_size_per_dp_group[model.mapping.attn_dp.rank], dtype=torch.int32)

    return {
        "token_size_per_dp_group": token_size_per_dp_group,
        "sum_token_size_per_dp_group": token_size_per_dp_group.sum().tolist(),
        "shard_effective_token_indices": shard_effective_token_indices.npu(),
        "token_index_with_padding": token_index_with_padding.npu(),
        "skip_padding_token_indices": skip_padding_token_indices.npu(),
        "moe_skip_padding_token_indices": moe_skip_padding_token_indices.npu(),
    }


def save_logits_if_needed(model, base_filename, logits_tensor: torch.Tensor | List[torch.Tensor]):
    ENV.update()
    if ENV.logits_save_enable:
        import os
        logits_save_filename = f"logits_{base_filename}.pth"
        logits_save_filepath = os.path.join(ENV.logits_save_folder, logits_save_filename)
        logits_save_filepath = file_utils.standardize_path(logits_save_filepath)
        file_utils.check_file_safety(logits_save_filepath, 'w', is_check_file_size=False)

        if not is_pp_activated(model) and model.rank == 0: 
            torch.save(logits_tensor.cpu(), logits_save_filepath)
            os.chmod(logits_save_filepath, 0o640)

        if is_pp_activated(model) and model.rank == model.world_size - 1:
            if (len(logits_tensor) == 0):
                raise AssertionError("save logits failed: no logits is found")
            total_logits = torch.concat(logits_tensor, dim=0)
            torch.save(total_logits.cpu(), logits_save_filepath)
            os.chmod(logits_save_filepath, 0o640)


def generate_token_from_microbatch(model, cache_manager, batches: List[Batch]):
    if len(batches) == 0:
        return 0, 0
    finish_batch_num = 0
    next_token_list = []
    logits_list = []
    forward_time_used = 0
    for batch in batches:
        input_ids = batch.batch_input_ids.npu()
        position_ids = batch.batch_position_ids.npu()
        is_prefill = batch.cu_seqlen_prefill is not None
        block_tables = batch.batch_block_tables.npu()
        kv_cache = cache_manager.kv_cache
        slots = batch.batch_slots_tables[batch.batch_slot_indices].npu()
        input_lengths = batch.context_length.npu()
        lm_head_indices = None if batch.lm_head_indices is None else batch.lm_head_indices.npu()
 
        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            is_prefill=is_prefill,
            block_tables=block_tables,
            kv_cache=kv_cache,
            slots=slots,
            input_lengths=input_lengths,
            max_seq_len=batch.max_s,
            lm_head_indices=lm_head_indices
        )

        if model.mapping.is_last_pp_rank():
            if is_prefill and logits.size(0) != batch.batch_num:
                if logits.size(0) != batch.lm_head_indices[-1] + 1:
                    logger.error(
                        f"prefill logits is invalid, batch num: {batch.batch_num}, " \
                        f"total token: {int(batch.lm_head_indices[-1] + 1)}, but logits shape is: {logits.shape}")
                    raise AssertionError
                logits = logits[batch.lm_head_indices]
            next_token = next_token_chooser(logits)
            ENV.update()
            if ENV.logits_save_enable:
                logits_list.append(logits.cpu())
            next_token_list.append(next_token.cpu())

    # broadcast next token to pp group
    if model.mapping.is_last_pp_rank():
        total_next_token = torch.concat(next_token_list, dim=0).int()
    else:
        total_batch_size = sum([batch.batch_num for batch in batches])
        total_next_token = torch.zeros([total_batch_size], dtype=torch.int32)
    torch.distributed.broadcast(tensor=total_next_token, 
                                src=model.mapping.pp.rank_per_group[-1][-1],
                                group=model.mapping.pp.pp_bcast_group,
                                async_op=False)
    
    if ENV.modeltest_dataset_specified:
        save_logits_if_needed(model, str(len(batches[0].req_list[0].out_token_list)), logits_list)
 
    total_token_list = total_next_token.tolist()
    start = 0
    end = 0
    for batch in batches:
        end += batch.batch_num
        for i, req in enumerate(batch.req_list):
            req.out_token_list.append(total_token_list[start + i])
        batch.batch_input_ids = total_next_token[start:end].to(torch.int64).npu()
        batch.batch_position_ids = batch.context_length.clone().to(torch.long)
        if batch.cu_seqlen_prefill is not None:
            batch.batch_slot_indices = batch.batch_slot_indices[batch.lm_head_indices]
            batch.cu_seqlen_prefill = None
            batch.lm_head_indices = None
 
        batch.batch_slot_indices += 1
        batch.context_length += 1
        batch.max_s += 1
        finish_batch_num += batch.filter(model.postprocessor, cache_manager)
        start = end
    return finish_batch_num, forward_time_used

previous_prefill_type = None
additional_kwargs = None
previous_batch_num = None


def verify(model, next_token_ids, draft_ids, num_speculative_tokens, rank):
    next_token_ids = next_token_ids.view(-1, num_speculative_tokens)[:, :num_speculative_tokens - 1]
    next_token_ids[next_token_ids == model.postprocessor.eos_token_id] = -1
    draft_ids = draft_ids.view(-1, num_speculative_tokens)[:, 1:]
    matches = torch.eq(next_token_ids, draft_ids).int()
    accepted_len = torch.cumprod(matches, dim=1).sum(dim=1)
    return accepted_len

accept_num = 0
token_num = 0


def generate_token(model, cache_manager, batch: Batch, eplb_forwarder):
    global accept_num, token_num
    global previous_prefill_type, additional_kwargs, previous_batch_num
    global decode_lm_head_indices_with_dp, lm_head_indices_with_dp
    perf_time_start = 0

    is_input_empty = False
    if previous_batch_num is None or previous_batch_num != batch.batch_num:
        previous_prefill_type = None
        additional_kwargs = None
        previous_batch_num = batch.batch_num
        decode_lm_head_indices_with_dp = None
        lm_head_indices_with_dp = None
        reset = True
    else:
        reset = False
    mtp_time = 0
    input_ids = batch.batch_input_ids
    position_ids = batch.batch_position_ids
    is_prefill = batch.cu_seqlen_prefill is not None
    if MTP > 0:
        cache_ids = cache_manager.hidden_state_cache.save_input_cache(batch.req_ids, is_prefill)
    
    kv_cache = cache_manager.kv_cache
    if ENV.omni_attention_enable:
        # This variable will allocate by cache_manager.allocate()
        forward_kwargs = batch.get_omni_input_tensor(cache_manager)
        block_tables = forward_kwargs.get("batch_block_tables")
        slots = forward_kwargs.get("batch_slots_tables")
    else:
        block_tables = batch.batch_block_tables
        slots = batch.batch_slots_tables[batch.batch_slot_indices]
    dp_rank_ids = batch.batch_dp_rank_ids
    input_lengths = batch.context_length
    input_lengths_sp = batch.context_length_sp
    max_seq_len = batch.max_s
    lm_head_indices = None if batch.lm_head_indices is None else batch.lm_head_indices

    positional_args = (
        input_ids, position_ids, is_prefill, cache_manager.num_blocks, cache_manager.block_size,
        block_tables, slots, input_lengths, input_lengths_sp
    )
    if (not is_prefill) and (MTP != 0):
        mtp_n_step = MTP + 1 
        position_ids = (position_ids.view(-1, 1).repeat(1, mtp_n_step) + torch.arange(mtp_n_step).view(1, -1)).view(-1)
        batch_slot_indices = batch.batch_slot_indices
        batch_slot_indices = (
            batch_slot_indices.view(-1, 1).repeat(1, mtp_n_step) + torch.arange(mtp_n_step).view(1, -1)).view(-1)
        slots = batch.batch_slots_tables[batch_slot_indices]
    kwargs = {
        "adapter_ids": batch.batch_adapter_ids,
        "max_out_len": cache_manager.output_max_length,
    }
    kwargs['lm_head_indices_dp_rank_ids'] = dp_rank_ids
    if ENV.omni_attention_enable:
        kwargs.update(forward_kwargs)

    dep_inputs = batch.dep_inputs
    kwargs.update(dict(dep_inputs=dep_inputs))

    if MTP and not is_prefill:
        input_ids_mtp = batch.mtp.batch_input_ids_mtp
        position_ids_mtp = position_ids
        block_tables_mtp = block_tables
        slots_mtp = slots - 1
        input_lengths_mtp = input_lengths * (MTP + 1) if -1 in batch.req_ids else input_lengths
        positional_args = (
            input_ids_mtp, position_ids_mtp, is_prefill, cache_manager.num_blocks, cache_manager.block_size,
            block_tables_mtp, slots_mtp, input_lengths_mtp, input_lengths_sp
        )
        lm_head_indices_mtp = None
        final_hidden_state_mtp = cache_manager.hidden_state_cache.get_hidden_states(cache_ids)
        q_lens = [MTP + 1] * block_tables_mtp.size(0)
        if model.mapping.has_dp() and not ENV.enable_dp_partition_up:
            if dp_rank_ids is None:
                raise ValueError("dp_rank_ids is not given when data parallel size > 1.")
            dp_rank_ids_per_token = dp_rank_ids.repeat_interleave(MTP + 1, dim=0)
            dp_rank_ids_per_token_sp = dp_rank_ids
            res = partition_data(model, dp_rank_ids, dp_rank_ids_per_token, dp_rank_ids_per_token_sp, *positional_args)
            input_ids_mtp, position_ids_mtp, is_prefill, block_tables_mtp, slots_mtp, input_lengths_mtp, _, \
                    max_seq_len_mtp, lm_head_indices_mtp, q_lens, is_input_empty = res

            additional_kwargs = gather_dp_data(model, dp_rank_ids_per_token)
            kwargs.update(additional_kwargs)

            dp_rank_ids_mask = dp_rank_ids_per_token == model.mapping.attn_dp.rank
            final_hidden_state_mtp_from_cache = cache_manager.hidden_state_cache.get_hidden_states(cache_ids)
            final_hidden_state_mtp = final_hidden_state_mtp_from_cache[dp_rank_ids_mask]
            if final_hidden_state_mtp.size(0) == 0:
                final_hidden_state_mtp = final_hidden_state_mtp_from_cache[:1]
        
        if ENV.enable_dp_partition_up:
            kwargs.update(dict(token_size_per_dp_group=batch.token_size_per_dp_group,
                               shard_effective_token_indices=batch.shard_effective_token_indices))

        # 使用mtp中的变量
        if lm_head_indices_mtp is not None:
            lm_head_indices_mtp = lm_head_indices_mtp + batch.mtp.accepted_lens - MTP
        else:
            lm_head_indices_mtp = torch.arange(0, batch.mtp.batch_input_ids_mtp.size(0), MTP + 1) + \
                                    batch.mtp.accepted_lens

        kwargs["hidden_states"] = final_hidden_state_mtp
        kwargs['q_lens'] = q_lens

        pre_mask_factor = -10000.0
        if model.dtype == torch.bfloat16:
            pre_mask_factor = 1.0
        mask = np.zeros(shape=(sum(q_lens), input_lengths_mtp.max())).astype(np.float16)
        pre_qseqlen = 0
        for i, _ in enumerate(q_lens):
            qseqlen = q_lens[i]
            kseqlen = input_lengths_mtp[i]
            tri = np.ones((qseqlen, qseqlen))
            tri = np.triu(tri, 1)
            tri *= pre_mask_factor
            mask[pre_qseqlen:(pre_qseqlen + qseqlen), kseqlen - qseqlen:kseqlen] = tri
            pre_qseqlen += qseqlen
        kwargs['spec_mask'] = torch.from_numpy(mask).to(model.dtype).npu()

        position_ids_mtp = position_ids_mtp.npu()
        input_lengths_mtp = input_lengths_mtp.npu()
        slots_mtp = slots_mtp.npu()
        block_tables_mtp = block_tables_mtp.npu()
        input_ids_mtp = input_ids_mtp.npu()

        input_lengths_lm = input_lengths + 1
        if is_input_empty and not ENV.enable_dp_partition_up:
            input_lengths = input_lengths_mtp
            slots = slots_mtp
        else:
            input_lengths = input_lengths_mtp + 1
            slots = slots_mtp + 1
        position_ids = position_ids_mtp
        block_tables = block_tables_mtp

        if model.mapping.has_dp() and not ENV.enable_dp_partition_up:
            lm_head_indices = partition_data_lm_head(model, dp_rank_ids, input_lengths_lm)
            lm_head_indices = lm_head_indices - MTP
            lm_head_indices = lm_head_indices.repeat_interleave(MTP + 1, dim=0)
            lm_head_indices = lm_head_indices + (torch.arange(lm_head_indices.size(0)) % (MTP + 1))
            lm_head_indices = lm_head_indices.npu()
        
        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids_mtp.shape[0]),
                                           dtype=torch.int64, device=input_ids_mtp.device)

        if ENV.benchmark_enable:
            import time
            torch.npu.synchronize()
            perf_time_start = time.time() 

        logits = model.forward(
            input_ids=input_ids_mtp,
            position_ids=position_ids_mtp,
            is_prefill=False,
            block_tables=block_tables_mtp,
            kv_cache=kv_cache,
            slots=slots_mtp,
            input_lengths=input_lengths_mtp,
            max_seq_len=batch.mtp.max_seq_len,
            lm_head_indices=lm_head_indices_mtp.npu(),
            input_lengths_sp=input_lengths_sp.to(input_ids.device),
            **kwargs
        )

        logits = logits
        input_ids2 = next_token_chooser(logits)
        input_ids = torch.concat([batch.batch_input_ids, input_ids2.view(-1, MTP)], dim=1).view(-1)
        input_ids_all = input_ids.clone()
        
        if not ENV.enable_dp_partition_up:
            if is_input_empty:
                input_ids = input_ids_mtp
            elif model.mapping.has_dp():
                input_ids = input_ids[dp_rank_ids_mask]

        kwargs['hidden_states'] = None

        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids_mtp,
            is_prefill=False,
            block_tables=block_tables_mtp,
            kv_cache=kv_cache,
            slots=slots,
            input_lengths=input_lengths,
            max_seq_len=max_seq_len,
            lm_head_indices=lm_head_indices,
            reset=reset,
            input_lengths_sp=input_lengths_sp.to(input_ids.device),
            **kwargs
        )

    else:
        positional_args = (
            input_ids, position_ids, is_prefill, cache_manager.num_blocks, cache_manager.block_size,
            block_tables, slots, input_lengths, input_lengths_sp
        )
        q_lens = [MTP + 1] * block_tables.size(0)
        if model.mapping.has_dp() and not ENV.enable_dp_partition_up:
            if dp_rank_ids is None:
                raise ValueError("dp_rank_ids is not given when data parallel size > 1.")
            if is_prefill:
                dp_rank_ids_per_token = torch.repeat_interleave(dp_rank_ids, input_lengths)
                dp_rank_ids_per_token_sp = torch.repeat_interleave(dp_rank_ids, input_lengths_sp) \
                    if model.mapping.has_attn_inner_sp() else None
            else:
                dp_rank_ids_per_token = dp_rank_ids
                dp_rank_ids_per_token_sp = dp_rank_ids
                if MTP:
                    dp_rank_ids_per_token = dp_rank_ids.repeat_interleave(MTP + 1, dim=0)
            res = partition_data(model, dp_rank_ids, dp_rank_ids_per_token, dp_rank_ids_per_token_sp, *positional_args)
            input_ids, position_ids, is_prefill, block_tables, slots, input_lengths, input_lengths_sp, \
                    max_seq_len, lm_head_indices, q_lens, _ = res

            if model.mapping.has_attn_cp() and is_prefill:
                # The input of gather_dp_data is the full amount of tokens.
                # When CP is enabled, it is necessary to multiply by the cp_size.
                dp_rank_ids_per_token = torch.repeat_interleave(dp_rank_ids,
                                                                input_lengths * model.mapping.attn_cp.group_size)
            additional_kwargs = gather_dp_data(model, dp_rank_ids_per_token)
            previous_prefill_type = is_prefill
            kwargs.update(additional_kwargs)
            if MTP and not is_prefill:
                lm_head_indices = lm_head_indices - MTP
                lm_head_indices = lm_head_indices.repeat_interleave(MTP + 1, dim=0)
                lm_head_indices = lm_head_indices + (torch.arange(lm_head_indices.size(0)) % (MTP + 1))

        if ENV.enable_dp_partition_up:
            kwargs.update(dict(token_size_per_dp_group=batch.token_size_per_dp_group,
                               shard_effective_token_indices=batch.shard_effective_token_indices))

        if batch.batch_cross_attn_mask is not None:
            if is_prefill:
                cross_slots_mapping = batch.batch_slots_tables[batch.batch_cross_attn_slot_indices].npu()
                multi_modal_inputs = batch.batch_multi_modal
            else:
                cross_slots_mapping = None
                multi_modal_inputs = None
            cross_attention_mask = batch.batch_cross_attn_mask.npu()
            cross_context_lens = batch.batch_cross_attn_context_length.npu()
            full_text_row_masked_out_mask = batch.batch_cross_attn_full_row_mask.npu()
            kwargs.update(dict(multi_modal_inputs=multi_modal_inputs,
                            cross_slots_mapping=cross_slots_mapping,
                            cross_attention_mask=cross_attention_mask,
                            cross_context_lens=cross_context_lens,
                            full_text_row_masked_out_mask=full_text_row_masked_out_mask,))

        if not is_prefill and MTP:
            kwargs['q_lens'] = q_lens

            pre_mask_factor = -10000.0
            if model.dtype == torch.bfloat16:
                pre_mask_factor = 1.0
            mask = np.zeros(shape=(sum(q_lens), input_lengths.max())).astype(np.float16)
            pre_qseqlen = 0
            for i, _ in enumerate(q_lens):
                qseqlen = q_lens[i]
                kseqlen = input_lengths[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= pre_mask_factor
                mask[pre_qseqlen:(pre_qseqlen + qseqlen), kseqlen - qseqlen:kseqlen] = tri
                pre_qseqlen += qseqlen
            kwargs['spec_mask'] = torch.from_numpy(mask).to(model.dtype).npu()

        if ENV.benchmark_enable:
            import time
            torch.npu.synchronize()
            perf_time_start = time.time()

        if model.mapping.has_attn_cp() or model.mapping.has_attn_inner_sp():
            cur_dp_rank_id_mask = dp_rank_ids == model.mapping.attn_dp.rank
            cur_dp_rank_req = [req for req, mask in zip(batch.req_list, cur_dp_rank_id_mask) if mask == 1]
            cp_rank = model.mapping.attn_cp.rank
            cp_size = model.mapping.attn_cp.group_size
            sp_rank = model.mapping.attn_inner_sp.rank
            if is_prefill and len(cur_dp_rank_req) != 0:
                slots_tmp = []
                req_offset = 0
                for req in cur_dp_rank_req:
                    req_cur_rank_slot_len = 0  # the length of kv that will be cached on the current rank
                    prefix_cp, prefix_sp, suffix_sp, suffix_cp = [], [], [], []
                    if model.mapping.has_attn_cp():
                        req_cur_rank_slot_len = req.input_length
                        prefix_cp = [-1] * cp_rank * req.input_length
                        suffix_cp = [-1] * (cp_size - 1 - cp_rank) * req.input_length
                    if model.mapping.has_attn_inner_sp():
                        req_cur_rank_slot_len = req.input_len_per_sp[sp_rank]
                        prefix_sp = [-1] * sum(req.input_len_per_sp[:sp_rank])
                        suffix_sp = [-1] * sum(req.input_len_per_sp[sp_rank + 1:])
                    # During CP or SP, the all KVs of the request will be gathered before cache
                    slots_tmp.extend([torch.tensor(prefix_cp),
                                      torch.tensor(prefix_sp),
                                      slots[req_offset: (req_offset + req_cur_rank_slot_len)],
                                      torch.tensor(suffix_sp),
                                      torch.tensor(suffix_cp)])
                    req_offset += req_cur_rank_slot_len
                slots = torch.concat(slots_tmp, dim=0).to(torch.long)
            elif not is_prefill:  # decode
                if model.mapping.has_attn_cp():
                    if cp_rank != batch.cp_cache_rank:
                        slots = slots.fill_(-1)
                if model.mapping.has_attn_inner_sp():
                    if sp_rank != batch.sp_cache_rank:
                        slots = slots.fill_(-1)
                    if len(cur_dp_rank_req) == 0:  # DP dummy run
                        input_lengths_sp = torch.tensor([1], dtype=torch.int32)
                    kwargs.update({
                        "input_lengths_sp": input_lengths_sp.npu()  # decode pa input
                    })

        logits = model.forward(
            input_ids=input_ids.npu(),
            position_ids=position_ids.npu(),
            is_prefill=is_prefill,
            block_tables=block_tables.npu(),
            kv_cache=kv_cache,
            slots=slots.npu(),
            input_lengths=input_lengths.npu(),
            max_seq_len=max_seq_len,
            lm_head_indices=lm_head_indices.npu() if lm_head_indices is not None else None,
            reset=reset,
            **kwargs
        )

    if isinstance(logits, tuple):
        if MTP:
            logits, hidden_states_output = logits        
            batch.mtp.max_seq_len = max_seq_len
            hidden_states_tmp = hidden_states_output.clone()
            if is_prefill:
                hidden_states_tmp = hidden_states_tmp.repeat_interleave(MTP + 1, dim=0)
            cache_manager.hidden_state_cache.save_hidden_states(cache_ids, hidden_states_tmp)
        else:
            logits, perf_time_start = logits

    if batch.cu_seqlen_prefill is not None and logits.size(0) != batch.batch_num and (not MTP):
        if logits.size(0) != batch.lm_head_indices[-1] + 1:
            logger.error(f"prefill logits is invalid, batch num: {batch.batch_num}," +
                         f" total token: {int(batch.lm_head_indices[-1] + 1)}, but logits shape is: {logits.shape}")
            raise AssertionError
        logits = logits[batch.lm_head_indices]

    if ENV.modeltest_dataset_specified:
        save_logits_if_needed(model, str(len(batch.req_list[0].out_token_list)), logits)

    if logits.size(1) > 1:
        next_token = next_token_chooser(logits)
    elif logits.dim() == 2 and logits.size(0) == 1:
        next_token = logits.squeeze().unsqueeze(0)
    else:
        next_token = logits.squeeze()
    eplb_level = getattr(model.config, "eplb_level", 0)
    if model.model.warmup_is_end and (ENV.enable_expert_hotpot_gather or eplb_level == EPLBType.DYNAMIC_EPLB):
        EplbExpertDataCollect().accumulation_expert_cumsum(is_prefill=is_prefill)
        if is_prefill:
            save_eplb_data(
                        model.rank,
                        EplbExpertDataCollect().get_prefill_token_num_per_expert(),
                        "prefill", EplbExpertDataCollect().prefill_forward_count
                        )
        else:
            save_eplb_data(
                        model.rank,
                        EplbExpertDataCollect().get_decode_token_num_per_expert(),
                        "decode", EplbExpertDataCollect().decode_forward_count
                        )
    topk_output = getattr(model.model, "topk_output", False)
    if model.model.warmup_is_end and topk_output:
        if is_prefill:
            save_eplb_data(
                        model.rank,
                        EplbExpertDataCollect().get_topk(),
                        "prefill", EplbExpertDataCollect().prefill_forward_count, True
                        )
        else:
            save_eplb_data(
                        model.rank,
                        EplbExpertDataCollect().get_topk(),
                        "decode", EplbExpertDataCollect().decode_forward_count, True
                        )
    if eplb_level == EPLBType.DYNAMIC_EPLB and eplb_forwarder is not None:
        eplb_forwarder.do_aggregate()
    if MTP and is_prefill:
        next_token = next_token.repeat_interleave(MTP + 1)

    accepted_lens = torch.zeros(batch.batch_block_tables.size(0), dtype=next_token.dtype, device=next_token.device)
    if MTP:
        next_token_list = next_token.cpu()
        mtp_step_n = MTP + 1
        if not is_prefill:
            accepted_lens = verify(model, next_token_list, input_ids_all.cpu(), mtp_step_n, model.rank)
            accepted_lens = accepted_lens.to(next_token.dtype).to(next_token.device)
    req_id = torch.tensor((batch.req_ids), dtype=next_token.dtype, device=next_token.device)
    res = (next_token, accepted_lens, perf_time_start, mtp_time, req_id)
    return res


def prepare_for_next_generate(model, cache_manager, batch, res):
    global accept_num, token_num
    global previous_prefill_type, additional_kwargs, previous_batch_num
    global decode_lm_head_indices_with_dp, lm_head_indices_with_dp
    (next_token, accepted_lens, perf_time_start, mtp_time) = res
    next_token_list = next_token.cpu()
    is_prefill = batch.cu_seqlen_prefill is not None
    perf_time = 0

    if MTP:
        accepted_lens = accepted_lens.cpu()
        batch.mtp.batch_input_ids_mtp = next_token_list
        batch.mtp.accepted_lens = accepted_lens
        accepted_lens_numpy = accepted_lens.numpy()
        next_token_list_numpy = next_token_list.numpy()
        for i, req in enumerate(batch.req_list):
            for j in range(accepted_lens_numpy[i] + 1):
                req.out_token_list.append(next_token_list_numpy[j + (MTP + 1) * i])
    else:
        accepted_lens_numpy = next_token_list.numpy()
        for i, req in enumerate(batch.req_list):
            req.out_token_list.append(accepted_lens_numpy[i])

    batch.batch_input_ids = next_token_list.to(torch.int64)

    if MTP:
        abc = torch.gather(batch.batch_input_ids.view(-1, MTP + 1).cpu(), 1, accepted_lens.long().unsqueeze(1))
        batch.batch_input_ids = abc.to(torch.int64).npu()

    if batch.multi_context_length is not None:
        batch.batch_position_ids = batch.multi_context_length.clone().to(torch.long)
        batch.multi_context_length += 1
    else:
        batch.batch_position_ids = batch.context_length.clone().to(torch.long)
    if batch.cu_seqlen_prefill is not None:
        if batch.batch_cross_attn_mask is not None: # for llama3.2
            batch.batch_cross_attn_mask = batch.batch_cross_attn_mask[batch.lm_head_indices]
            batch.batch_cross_attn_context_length = batch.batch_cross_attn_context_length[batch.lm_head_indices]
            batch.batch_cross_attn_full_row_mask = batch.batch_cross_attn_full_row_mask[batch.lm_head_indices]
        if model.mapping.has_attn_inner_sp():
            batch.batch_slot_indices = batch.batch_slot_indices[batch.lm_head_indices_sp]
        elif model.mapping.has_attn_cp():
            batch.batch_slot_indices = batch.batch_slot_indices[torch.cumsum(batch.context_length, dim=0) - 1]
        else:
            batch.batch_slot_indices = batch.batch_slot_indices[batch.lm_head_indices]
        batch.cu_seqlen_prefill = None
        batch.lm_head_indices = None
    
    if model.mapping.has_attn_cp() and model.mapping.has_attn_inner_sp():
        global_cache_rank = batch.step % (model.mapping.attn_cp.group_size * model.mapping.attn_inner_sp.group_size)
        batch.cp_cache_rank = global_cache_rank // model.mapping.attn_inner_sp.group_size
        batch.sp_cache_rank = global_cache_rank % model.mapping.attn_inner_sp.group_size
        batch.batch_slot_indices[torch.nonzero((batch.batch_cp_rank_ids == batch.cp_cache_rank) &
                                               (batch.batch_sp_rank_ids == batch.sp_cache_rank)).squeeze()] += 1
        batch.context_length_sp[torch.nonzero(batch.batch_sp_rank_ids == batch.sp_cache_rank).squeeze()] += 1
        batch.context_length[torch.nonzero(batch.batch_cp_rank_ids == batch.cp_cache_rank).squeeze()] += 1
        batch.batch_position_ids = batch.context_length_cp.clone().to(torch.long)
        batch.context_length_cp += 1
        batch.step += 1
    elif model.mapping.has_attn_inner_sp():
        # sp, decode, kvcache split to sp rank 
        batch.sp_cache_rank = batch.step % model.mapping.attn_inner_sp.group_size
        sp_cache_rank_indices = torch.nonzero(batch.batch_sp_rank_ids == batch.sp_cache_rank).squeeze()
        batch.batch_slot_indices[sp_cache_rank_indices] += 1
        batch.context_length_sp[sp_cache_rank_indices] += 1
        batch.context_length += 1
        batch.step += 1
    elif model.mapping.has_attn_cp():
        # decode, kvcache split to cp rank
        batch.cp_cache_rank = batch.step % model.mapping.attn_cp.group_size
        cp_cache_rank_indices = torch.nonzero(batch.batch_cp_rank_ids == batch.cp_cache_rank).squeeze()
        batch.batch_slot_indices[cp_cache_rank_indices] += 1
        batch.context_length[cp_cache_rank_indices] += 1
        batch.batch_position_ids = batch.context_length_cp.clone().to(torch.long)
        batch.context_length_cp += 1
        batch.step += 1
    else:
        batch.batch_slot_indices += 1
        batch.context_length += 1

    batch.max_s += 1

    if MTP:
        batch.batch_slot_indices += accepted_lens
        batch.context_length += accepted_lens
        batch.batch_position_ids = batch.context_length.clone().to(torch.long) - 1
        batch.max_s += accepted_lens.max()

        if not is_prefill:
            accept_num += accepted_lens.sum()
            token_num += accepted_lens.numel() * MTP
            print_log(model.rank, logger.info, 
                      f"accept_num {accept_num} token_num {token_num} ratio {accept_num/token_num}")
    if ENV.benchmark_enable:
        import time
        torch.npu.synchronize()
        perf_time_end = time.time()
        perf_time = perf_time_end - perf_time_start
    filter_res = batch.filter(model.postprocessor, cache_manager)
    return filter_res, perf_time


def find_first_indices(item_a, item_b):
    for i, _ in enumerate(item_a):
        item_a[i] = item_a[i].cpu().numpy() if isinstance(item_a[i], torch.Tensor) else item_a[i]
    
    # 创建一个字典来存储每个元素在 item_a 中的首次出现位置
    unique_elements, first_indices = np.unique(item_a, return_index=True)
    index_dict = dict(zip(unique_elements, first_indices))
    
    # 根据字典查找 item_b 中每个元素在 item_a 中的首次出现位置
    indices = np.array([index_dict.get(value, -1) for value in item_b])

    return indices


def dp_all_gather(model, in_tensor, out_tensor):
    inputs = {model.engine_input_key: in_tensor}
    outputs = {model.engine_output_key: out_tensor}
    model.dp_all_gather_engine.forward(inputs, outputs)
    gathered_tensor = outputs.get(model.engine_output_key)
    return gathered_tensor


def gather_mtp_tokens_v2(model, input_batch: Batch, accept_lens, next_tokens):
    dp_rank_ids = input_batch.batch_dp_rank_ids
    bs_per_dp_group = np.bincount(np.array(dp_rank_ids), minlength=model.mapping.attn_dp.group_size)
    is_dummy_dp = bs_per_dp_group == 0
    bs_per_dp_group = np.where(bs_per_dp_group == 0, 1, bs_per_dp_group)

    # 用于数据汇总：包含padding token的token index（每个dp组的请求统一padding到所有请求的最大的输入长度，padding token index使用0表示）
    max_bs_per_dp_group = bs_per_dp_group.max().item()
    skip_padding_token_indices = np.arange(
        model.mapping.attn_dp.group_size * max_bs_per_dp_group, dtype=np.int64
    ).reshape((model.mapping.attn_dp.group_size, max_bs_per_dp_group))
    token_offset_per_dp_group = np.arange(
        0, (model.mapping.attn_dp.group_size - 1) * max_bs_per_dp_group + 1,
        step=max_bs_per_dp_group,
        dtype=np.int64)
    token_index_with_padding = skip_padding_token_indices - token_offset_per_dp_group.reshape((-1, 1))
    padding_mask = token_index_with_padding >= bs_per_dp_group.reshape((-1, 1))
    token_index_with_padding = token_index_with_padding[model.mapping.attn_dp.rank]
    token_index_with_padding = np.where(padding_mask[model.mapping.attn_dp.rank], 0, token_index_with_padding)

    # 用于跳过padding token的token index并去除陪跑数据
    padding_mask[is_dummy_dp] = True
    skip_padding_token_indices = skip_padding_token_indices[~padding_mask]
    sorted_indices = np.argsort(dp_rank_ids, kind="stable")
    reverse_indices = np.argsort(sorted_indices, kind="stable")
    skip_padding_token_indices = skip_padding_token_indices[reverse_indices.tolist()]

    reordered_accept_lens = torch.empty(len(input_batch.req_ids), dtype=torch.int64, device=model.device)
    reordered_next_tokens = torch.empty(len(input_batch.req_ids), (MTP + 1), dtype=torch.int64, device=model.device)

    accept_lens_next_tokens = torch.cat([accept_lens.reshape([-1, 1]), next_tokens.reshape(-1, MTP + 1)], dim=1)

    output = {"reordered_accept_lens": reordered_accept_lens, "reordered_next_tokens": reordered_next_tokens}

    model.dp_all_gather_engine.forward(
        {"accept_lens_next_tokens": accept_lens_next_tokens,
         "padding": torch.tensor(token_index_with_padding).to(model.device),
         "unpadding": torch.tensor(skip_padding_token_indices).to(model.device)
        },
        output,
    )

    return reordered_accept_lens, reordered_next_tokens.cpu().flatten()


def generate_token_with_clocking(model, cache_manager, input_batch: Batch | List[Batch], eplb_forwarder):
    dep_inputs = None
    if model.mapping.has_dp() and ENV.enable_dp_move_up:
        input_batch_dp, dp_rank_ids_per_token = split_input_batch_by_dp(model, cache_manager, input_batch)
        dep_inputs, token_size_per_dp_group, shard_effective_token_indices, max_dp_batch_size = \
            get_dep_inputs(model, input_batch, input_batch_dp, dp_rank_ids_per_token)
    if ENV.enable_dp_partition_up:
        input_batch_dp.dep_inputs = dep_inputs
        input_batch_dp.token_size_per_dp_group = token_size_per_dp_group
        input_batch_dp.shard_effective_token_indices = shard_effective_token_indices
        input_batch_dp.max_dp_batch_size = max_dp_batch_size
        input_batch_in = input_batch_dp
    else:
        input_batch.dep_inputs = dep_inputs
        input_batch_in = input_batch
    
    perf_time = 0
    if is_pp_activated(model):
        req_finished, perf_time = generate_token_from_microbatch(model, cache_manager, input_batch_in)
    else:
        res = generate_token(model, cache_manager, input_batch_in, eplb_forwarder)
        (next_token, accepted_lens, perf_time_start, mtp_time, req_id) = res
        if model.mapping.has_dp() and ENV.enable_dp_partition_up:            
            accepted_lens_total, next_token_total = gather_mtp_tokens_v2(model, input_batch, accepted_lens, next_token)
        else:
            next_token_total = next_token
            accepted_lens_total = torch.zeros(len(next_token_total), dtype=next_token.dtype) \
                                    if accepted_lens is None else accepted_lens
        res = (next_token_total, accepted_lens_total, perf_time_start, mtp_time)
        req_finished, perf_time = prepare_for_next_generate(model, cache_manager, input_batch, res)

    if model.model.warmup_is_end:
        model.clear_internal_tensors()

    return req_finished, perf_time


def get_dep_inputs(model, input_batch: Batch | List[Batch], input_batch_dp: Batch | List[Batch], dp_rank_ids_per_token):
    input_ids = input_batch_dp.batch_input_ids
    input_ids_len = len(input_ids)
    token_size_per_dp_group = np.bincount(dp_rank_ids_per_token, minlength=model.mapping.attn_dp.group_size)
    token_size_per_dp_group = np.where(token_size_per_dp_group == 0, 1, token_size_per_dp_group)
    max_dp_batch_size = max(token_size_per_dp_group)
    is_prefill = input_batch.cu_seqlen_prefill is not None

    if MTP > 0 and not is_prefill:
        token_size_per_dp_group = token_size_per_dp_group * (MTP + 1)
        input_ids_len = input_ids_len * (MTP + 1)
    expert_parallel_degree = model.config.ep_level

    group_size = model.mapping.attn_tp.group_size
    max_token_size_per_dp_group = np.max(token_size_per_dp_group)

    if max_token_size_per_dp_group % group_size != 0:
        padding_tmp = group_size - max_token_size_per_dp_group % group_size
        max_token_size_per_dp_group += padding_tmp
    token_size_per_dp_group_startid = np.cumsum(token_size_per_dp_group)
    token_size_per_dp_group_startid[-1] = 0

    lm_head_skip_padding_token_indices = np.array([0])
    if ENV.lm_head_local_tp:
        rank = model.mapping.attn_tp.rank + model.mapping.attn_dp.rank * model.mapping.attn_tp.group_size
        local_dp_count = model.mapping.local_world_size // model.mapping.attn_tp.group_size
        node_rank = rank // model.mapping.local_world_size
        local_token_size_per_dp_group = token_size_per_dp_group[
            node_rank * local_dp_count:(node_rank + 1) * local_dp_count
        ]
        lm_head_skip_padding_token_indices = np.concatenate([
            np.arange(j, dtype=np.int32) + max_token_size_per_dp_group * rank_id
            for rank_id, j in enumerate(local_token_size_per_dp_group)
        ])
    else:
        lm_head_skip_padding_token_indices = np.concatenate([
            np.arange(j, dtype=np.int32) + max_token_size_per_dp_group * rank_id
            for rank_id, j in enumerate(token_size_per_dp_group)
        ])

    in_post_lmhead_unpadding_indices = input_batch_dp.post_lmhead_gather_indices

    atom_dp_size = max_token_size_per_dp_group // model.mapping.attn_tp.group_size
    input_length_padding = max_token_size_per_dp_group - input_ids_len
    attn_padding_idx = np.concatenate([
            np.arange(input_ids_len, dtype=np.int32),
            np.zeros(input_length_padding, dtype=np.int32)
        ]).reshape(-1)

    if model.mapping.has_attn_o_proj_tp():
        gather_prenorm_idx = np.arange(input_ids_len, dtype=np.int32)
    else:
        gather_prenorm_idx = \
            attn_padding_idx[model.mapping.attn_tp.rank * atom_dp_size: \
            (model.mapping.attn_tp.rank + 1) * atom_dp_size]

    if expert_parallel_degree == ExpertParallelDegree.DYNAMIC_EP or \
    (expert_parallel_degree == ExpertParallelDegree.MIX_EP and is_prefill):
        attn_unpadding_idx = np.arange(atom_dp_size, dtype=np.int32)
        ffn_padding_idx = attn_unpadding_idx
    else:
        attn_unpadding_idx = np.concatenate(
            [np.arange(s) + max_token_size_per_dp_group * i
                for i, s in enumerate(token_size_per_dp_group)])
        ffn_padding_idx = np.concatenate([
            np.concatenate([
                np.arange(j, dtype=np.int32) + token_size_per_dp_group_startid[rank_id - 1],
                np.zeros(max_token_size_per_dp_group - j, dtype=np.int32)
            ]) for rank_id, j in enumerate(token_size_per_dp_group)
        ])
    ffn_unpadding_idx = np.arange(input_ids_len, dtype=np.int32)

    dynamic_ep_idx = np.zeros(1, dtype=np.int32)
    moe_idx = np.zeros(1, dtype=np.int32)

    if expert_parallel_degree == ExpertParallelDegree.DYNAMIC_EP or \
        (expert_parallel_degree == ExpertParallelDegree.MIX_EP and is_prefill):
        num_experts_per_tok = model.config.num_experts_per_tok
        mix_shared_routing = getattr(model.model, "mix_shared_routing", False)
        if mix_shared_routing:
            num_experts_per_tok += 1
        if model.mapping.attn_tp.group_size == 1:
            dynamic_ep_idx = np.arange(input_ids_len * num_experts_per_tok, dtype=np.int32)
            dynamic_ep_idx_padding = \
                np.arange(attn_unpadding_idx.shape[0] * num_experts_per_tok, dtype=np.int32)
        else:
            dynamic_ep_idx = np.arange(attn_unpadding_idx.shape[0] * num_experts_per_tok, dtype=np.int32)
            dynamic_ep_idx_padding = dynamic_ep_idx

        ep_input_length = \
            int(dynamic_ep_idx_padding.shape[0] * get_all2all_buffer_factor(model, dynamic_ep_idx_padding.shape[0]))
        all2all_padding = ep_input_length % model.mapping.moe_ep.group_size
        ep_input_length_padding = (model.mapping.moe_ep.group_size - all2all_padding) if all2all_padding != 0 else 0
        ep_input_length_padding += ep_input_length
        moe_idx = np.arange(1, ep_input_length_padding + 1, dtype=np.int32)

    dep_inputs = [attn_padding_idx, attn_unpadding_idx, ffn_padding_idx,
            ffn_unpadding_idx, lm_head_skip_padding_token_indices, gather_prenorm_idx,
            dynamic_ep_idx, moe_idx, in_post_lmhead_unpadding_indices]

    tensor_dep_inputs = [torch.tensor(arr).npu() for arr in dep_inputs]
    if ENV.enable_dp_move_up:
        token_size_per_dp_group_tensor = torch.tensor(token_size_per_dp_group).npu()
        start_indices = torch.cumsum(token_size_per_dp_group_tensor, dim=0) - token_size_per_dp_group_tensor
        end_indices = torch.cumsum(token_size_per_dp_group_tensor, dim=0)
        shard_effective_token_indices = torch.arange(
            start_indices[model.mapping.attn_dp.rank], end_indices[model.mapping.attn_dp.rank],
            dtype=torch.int64
        )
    else:
        token_size_per_dp_group_tensor = None
        shard_effective_token_indices = None

    return tensor_dep_inputs, token_size_per_dp_group_tensor, shard_effective_token_indices, max_dp_batch_size


def _interpolate_length_thresholds(length, length_thresholds, idx):
    weight = (length - length_thresholds[idx - 1][0]) / (length_thresholds[idx][0] - length_thresholds[idx - 1][0])
    max_scale = length_thresholds[idx - 1][1]
    min_scale = length_thresholds[idx][1]
    return max_scale * (1 - weight) + min_scale * weight


def get_all2all_buffer_factor(model, length):
    length = length * model.mapping.attn_dp.group_size
    if hasattr(model.config, "alltoall_ep_buffer_scale_factors"):
        length_thresholds = model.config.alltoall_ep_buffer_scale_factors
    else:
        length_thresholds = \
            [[1048576, 1.32], [524288, 1.4], [262144, 1.53],
                [131072, 1.8], [32768, 3.0], [8192, 5.2], [0, 8.0]]
    length_thresholds = length_thresholds[::-1]
    for idx in range(1, len(length_thresholds)):
        if length <= length_thresholds[idx][0]:
            return _interpolate_length_thresholds(length, length_thresholds, idx)
    return length_thresholds[-1][1]


def split_input_batch_by_dp(model, cache_manager, batch: Batch | List[Batch]):
    is_prefill = batch.cu_seqlen_prefill is not None
    if is_prefill:
        dp_rank_ids_per_token = torch.repeat_interleave(batch.batch_dp_rank_ids, batch.context_length)
    else:
        dp_rank_ids_per_token = batch.batch_dp_rank_ids
    # 获取当前dp组，batch级别
    cur_dp_rank_id_idx = torch.where(batch.batch_dp_rank_ids == model.mapping.attn_dp.rank)[0]
    cur_dp_rank_id_per_token_idx = torch.where(dp_rank_ids_per_token == model.mapping.attn_dp.rank)[0]

    # 获取全量的req_list
    req_list = batch.req_list

    # 提取当前dp组的req_list
    cur_dp_rank_id_idx_list = cur_dp_rank_id_idx.tolist()
    req_list_dp = [req_list[i] for i in cur_dp_rank_id_idx_list]

    if len(req_list_dp) > 0:
        # 存在当前dp组的数据
        batch_dp = Batch(req_list_dp)
        batch_dp.cu_seqlen_prefill = batch.cu_seqlen_prefill
        batch_dp.batch_input_ids = batch.batch_input_ids[cur_dp_rank_id_per_token_idx]
        batch_dp.batch_position_ids = batch.batch_position_ids[cur_dp_rank_id_per_token_idx]
        batch_dp.batch_block_tables = batch.batch_block_tables[cur_dp_rank_id_idx]
        batch_dp.batch_slots_tables = batch.batch_slots_tables
        batch_dp.batch_slot_indices = batch.batch_slot_indices[cur_dp_rank_id_per_token_idx]
        batch_dp.batch_dp_rank_ids = batch.batch_dp_rank_ids[cur_dp_rank_id_idx]
        batch_dp.context_length = batch.context_length[cur_dp_rank_id_idx]
        batch_dp.lm_head_indices = np.cumsum(batch_dp.context_length) - 1 if batch.lm_head_indices is not None else None
        if batch.mtp.accepted_lens is not None and ENV.enable_dp_partition_up:
            dp_rank_ids_per_token_for_mtp = torch.repeat_interleave(batch.batch_dp_rank_ids, MTP + 1)
            cur_dp_rank_ids_per_token_for_mtp_idx = \
                torch.where(dp_rank_ids_per_token_for_mtp == model.mapping.attn_dp.rank)[0]
            batch_dp.mtp.batch_input_ids_mtp = batch.mtp.batch_input_ids_mtp[cur_dp_rank_ids_per_token_for_mtp_idx]
            batch_dp.mtp.accepted_lens = batch.mtp.accepted_lens[cur_dp_rank_id_idx]
            batch_dp.mtp.max_seq_len = batch.mtp.max_seq_len
    else:
        # 构造陪跑req
        req_new = Request(1, cache_manager.block_size, -1,
                          torch.tensor([1], dtype=torch.int64), model.mapping.attn_dp.rank)
        batch_dp = Batch([req_new])
        batch_dp.cu_seqlen_prefill = batch.cu_seqlen_prefill
        if MTP and not is_prefill:
            batch_dp.batch_input_ids = torch.tensor([[1]], dtype=torch.int64).npu()
        else:
            batch_dp.batch_input_ids = torch.tensor([1], dtype=torch.int64).npu()
        batch_dp.batch_position_ids = torch.tensor([0], dtype=torch.int32)
        batch_dp.batch_block_tables = torch.tensor([[cache_manager.num_blocks - 1]], dtype=torch.int32)
        batch_dp.batch_slots_tables = cache_manager.total_slots[batch_dp.batch_block_tables].flatten()
        batch_dp.batch_slot_indices = torch.tensor([0], dtype=torch.int64)
        batch_dp.batch_dp_rank_ids = None
        batch_dp.context_length = torch.tensor([1], dtype=torch.int64)
        batch_dp.lm_head_indices = None
        if batch.mtp.accepted_lens is not None and ENV.enable_dp_partition_up:
            batch_dp.mtp.batch_input_ids_mtp = torch.tensor([1, 1], dtype=torch.int64)
            batch_dp.mtp.accepted_lens = torch.tensor([1], dtype=torch.int64)
            batch_dp.mtp.max_seq_len = batch.mtp.max_seq_len
    
    if ENV.lm_head_local_tp:
        # lmhead 独立tp8+dp并行
        rank = model.mapping.attn_tp.rank + model.mapping.attn_dp.rank * model.mapping.attn_tp.group_size
        local_dp_count = model.mapping.local_world_size // model.mapping.attn_tp.group_size
        node_rank = rank // model.mapping.local_world_size
        local_dp_rank_list = range(node_rank * local_dp_count, (node_rank + 1) * local_dp_count)
        local_index = local_dp_rank_list.index(model.mapping.attn_dp.rank)
        local_dp_rank_list = torch.tensor(local_dp_rank_list)
        ma_mask = local_dp_rank_list.view(-1, 1) == batch.batch_dp_rank_ids.view(1, -1)
        token_size_per_dp_group_ma = ma_mask.sum(1)
        if batch.lm_head_indices is not None:
            context_length_ma = ma_mask * batch.context_length
            context_length_ma[token_size_per_dp_group_ma == 0, 0] = 1
            context_length_ma = context_length_ma[context_length_ma != 0]
            batch_dp.lm_head_indices = torch.cumsum(context_length_ma, dim=0) - 1

        token_size_per_dp_group_ma[token_size_per_dp_group_ma == 0] = 1
        token_size_per_dp_group_ma = token_size_per_dp_group_ma.cumsum(dim=0)
        batch_dp.post_lmhead_gather_indices = torch.arange(0 if local_index == 0 else \
                                                           token_size_per_dp_group_ma[local_index - 1],
                                                           token_size_per_dp_group_ma[local_index])

    return batch_dp, dp_rank_ids_per_token



def generate_req(req_list, model, max_batch_size, max_prefill_tokens, cache_manager, eplb_forwarder=None):
    req_num = len(req_list)
    print_log(model.rank, logger.info, f"------total req num: {req_num}, infer start--------")

    req_idx = 0
    total_req_finished = 0

    generate_batch_size_per_dp_group = [0] * model.mapping.attn_dp.group_size
    total_generate_batch_size = 0
    generate_batches = []

    prefill_benchmark_timelist = []
    decoder_benchmark_timelist = []

    while total_req_finished < req_num:
        do_generate = True
        # 仍有新请求待处理，且decode阶段的请求未满
        if req_idx < req_num and min(generate_batch_size_per_dp_group) < max_batch_size:
            prefill_start = req_idx
            free_block_per_dp_group = [cache_manager.get_free_block_num(i)
                                       for i in range(model.mapping.attn_dp.group_size)]
            total_need_blocks_per_dp_group = [0] * model.mapping.attn_dp.group_size
            total_prefill_token_per_dp_group = [0] * model.mapping.attn_dp.group_size
            prefill_batch_size_per_dp_group = [0] * model.mapping.attn_dp.group_size
            total_prefill_batch_size = 0

            # 请求分配给最空闲的dp组
            dp_rank = np.argmax(np.array(free_block_per_dp_group) - np.array(total_need_blocks_per_dp_group))
            while generate_batch_size_per_dp_group[dp_rank] + prefill_batch_size_per_dp_group[dp_rank] < max_batch_size:
                if req_idx >= req_num:
                    break
                cur_need_blocks = req_list[req_idx].need_blocks
                cur_context_len = req_list[req_idx].input_length
                if total_need_blocks_per_dp_group[dp_rank] + cur_need_blocks > free_block_per_dp_group[dp_rank]:
                    raise ValueError(f"req: {req_idx} out of memory, need block:" +
                                     f"{total_need_blocks_per_dp_group[dp_rank] + cur_need_blocks} is more than " +
                                     f"free block {free_block_per_dp_group[dp_rank]}")
                if cur_context_len > max_prefill_tokens:
                    raise ValueError(f"req {req_idx}'s  input length is {cur_context_len}, which is longer than " +
                                     f"max_prefill_tokens {max_prefill_tokens}")
                if total_prefill_token_per_dp_group[dp_rank] + cur_context_len > max_prefill_tokens:
                    do_generate = False
                    break
                req_list[req_idx].dp_rank = dp_rank
                total_need_blocks_per_dp_group[dp_rank] += cur_need_blocks
                total_prefill_token_per_dp_group[dp_rank] += cur_context_len
                prefill_batch_size_per_dp_group[dp_rank] += 1
                total_prefill_batch_size += 1
                req_idx += 1
                dp_rank = np.argmax(np.array(free_block_per_dp_group) - np.array(total_need_blocks_per_dp_group))

            if total_prefill_batch_size > 0:
                if is_pp_activated(model):
                    start = prefill_start
                    batch_list = []
                    while start < prefill_start + total_prefill_batch_size:
                        if start + model.mapping.pp.microbatch_size > prefill_start + total_prefill_batch_size:
                            end = prefill_start + total_prefill_batch_size
                        else:
                            end = start + model.mapping.pp.microbatch_size
                        batch = Batch(req_list[start:end])

                        cache_manager.allocate(batch)
                        batch_list.append(batch)
                        start = end
                    req_finished, perf_time = \
                        generate_token_with_clocking(model, cache_manager, batch_list, eplb_forwarder)
                    if ENV.benchmark_enable:
                        prefill_benchmark_timelist.append(perf_time)

                    prefill_summation = 0
                    for i in range(len(batch_list) - 1, -1, -1): # count down batch_list
                        prefill_summation += batch_list[i].batch_num
                        if batch_list[i].batch_num == 0:
                            del batch_list[i]
 
                    if req_finished != (total_prefill_batch_size - prefill_summation):
                        logger.error("batch filter error")
                        raise AssertionError
 
                    if prefill_summation > 0:
                        for batch in batch_list:
                            generate_batches.append(batch)
                            total_generate_batch_size += batch.batch_num

                else:
                    batch = Batch(req_list[prefill_start:prefill_start + total_prefill_batch_size],
                        mapping=model.mapping)
                    cache_manager.allocate(batch)
                    req_finished, perf_time = \
                        generate_token_with_clocking(model, cache_manager, batch, eplb_forwarder)
                    if ENV.benchmark_enable:
                        prefill_benchmark_timelist.append(perf_time)

                    if req_finished != (total_prefill_batch_size - batch.batch_num):
                        raise AssertionError(
                            "Batch filter error: [Prefill] the total number of requests processed "
                            "does not equal the number of requests left + the number of requests completed.")

                    if batch.batch_num > 0:
                        generate_batches.append(batch)
                        total_generate_batch_size += batch.batch_num

                if req_finished > 0:
                    do_generate = False
                    total_req_finished += req_finished

        if do_generate:
            if not is_pp_activated(model) and len(generate_batches) > 1:
                Batch.concatenate(generate_batches)
                if total_generate_batch_size != generate_batches[0].batch_num:
                    raise AssertionError(f"Batch concatenate error, expect batchnum: {total_generate_batch_size}, "
                                         f"in fact: {generate_batches[0].batch_num}")

            batch_used = generate_batches if is_pp_activated(model) else generate_batches[0]
            req_finished, perf_time = \
                generate_token_with_clocking(model, cache_manager, batch_used, eplb_forwarder)
            if ENV.benchmark_enable:
                decoder_benchmark_timelist.append(perf_time)
            
            generate_summation = 0
            if is_pp_activated(model):
                for i in range(len(generate_batches) - 1, -1, -1): # count down
                    generate_summation += generate_batches[i].batch_num
                    if generate_batches[i].batch_num == 0:
                        del generate_batches[i]
                        continue
                    for req in generate_batches[i].req_list:
                        generate_batch_size_per_dp_group[req.dp_rank] += 1
            else:
                generate_summation = generate_batches[0].batch_num
                if generate_summation == 0:
                    del generate_batches[0]
           
            if req_finished != (total_generate_batch_size - generate_summation):
                raise AssertionError("Batch filter error: [Decode] the total number of requests processed does not "
                                    "equal the number of requests left + the number of requests completed.")
            total_generate_batch_size = generate_summation
            total_req_finished += req_finished
            
            generate_batch_size_per_dp_group = [0] * model.mapping.attn_dp.group_size
            if len(generate_batches) > 0:
                for req in generate_batches[0].req_list:
                    generate_batch_size_per_dp_group[req.dp_rank] += 1

    if ENV.benchmark_enable:
        prefill_generate_time = sum(prefill_benchmark_timelist)
        batch_size = len(req_list)
        prefill_average_time = prefill_generate_time / batch_size \
            if batch_size > 0 else 0
        decode_generate_time = sum(decoder_benchmark_timelist)
        e2e_time = prefill_generate_time + decode_generate_time
        try:
            decode_token_time = decode_generate_time / (model.postprocessor.max_new_tokens - 1)
        except ZeroDivisionError:
            decode_token_time = 0

        logger.info(
            f"Prefill time: {prefill_generate_time * 1000}ms, "
            f"Prefill average time: {prefill_average_time * 1000}ms, "
            f"Decode token time: {decode_token_time * 1000}ms, "
            f"E2E time: {e2e_time * 1000}ms")
        input_len = req_list[0].input_length
        output_len = model.postprocessor.max_new_tokens
        prefill_token_times = ','.join(list(map(str, prefill_benchmark_timelist)))
        decode_token_times = ','.join(list(map(str, decoder_benchmark_timelist)))
        if model.rank == 0:
            import os
            benchmark_filepath = ENV.benchmark_filepath \
                if ENV.benchmark_filepath else './benchmark_result/benchmark.csv'
            benchmark_filepath = file_utils.standardize_path(benchmark_filepath)
            benchmark_folder = os.path.dirname(benchmark_filepath)
            if benchmark_folder and not os.path.exists(benchmark_folder):
                os.makedirs(benchmark_folder, mode=0o750, exist_ok=True)
            file_utils.check_file_safety(benchmark_filepath, 'w')
            stat_data = {
                'batch_size': [batch_size],
                'input_seq_len': [input_len],
                'output_seq_len': [output_len],
                'e2e_time(ms)': [f'{e2e_time * 1000: .2f}'],
                'prefill_time(ms)': [f'{prefill_generate_time * 1000: .2f}'],
                'decoder_token_time(ms)': [f'{decode_token_time * 1000: .2f}'],
                'prefill_count': [len(prefill_benchmark_timelist)],
                'prefill_token_times': [prefill_token_times],
                'decode_token_times': [decode_token_times],
                'prefill_average_time(ms)': [f'{prefill_average_time * 1000: .2f}'],
            }
            df = pd.DataFrame(stat_data)
            df.to_csv(benchmark_filepath, index=False)
            os.chmod(benchmark_filepath, 0o640)
            logger.info('-------------------performance dumped------------------------')
            df = df.drop('prefill_token_times', axis=1)
            df = df.drop('decode_token_times', axis=1)
            logger.info(df.to_markdown(index=False))


def decode_token(req_list, tokenizer, skip_special_tokens=False):
    decode_res_list = []
    token_num_list = []
    request_id = 0
    token_num_ = 0
    for req in req_list:
        out_token = len(req.out_token_list)
        token_tensor = torch.tensor(req.out_token_list, dtype=torch.int64)
        if tokenizer is not None:
            decode_text = tokenizer.decode(token_tensor, skip_special_tokens)
            decode_res_list.append(decode_text)
        else:
            decode_res_list.append(token_tensor)
        token_num_ += out_token
        token_num_list.append((request_id, token_num_))
        request_id += 1
    return decode_res_list, token_num_list