# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import math
import torch
import torch_npu
import numpy as np

from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.env import ENV
from examples.convert.model_slim.get_razor_attention_wins import get_global_wins


class HiddenStateCache:
    def __init__(self, hidden_size, mtp, max_batch_size, dtype, device):
        self.map = {}
        self.hidden_size = hidden_size
        self.mtp_num = mtp
        self.hidden_states_cache = torch.zeros((max_batch_size, mtp + 1, hidden_size), dtype=dtype, device=device)
        self.used_idx = np.zeros(max_batch_size, dtype=np.int64)

    def save_input_cache(self, req_ids, is_prefill):
        cache_ids = np.zeros(len(req_ids), np.int32)
        for i, req_id in enumerate(req_ids):
            if req_id not in self.map:
                if is_prefill:
                    cache_ids[i] = self.get_cache_id(req_id)
            else:
                cache_ids[i] = self.map[req_id]
        return cache_ids

    def get_cache_id(self, req_id):
        available_ids = np.flatnonzero(self.used_idx == 0)
        available_id = available_ids[0]
        self.used_idx[available_id] = 1
        self.map[req_id] = available_id
        return available_id

    def save_hidden_states(self, cache_ids, hidden_states):
        self.hidden_states_cache[cache_ids] = hidden_states.view(-1, self.mtp_num + 1, self.hidden_size)

    def get_hidden_states(self, cache_ids):
        return self.hidden_states_cache[cache_ids].view(-1, self.hidden_size)
    
    def clear_cache(self, req_ids):
        for req_id in req_ids:
            if req_id in self.map:
                cache_id = self.map[req_id]
                self.map.pop(req_id)
                self.used_idx[cache_id] = 0


class CacheConfig:
    def __init__(self, num_blocks=1024, block_size=128, input_max_length=2048, output_max_length=128, batch_size=1,
                 rank=0, world_size=1):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.pattern_path = ENV.omni_attention_pattern_file # for Omni attention


class ModelConfig:
    def __init__(self, num_heads, num_kv_heads, num_kv_heads_origin, k_head_size, v_head_size,
                num_layers, device, dtype, soc_info, kv_quant_type, fa_quant_type=None,
                mapping=None, kvcache_quant_layers=None, cla_share_factor=1, model_type=None, enable_nz=False):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_origin = num_kv_heads_origin
        self.head_size = k_head_size
        self.k_head_size = k_head_size
        self.v_head_size = v_head_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.soc_info = soc_info
        self.kv_quant_type = kv_quant_type
        self.fa_quant_type = fa_quant_type
        self.mapping = mapping
        self.kvcache_quant_layers = kvcache_quant_layers
        self.cla_share_factor = cla_share_factor
        self.model_type = model_type
        self.enable_nz = enable_nz

    def __repr__(self):
        return (
                "ModelConfig("
                + f"num_heads={self.num_heads}, "
                + f"num_kv_heads={self.num_kv_heads}, "
                + f"num_kv_heads_origin={self.num_kv_heads_origin}, "
                + f"head_size={self.head_size}, "
                + f"k_head_size={self.k_head_size}, "
                + f"v_head_size={self.v_head_size}, "
                + f"num_layers={self.num_layers}, "
                + f"device={self.device}, "
                + f"dtype={self.dtype}, "
                + f"soc_info={self.soc_info}, "
                + f"kv_quant_type={self.kv_quant_type}, "
                + f"fa_quant_type={self.fa_quant_type}, "
                + f"mapping={self.mapping}, "
                + f"cla_share_factor={self.cla_share_factor}, "
                + f"model_type={self.model_type}, "
                + f"enable_nz={self.enable_nz})"
        )


class CacheManager:
    def __init__(self, cache_config, model_config):
        self.block_size = cache_config.block_size
        self.num_blocks = cache_config.num_blocks
        self.new_num_blocks = self.num_blocks
        self.input_max_length = cache_config.input_max_length
        self.output_max_length = cache_config.output_max_length
        self.batch_size = cache_config.batch_size
        self.rank = cache_config.rank
        self.world_size = cache_config.world_size        
        self.compress_head_enable = ENV.compress_head_enable
        self.compress_head_rope = ENV.compress_head_rope
        self.omni_attention_enable = ENV.omni_attention_enable
        self.mod_mask = None
        self.num_heads = 1 if self.compress_head_enable else model_config.num_kv_heads
        self.num_kv_heads = model_config.num_kv_heads 
        self.k_head_size = model_config.k_head_size
        self.v_head_size = model_config.v_head_size
        self.num_layers = model_config.num_layers
        self.enable_nz = model_config.enable_nz
        if ENV.deepseek_mtp:
            self.num_layers += ENV.deepseek_mtp
            self.hidden_state_cache = None
        self.layer_list = model_config.mapping.pp_layers(self.num_layers) if model_config.mapping is not None \
            else list(range(self.num_layers))
        self.v_cache_share_fractor = model_config.cla_share_factor
        self.device = model_config.device
        self.dtype = torch.int8 if model_config.kv_quant_type is not None or \
            model_config.fa_quant_type is not None else model_config.dtype
        self.v_cache_dtype = model_config.dtype if model_config.fa_quant_type == "FAKQuant" else self.dtype
        self.soc_info = model_config.soc_info
        self.model_type = model_config.model_type
        self.attn_dp_size = model_config.mapping.attn_dp.group_size if model_config.mapping is not None else 1
        self.enable_data_parallel = model_config.mapping.has_dp() if model_config.mapping is not None else False
        if self.enable_data_parallel:
            # dp场景下多申请一个block用于存放陪跑数据
            self.num_blocks += 1
            self.new_num_blocks += 1

        if self.v_cache_share_fractor < 1:
            logger.error("cross layer attention param error, v cache should be shared by at least one layer")
            raise AssertionError
        mem_need = self.num_blocks * self.block_size * self.num_heads * (self.k_head_size * self.num_layers \
                    + self.v_head_size * self.num_layers / self.v_cache_share_fractor) * \
                    self.get_dtype_size(self.dtype) / 1024 / 1024 / 1024
        logger.info(f"kv cache will allocate {mem_need}GB memory")

        if self.compress_head_enable:
            if self.compress_head_rope:
                head_dict, first_sink = get_global_wins(self.model_type, self.num_layers)
                if head_dict is not None and first_sink is not None:
                    inductive_head = head_dict.get("prefix_matching")
                    copying_head = head_dict.get("copying")
                    last_sink = max(4000, self.input_max_length // 5)
                    self.new_layers_num_blocks = []
                    kv_tp_size = min(cache_config.world_size, model_config.num_kv_heads_origin)
                    for layer_idx in range(self.num_layers):
                        global_need_block = 0
                        for head_idx in range(model_config.num_kv_heads):
                            cur_head_idx = head_idx + self.rank * kv_tp_size // \
                                self.world_size * model_config.num_kv_heads
                            is_inductive_head = layer_idx in inductive_head \
                                and cur_head_idx in inductive_head.get(layer_idx)
                            is_copying_head = layer_idx in copying_head and cur_head_idx in copying_head[layer_idx]
                            if (is_inductive_head or is_copying_head) or \
                                (self.input_max_length - first_sink - last_sink - 1 <= 0):
                                temp_length = self.input_max_length + self.output_max_length
                            else:
                                temp_length = first_sink + 1 + last_sink + self.output_max_length

                            need_block = math.ceil(temp_length / self.block_size)
                            global_need_block = global_need_block + need_block
                        self.new_layers_num_blocks.append(global_need_block)
                else:
                    self.new_layers_num_blocks = self.new_num_blocks * model_config.num_kv_heads
            else:
                wins = [
                    105, 125, 148, 176, 210, 250, 297, 353, 420, 500, 595, 707, 841, 1001, 1190, 1415, 1683, 2002, 2381,
                    2831, 3367, 4004, 4762, 5663, 6734, 8008, 9524, 11326, 13469, 16017, 19048, 22652
                ]
                if self.num_layers == 40:
                    wins = [
                        105, 125, 149, 178, 211, 251, 299, 356, 423, 503,
                        598, 712, 847, 1007, 1198, 1424, 1694, 2014, 2396, 2849,
                        3388, 4031, 4791, 5699, 6779, 8061, 9583, 11399, 13559, 16117,
                        19176, 22790, 97, 115, 137, 163, 194, 230, 274, 326
                    ]
                temp_c = self.input_max_length
                all_block_num = 0
                temp_length = 0
                num_block = 0
                for wins_item in enumerate(wins):
                    wins_index = wins_item[0]
                    wins_val = wins_item[1]
                    temp_length = min(wins_val, temp_c) + self.output_max_length
                    if self.block_size != 0:
                        num_block = num_block + math.ceil(temp_length / self.block_size)
                    if (wins_index + 1) % model_config.num_kv_heads_origin == 0:
                        all_block_num = max(all_block_num, num_block)
                        temp_length = 0
                        num_block = 0
                self.new_num_blocks = all_block_num * self.batch_size + 100
        elif self.omni_attention_enable:
            self.first_sink = 128
            self.last_sink = 256
            self.num_blocks_per_head = torch.ones(
                (self.num_layers, model_config.num_kv_heads), 
                dtype=torch.int32, 
                device=self.device
            ) * math.ceil((self.input_max_length + self.output_max_length) / self.block_size)
            streaming_mask = torch.tensor(np.loadtxt(cache_config.pattern_path), device="npu").to(bool)
            streaming_mask = streaming_mask[
                :, 
                self.rank * model_config.num_kv_heads:(self.rank + 1) * model_config.num_kv_heads
            ]
            if self.input_max_length > self.first_sink + self.last_sink:
                if ENV.omni_shift_windows_enable:
                    self.num_blocks_per_head[streaming_mask] = math.ceil(
                        (self.first_sink + self.last_sink) / self.block_size
                    )
                else:
                    self.num_blocks_per_head[streaming_mask] = math.ceil(
                        (self.first_sink + self.last_sink + self.output_max_length) / self.block_size
                    )
                self.new_layers_num_blocks = self.num_blocks_per_head.sum(dim=1)
            self.new_layers_num_blocks = self.new_layers_num_blocks.int().tolist()

            self.new_layers_num_blocks = [tmp * self.batch_size for tmp in self.new_layers_num_blocks]
            mem_cal_blocks = sum(self.new_layers_num_blocks)

            mem_need = (
                mem_cal_blocks * self.num_heads * self.block_size * 
                (self.k_head_size + self.v_head_size) * 
                self.get_dtype_size(self.dtype) / 
                1024 / 1024 / 1024
            )
            logger.info(f"kv cache will allocate {mem_need}GB memory")

            # build the block index, [layer, head] -> [BS, layer, head]
            self.streaming_mask = streaming_mask.unsqueeze(0).repeat(self.batch_size, 1, 1)
            self.block_tables = self.generate_block_tables(shift_windows=ENV.omni_shift_windows_enable)
            self.global_block_int_offsets_per_head = (self.block_tables[:, :, :, 0] * self.block_size).to(torch.int32)
            self.max_blocks_per_layer = max(self.new_layers_num_blocks)
            self.free_batch_mask = torch.ones((self.batch_size), dtype=torch.bool)

        per_layer_kv_cache_dtype = [self.v_cache_dtype for layer_id in range(self.num_layers)]
        if model_config.kvcache_quant_layers:
            for i, kvcache_quant in enumerate(model_config.kvcache_quant_layers):
                if kvcache_quant:
                    per_layer_kv_cache_dtype[i] = self.dtype

        if self.soc_info.need_nz or self.enable_nz:
            v_cache_shape = (self.new_num_blocks, self.num_heads * self.v_head_size // 16, self.block_size, 16)
            self.kv_cache = [
                (
                    torch_npu.empty_with_format(
                        (self.new_num_blocks, self.num_heads * self.k_head_size // 16, self.block_size, 16) 
                            if dtype != torch.int8 else
                            # 32: nd to nz for int8
                            (self.new_num_blocks, self.num_heads * self.k_head_size // 32, self.block_size, 32),
                        dtype=dtype,
                        device=self.device,
                        acl_format=29       # nz format
                    ),
                    torch_npu.empty_with_format(
                        v_cache_shape, dtype=self.v_cache_dtype, device=self.device, acl_format=29      # nz format
                    ) if self.v_head_size > 0 and layer_id % self.v_cache_share_fractor == 0
                    else torch.empty((1,), dtype=self.v_cache_dtype, device=self.device)
                )
                for layer_id, dtype in zip(range(self.num_layers), per_layer_kv_cache_dtype)
            ]
        else:
            self.kv_cache = [
                (
                    torch.empty(
                        (self.new_layers_num_blocks[layer_id] if (self.compress_head_rope or
                                                                  self.omni_attention_enable)
                         else self.new_num_blocks, self.block_size, self.num_heads, self.k_head_size),
                        dtype=dtype, device=self.device,
                    ),
                    torch.empty(
                        self.get_v_cache_shape(layer_id),
                        dtype=self.v_cache_dtype,
                        device=self.device,
                    ),
                )
                for layer_id, dtype in zip(self.layer_list, per_layer_kv_cache_dtype)
            ]

        random_block_allocate = False
        if random_block_allocate:
            self.block_map = torch.randperm(self.new_num_blocks, dtype=torch.long)
            self.contrary_block_map = torch.zeros(self.new_num_blocks, dtype=torch.long)
            for i in range(self.new_num_blocks):
                self.contrary_block_map[self.block_map[i]] = i
        else:
            self.block_map = torch.arange(self.new_num_blocks, dtype=torch.long)
            self.contrary_block_map = torch.arange(self.new_num_blocks, dtype=torch.long)

        if self.enable_data_parallel:
            # DP场景下最后一个block对外不可见，仅用于存放陪跑数据
            self.free_block_mask = [torch.ones(self.new_num_blocks - 1, dtype=torch.long)
                                    for _ in range(model_config.mapping.attn_dp.group_size)]
        else:
            self.free_block_mask = [torch.ones(self.new_num_blocks, dtype=torch.long)]
        self.total_slots = torch.arange(self.new_num_blocks * self.block_size, dtype=torch.long)
        self.total_slots = self.total_slots.view(self.new_num_blocks, self.block_size)

    @staticmethod
    def get_dtype_size(dtype):
        dtype_size_map = {torch.float16: 2, torch.float32: 4, torch.bfloat16: 2, torch.int8: 1}
        return dtype_size_map.get(dtype, 2)

    def generate_block_tables(self, shift_windows=True):
        max_kvcache_lengths = torch.full(
            (self.batch_size, self.num_layers, self.num_kv_heads), 
            self.input_max_length + self.output_max_length, 
            dtype=torch.int32).npu() # 每个头最大的kv_cache长度
        max_kvcache_lengths[self.streaming_mask] = self.first_sink + self.last_sink
        max_kvcache_need_blocks = torch.ceil(max_kvcache_lengths / self.block_size).to(torch.int32)
        max_need_blocks_per_head = math.ceil((self.input_max_length + self.output_max_length) / self.block_size)
        block_indices = (
            torch.arange(max_need_blocks_per_head, dtype=torch.int32).npu()
            .expand(self.batch_size, self.num_layers, self.num_kv_heads, max_need_blocks_per_head)
        )

        global_offsets = (
            max_kvcache_need_blocks
            .transpose(0, 1)
            .reshape(self.num_layers, self.batch_size * self.num_kv_heads)
        )
        global_offsets = torch.cumsum(global_offsets, dim=-1, dtype=torch.int32) - global_offsets
        global_offsets = global_offsets.reshape(self.num_layers, self.batch_size, self.num_kv_heads).transpose(0, 1)
        block_tables = block_indices + global_offsets.unsqueeze(-1) # The block_tables without shift_windows recircle

        if shift_windows:
            if self.first_sink % self.block_size != 0 or self.last_sink % self.block_size != 0:
                raise AssertionError
            recircle_blocks_mask = (max_kvcache_need_blocks.unsqueeze(-1) <= 
                                    block_indices) # The mask of kvcache to get kv gernerate at decoder stage
            first_sink_blocks = int(self.first_sink // self.block_size) # Per first sink allocate blocks nums
            last_sink_blocks = int(self.last_sink // self.block_size) # Per last sink allocate
            block_indices = (block_indices - first_sink_blocks) % last_sink_blocks + first_sink_blocks # kww: recircle!
            blocks_streaming_mask = self.streaming_mask.unsqueeze(-1).expand_as(block_indices)
            broadcasted_block_indices = block_tables.clone()
            broadcasted_block_indices[blocks_streaming_mask] = (
                (block_indices + global_offsets.unsqueeze(-1))[blocks_streaming_mask] 
            ) # recircle to streaming head
            recircle_blocks_mask_indices = recircle_blocks_mask.nonzero(as_tuple=True)
            block_tables[recircle_blocks_mask_indices] = broadcasted_block_indices[recircle_blocks_mask]

        return block_tables
    
    def allocate_compress_head(self, batch):
        # get the free batch_mask
        free_index = self.free_batch_mask.nonzero().flatten()
        if len(free_index) < batch.batch_num:
            raise ValueError(f"Current Available Batch {len(free_index)} is less than "
            f"the prefill need {batch.batch_num}")
        free_index = free_index[:batch.batch_num]
        batch.allocate_index = free_index

        ## 1. Get free Batch.block_tables
        batch_block_tables = self.block_tables[free_index]
        batch.batch_block_tables = batch_block_tables

        ## 2. Get free Batch.slots
        slots = self.global_block_int_offsets_per_head[free_index]
        batch.global_block_int_offsets_per_head = slots
        batch.batch_slots_tables = slots
        ## 3. Get Free Batch.patterns
        streaming_mask = self.streaming_mask[free_index]
        batch.streaming_mask = streaming_mask

        ## 4. Get the % tensor, ignore decoder have index ops
        mod_mask = torch.full(
            slots.size(), 
            self.input_max_length + self.output_max_length,
            dtype=torch.int32).npu() # 每个头最大的kv_cache长度
        mod_mask[streaming_mask] = self.last_sink
        self.mod_mask = mod_mask[0][None].transpose(0, 1).contiguous() # Layer, BS=0, Head

        ## 5. for batch seq_len & in_ra_seq_lens
        wins_keep = batch.context_length.unsqueeze(-1).unsqueeze(-1).expand_as(slots).to(torch.int32)
        wins_keep = wins_keep.to(slots.device)
        context_length = wins_keep.clone()
        prefill_compressed_mask = streaming_mask.clone()
        bs_prefill_need_compressed_mask = batch.context_length > (self.first_sink + self.last_sink)
        prefill_compressed_mask[~bs_prefill_need_compressed_mask] = 0
        wins_keep[prefill_compressed_mask] = self.first_sink + self.last_sink
        seq_len = wins_keep - 1 # for slos index
        batch.after_compress_seqlens = wins_keep # the length after compressed equals to slots index +1
        batch.seq_len = seq_len

        ## 6. The compress length after prefill
        batch.wins = torch.zeros((batch.batch_num, self.num_kv_heads), dtype=torch.int32).npu()
        batch.wins[bs_prefill_need_compressed_mask] = (
            batch.context_length[bs_prefill_need_compressed_mask]
            .to(batch.wins.device)
            .to(batch.wins.dtype)
            .unsqueeze(-1) 
            - self.first_sink 
            - self.last_sink
        )

        ## 7.  for batch.pffset_index
        batch.pffset_index = -1 * torch.ones(
            (batch.batch_num, self.num_layers, self.num_kv_heads), 
            dtype=torch.int32).npu()
        batch.pffset_index[prefill_compressed_mask] = self.first_sink

        ## 8. Batch.wins_drop
        wins_drop = torch.ones((batch.batch_num, self.num_layers, self.num_kv_heads), dtype=torch.int32).npu()
        wins_drop = wins_drop * -1 * (self.first_sink + self.last_sink) # no mask,
        wins_drop += context_length.to(wins_drop.device)

        ## 9. Seq use in reshap_and_cache:
        batch.in_reshape_seqlen = batch.context_length.npu().to(torch.int32)

        ## 10. Batch.ra_offset
        batch_max_blocks_per_layer = int(self.max_blocks_per_layer / self.batch_size * batch.batch_num)
        batch.batch_max_blocks_per_layer = batch_max_blocks_per_layer
        batch.ra_offset = torch.zeros(
            (self.num_layers, batch_max_blocks_per_layer * self.block_size), 
            dtype=torch.float32).npu()
        layer_indices = (
            torch.arange(self.num_layers, dtype=torch.int32, device="npu")
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch.batch_num, self.num_layers, self.num_kv_heads)
        )
        valid_offsets = slots + self.first_sink
        mask = wins_drop > 0 # compressed head BS,Layers,Heads
        log_wins_drop = torch.log(wins_drop) # BS,Layers,Heads
        valid_offsets_flat = valid_offsets[mask]
        layer_indices_flat = layer_indices[mask]
        batch.ra_offset.index_put_((layer_indices_flat, valid_offsets_flat), log_wins_drop[mask], accumulate=False)

        # Transpose
        batch.transpose_input_tensor()

        # contiguous
        batch.contiguous_input_tensor()
        # batch_mask
        self.free_batch_mask[free_index] = False
    
    def allocate(self, batch):
        if ENV.omni_attention_enable:
            self.allocate_compress_head(batch)
            return
        total_need_blocks_per_dp_group = [0] * self.attn_dp_size
        max_need_blocks = 0
        # 每个dp组所有请求所需block总数，以及单请求所需最大block数
        for req in batch.req_list:
            if req.block_tables:
                error_msg = f"req_id: {req.req_id} block has been allocated."
                logger.error(error_msg, ErrorCode.ATB_MODELS_INTERNAL_ERROR)
                raise AssertionError(error_msg)

            total_need_blocks_per_dp_group[req.dp_rank] += req.need_blocks
            max_need_blocks = max(max_need_blocks, req.need_blocks)

        allocate_block_indices = []
        allocate_blocks = []
        for i in range(self.attn_dp_size):
            free_block_indices = self.free_block_mask[i].nonzero().flatten()
            if free_block_indices.numel() < total_need_blocks_per_dp_group[i]:
                error_msg = f"Out of available cache blocks: asked {total_need_blocks_per_dp_group[i]}, " \
                            f"only {free_block_indices.numel()} free blocks."
                logger.error(error_msg, ErrorCode.ATB_MODELS_INTERNAL_ERROR)
                raise AssertionError(error_msg)

            allocate_block_indices.append(free_block_indices[:total_need_blocks_per_dp_group[i]])
            allocate_blocks.append(self.block_map[allocate_block_indices[i]])

        block_offset = [0] * self.attn_dp_size
        block_tables_list = []
        slot_tables_list = []
        for req in batch.req_list:
            req.block_tables = allocate_blocks[req.dp_rank][block_offset[req.dp_rank]:
                                                            block_offset[req.dp_rank] + req.need_blocks]
            req.slot_tables = self.total_slots[req.block_tables].flatten()
            block_tables = req.block_tables
            if req.need_blocks < max_need_blocks:
                block_tables = torch.concat(
                    [block_tables, torch.zeros(max_need_blocks - req.need_blocks, dtype=torch.long)], dim=0)
            block_tables_list.append(block_tables.view(1, -1))
            slot_tables_list.append(req.slot_tables)
            block_offset[req.dp_rank] += req.need_blocks

        batch.batch_block_tables = torch.concat(block_tables_list, dim=0)
        batch.batch_slots_tables = torch.concat(slot_tables_list, dim=0)

        for i in range(self.attn_dp_size):
            self.free_block_mask[i][allocate_block_indices[i]] = 0

    def free_omni(self, idx=None):
        if idx is None:
            # Both Decoding or Prefilling
            self.free_batch_mask = torch.ones(self.batch_size, dtype=torch.bool)
        else:
            self.free_batch_mask[idx] = True
    
    def free(self, block_tables_map):
        if block_tables_map is not None:
            for dp_rank, block_tables in block_tables_map.items():
                block_indices = self.contrary_block_map[block_tables]
                self.free_block_mask[dp_rank][block_indices] = 1

    def free_all(self):
        for free_block_mask in self.free_block_mask:
            free_block_mask[:] = 1

    def get_free_block_num(self, dp_rank):
        free_block_indices = self.free_block_mask[dp_rank].nonzero()
        return len(free_block_indices)

    def get_v_cache_shape(self, layer_id):
        if self.v_head_size == 0 or layer_id % self.v_cache_share_fractor != 0:
            return (1, 1, 1, 1)
        elif self.compress_head_rope or self.omni_attention_enable:
            return (self.new_layers_num_blocks[layer_id], self.block_size, self.num_heads, self.v_head_size)
        return (self.new_num_blocks, self.block_size, self.num_heads, self.v_head_size)