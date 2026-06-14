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
from typing import List
from dataclasses import dataclass

import torch


class Request:
    req_id: int

    input_ids: torch.Tensor
    input_length: int
    position_ids: torch.Tensor

    need_blocks: int
    need_slots: int
    block_tables: torch.Tensor
    slot_tables: torch.Tensor

    dp_rank: int

    out_token_list: List[int]

    sp_rank: int = -1
    input_length_sp: int = 0          # seqlen in cur sp_rank
    input_len_per_sp: List[int] = []  # seqlen in per sp_rank
    input_indices_sp: List[int] = []  # cur sequence indices in cur sp_rank

    cp_rank: int = -1

    def __init__(self, max_out_length: int, block_size: int, req_id: int,
                 input_ids: torch.Tensor, adapter_id: None | str, **kwargs):
        self.req_id = req_id
        self.input_ids = input_ids.flatten()
        self.adapter_id = adapter_id

        self.input_length = self.input_ids.numel()

        self.need_blocks = math.ceil((self.input_length + max_out_length) / block_size)
        self.need_slots = self.need_blocks * block_size
        self.block_tables: None | torch.Tensor = None
        self.slot_tables: None | torch.Tensor = None
        self.dp_rank = -1

        self.out_token_list = []

        mapping = kwargs.get("mapping", None)
        postprocessor = kwargs.get("postprocessor", None)
        pad_token = 0 if postprocessor is None or postprocessor.pad_token_id is None else postprocessor.pad_token_id

        if mapping is not None and mapping.has_attn_inner_sp():
            self.sp_rank = mapping.attn_inner_sp.rank
            sp_size = mapping.attn_inner_sp.group_size

        if mapping is not None and mapping.has_attn_cp():
            self.cp_rank = mapping.attn_cp.rank
            cp_size = mapping.attn_cp.group_size
            if mapping.has_attn_inner_sp() and self.input_length < cp_size * sp_size:
                self.input_ids = torch.cat((torch.full([cp_size * sp_size - self.input_length], pad_token),
                                            self.input_ids), 0)
                self.input_length = self.input_ids.numel()
            # Under load balancing, the sequence needs to be split into 2×cp chunks.
            num_chunks = cp_size * 2
            # Under load balancing, the length of each chunk.
            chunk_length = math.ceil(self.input_length / num_chunks)

            # Pad with bos/eos so that the total number of tokens is a multiple of cp_size*2.
            padding_len = num_chunks * chunk_length - self.input_length
            input_ids_padded = torch.concat([torch.tensor([pad_token] * padding_len), self.input_ids]).to(torch.int64)
            total_position_ids = torch.arange(input_ids_padded.numel(), dtype=torch.long)

            # The input is split according to load balancing,
            # and each cp_rank takes the front and rear two parts respectively.
            former_st_idx = chunk_length * self.cp_rank
            former_end_idx = chunk_length * (self.cp_rank + 1)
            latter_st_idx = chunk_length * (num_chunks - 1 - self.cp_rank)
            latter_end_idx = chunk_length * (num_chunks - self.cp_rank)

            self.input_ids = torch.concat([input_ids_padded[former_st_idx:former_end_idx],
                                           input_ids_padded[latter_st_idx:latter_end_idx]])
            self.position_ids = torch.concat([total_position_ids[former_st_idx:former_end_idx],
                                              total_position_ids[latter_st_idx:latter_end_idx]])
            self.input_length = self.input_ids.numel()
            self.need_blocks = math.ceil((self.input_length + math.ceil(max_out_length / cp_size)) / block_size)
            self.need_slots = self.need_blocks * block_size

        if mapping is not None and mapping.has_attn_inner_sp():
            if self.input_length < sp_size:
                self.input_ids = torch.cat((torch.full([sp_size - self.input_length], pad_token),
                                            self.input_ids), 0)
                self.input_length = sp_size

            input_len_avg = self.input_length // sp_size  # average sequence length pre sp_rank
            remainder = self.input_length % sp_size  # remain tokens

            self.input_len_per_sp = [input_len_avg + 1] * remainder + [input_len_avg] * (sp_size - remainder)
            self.input_length_sp = self.input_len_per_sp[self.sp_rank]

            self.need_blocks = math.ceil(math.ceil((self.input_length + max_out_length) / sp_size) / block_size)
            self.need_slots = self.need_blocks * block_size

            input_indices_per_sp = []
            start_index = 0
            for i in range(sp_size):
                end_index = start_index + self.input_len_per_sp[i]
                input_indices_per_sp.append(list(range(start_index, end_index)))
                start_index = end_index
            self.input_indices_sp = input_indices_per_sp[self.sp_rank]


class MultiModalRequest(Request):
    def __init__(self, max_out_length: int, block_size: int, req_id: int,
                 input_ids: torch.Tensor, adapter_id: None | str, position_ids=None):
        super().__init__(max_out_length, block_size, req_id, input_ids, adapter_id)
        self.req_id = req_id
        self.input_ids = input_ids
        self.adapter_id = adapter_id
        self.input_length = self.input_ids.shape[0]
        self.adapter_id = adapter_id
        self.position_ids = position_ids
        self.context_length = None if position_ids is None else position_ids[-1] + 1
        self.need_blocks = math.ceil((self.input_length + max_out_length) / block_size)
        self.need_slots = self.need_blocks * block_size
        self.block_tables = None
        self.slot_tables = None
        self.out_token_list = []


def request_from_token(input_ids, max_out_length, block_size,
                       req_idx=0, adapter_id=None, **kwargs) -> Request:
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
    request = Request(max_out_length, block_size, req_idx, input_ids, adapter_id, **kwargs)
    return request


def request_from_text(text, tokenizer, max_out_length, block_size, req_idx=0) -> Request:
    input_ids = tokenizer([text], return_tensors="pt")["input_ids"].flatten()
    request = request_from_token(input_ids, max_out_length, block_size, req_idx)
    return request


@dataclass
class MultiModalRequestParams:
    text: str
    image: str
    video: str
    max_out_length: int
    block_size: int
    req_idx: int
    adapter_id: str = None
    batch_size: int = 1
    

@dataclass
class MultiModalReqParams:
    text: List
    image: List
    video: List
    audio: List
    max_out_length: int
    block_size: int
    req_idx: int
    adapter_id: str = None
    batch_size: int = 1


def request_from_multimodalinputs(processor, model, multimodalparams):
    max_out_length = multimodalparams.max_out_length
    block_size = multimodalparams.block_size
    req_idx = multimodalparams.req_idx
    adapter_id = multimodalparams.adapter_id
    
    inputs_embeds = None
    position_ids = None
    prefill_inputs = model.model.prepare_prefill_token(multimodalparams, processor)
    if isinstance(prefill_inputs, tuple):
        inputs_embeds, position_ids = prefill_inputs
    else:
        inputs_embeds = prefill_inputs
    request = MultiModalRequest(max_out_length, block_size, req_idx, inputs_embeds, adapter_id, position_ids)
    return request