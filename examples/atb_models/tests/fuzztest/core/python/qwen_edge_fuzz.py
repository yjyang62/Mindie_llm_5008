# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import random
import torch
from base_fuzz import BaseFuzz


class QwenEdgeFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)

    def set_weight(self):
        weight_tensors = []
        num_weights = 1459 # number of weights in qwen3-0.6B
        for _ in range(num_weights):
            weight_tensors.append(torch.rand(random.randint(10, 200), random.randint(250, 1024)).npu())
        self.model.set_weight(weight_tensors)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # input_ids
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # position_ids
        input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # cos_embed
        input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # sin_embed
        input_tensors.append(torch.rand(random.randint(10, 200), random.randint(10, 200)).npu()) # attention mask
        input_tensors.append(torch.randint(1, 1024, (1, random.randint(10, 200))).npu()) # block_tables
        input_tensors.append(torch.randint(1, 1024, (random.randint(10, 1024),)).npu()) # slots
        input_tensors.append(torch.tensor(0, dtype=torch.int).npu()) # kv_cache_idx
        input_tensors.append(torch.rand(1).npu()) # token_offset
        input_tensors.append(torch.tensor(1, dtype=torch.int).npu()) # seq_len
        input_tensors.append(torch.tensor(0, dtype=torch.int).npu()) # logits_indices
        input_tensors.append(torch.tensor(1, dtype=torch.int).npu()) # seq_len_all

        past_keys_prefill = []
        past_values_prefill = []
        num_hidden_layers = 28 # number of layers in qwen3-0.6B
        for _ in range(num_hidden_layers):
            past_keys_prefill.append(torch.zeros((1, 8, 200, 128), dtype=torch.float16).npu())
            past_values_prefill.append(torch.zeros((1, 8, 200, 128), dtype=torch.float16).npu())
        input_tensors[12:12 + num_hidden_layers] = past_keys_prefill
        input_tensors[12 + num_hidden_layers:13 + num_hidden_layers * 2] = past_values_prefill
        self.model.execute(input_tensors, acl_params)
