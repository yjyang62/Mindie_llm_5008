# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import random
import torch
import torch_npu
from base_fuzz import BaseFuzz


class TelechatFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)
  
    def set_weight(self):
        weight_tensors = []
        num_weight = 1903
        for _ in range(num_weight):
            weight_tensors.append(torch.rand(random.randint(10, 200), random.randint(100, 1024)).npu())
        self.model.set_weight(weight_tensors)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # input_ids
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # position_ids
        input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # cos_embed
        input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # sin_enbed
        input_tensors.append(torch.rand(random.randint(10, 200), random.randint(10, 200)).npu()) # attention mask
        input_tensors.append(torch.randint(1, 1024, (1, random.randint(10, 200))).npu()) # block_tables
        input_tensors.append(torch.randint(1, 1024, (random.randint(10, 1024),)).npu()) # slots
        input_tensors.append(torch.rand(1).npu()) # placeholder
        input_tensors.append(torch.rand(1).npu()) # placeholder
        input_tensors.append(torch.rand(1).npu()) # placeholder
        input_tensors.append(torch.randint(1, 1024, (1,)).npu()) # input_length
        input_tensors.append(torch.randint(1, 1024, (1,)).npu()) # lm_head_indices
        self.model.execute(input_tensors, acl_params)