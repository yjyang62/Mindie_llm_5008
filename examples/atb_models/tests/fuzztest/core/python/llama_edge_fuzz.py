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


class LlamaEdgeFuzz(BaseFuzz):
    def __init__(self, model_name):
        super().__init__(model_name)
  
    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # input_ids
        input_tensors.append(torch.randint(1, 100, (random.randint(10, 1024),)).npu()) # position_ids
        input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # cos_embed
        input_tensors.append(torch.rand(random.randint(10, 4096), random.randint(10, 200)).npu()) # sin_enbed
        input_tensors.append(torch.rand(random.randint(10, 200), random.randint(10, 200)).npu()) # attention mask
        input_tensors.append(torch.randint(1, 1024, (1,)).npu()) # input_length
        input_tensors.append(torch.randint(1, 1024, (1,)).npu()) # lm_head_indices
        
        past_keys_prefill = []
        past_values_prefill = []
        for _ in range(acl_params['numHiddenLayers']):
            past_keys_prefill.append(torch.zeros((1, 20, 200, 64), dtype=torch.float16).npu())
            past_values_prefill.append(torch.zeros((1, 20, 200, 64), dtype=torch.float16).npu())
        input_tensors[7:7 + acl_params['numHiddenLayers']] = past_keys_prefill
        input_tensors[7 + acl_params['numHiddenLayers']:8 + acl_params['numHiddenLayers']*2] = past_values_prefill
        self.model.execute(input_tensors, acl_params)