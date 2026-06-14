# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement class TensorParallelEmbedding based on text-generation-inference
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from collections import OrderedDict

import torch

from torch import nn
from torch.functional import F


class TensorEmbedding(nn.Module):
    def __init__(self, prefix: str, weights):
        super().__init__()
        weight = weights.get_whole_tensor(f"{prefix}.weight", dim=0)
        num_embeddings = weights.get_shape(f"{prefix}.weight")[0]

        self.min_id = 0
        self.max_id = num_embeddings
        self.null_idx = num_embeddings

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        tensorembedding_input = torch.where(
            torch.lt(input_tensor, self.min_id) | torch.ge(input_tensor, self.max_id),
            self.null_idx,
            input_tensor - self.min_id,
        )
        out = torch.nn.functional.embedding(tensorembedding_input, self.weight)
        return out
    
    def get_weights(self, prefix):
        weight_dict = OrderedDict()
        weight_dict[f"{prefix}.weight"] = self.weight.data
        return weight_dict


class TensorParallelEmbedding(nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_partial_sharded(f"{prefix}.weight", dim=1)
        num_embeddings = weights.get_shape(f"{prefix}.weight")[0]

        process_group = weights.process_group

        world_size = process_group.size()
        rank = process_group.rank()

        block_size = num_embeddings // world_size
        self.min_id = rank * block_size
        self.max_id = min(num_embeddings, (rank + 1) * block_size)
        self.null_idx = block_size
        self.process_group = weights.process_group
        self.reduce = reduce

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        tensorparallelembedding_input = torch.where(
            torch.lt(input_tensor, self.min_id) | torch.ge(input_tensor, self.max_id),
            self.null_idx,
            input_tensor - self.min_id,
        )
        out = torch.nn.functional.embedding(tensorparallelembedding_input, self.weight)
        if self.reduce and self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out

    def get_weights(self, prefix):
        weight_dict = OrderedDict()
        weight_dict[f"{prefix}.weight"] = self.weight.data
        return weight_dict
