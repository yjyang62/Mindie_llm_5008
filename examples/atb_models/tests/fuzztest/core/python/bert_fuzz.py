# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import torch

from base_fuzz import BaseFuzz


class BertFuzz(BaseFuzz):
    def __init__(self, model_name: str, num_layers: int) -> None:
        super().__init__(model_name)
        self.num_layers = num_layers

    def set_weight(self) -> None:
        weight_tensors: list[torch.Tensor] = [
            torch.ones(250002, 1024, dtype=torch.float16).npu(),
            torch.ones(514, 1024, dtype=torch.float16).npu(),
            torch.ones(1, 1024, dtype=torch.float16).npu(),
            torch.ones(1024, dtype=torch.float16).npu(),
            torch.ones(1024, dtype=torch.float16).npu()
        ]
        num_weights_per_layer: int = 16
        for _ in range(self.num_layers):
            for i in range(num_weights_per_layer):
                match i:
                    case 0 | 2 | 4 | 6:
                        weight_tensors.append(torch.ones(1024, 1024, dtype=torch.float16).npu())
                    case 10:
                        weight_tensors.append(torch.ones(4096, 1024, dtype=torch.float16).npu())
                    case 11:
                        weight_tensors.append(torch.ones(4096, dtype=torch.float16).npu())
                    case 12 | 14:
                        weight_tensors.append(torch.ones(1024, 4096, dtype=torch.float16).npu())
                    case _:
                        weight_tensors.append(torch.ones(1024, dtype=torch.float16).npu())
        self.model.set_weight(weight_tensors)

    def execute_fa(self, acl_params: dict, enable_aclnn_attn: bool) -> None:
        mask_dtype = torch.int8 if enable_aclnn_attn else torch.float16
        input_tensors: list[torch.Tensor] = [
            torch.ones(2, 512, dtype=torch.int32).npu(),        # input_ids
            torch.ones(2, 512, dtype=torch.int32).npu(),        # position_ids
            torch.ones(2, 512, dtype=torch.int32).npu(),        # token_type_ids
            torch.ones(2, 512, 512, dtype=mask_dtype).npu(),    # attention_mask
            torch.ones(2, 512, dtype=torch.int32).npu(),        # block_tables
            torch.ones(1, dtype=torch.float16).npu(),           # k_cache
            torch.ones(1, dtype=torch.float16).npu(),           # v_cache
            torch.ones(2, dtype=torch.int32).npu(),             # token_offset
            torch.ones(2, dtype=torch.int32).npu()              # seq_len
        ]
        for i in range(self.num_layers):
            input_tensors.append(torch.tensor([i], dtype=torch.int32).npu())
        self.model.execute(input_tensors, acl_params)