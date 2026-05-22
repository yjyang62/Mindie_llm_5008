# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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


class BaseFuzz:
    def __init__(self, model_name):
        torch.npu.set_device(0)
        atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
        if atb_speed_home_path is None:
            raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
        lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
        torch.classes.load_library(lib_path)
        self.model = torch.classes.ModelTorch.ModelTorch(model_name)

    def set_param(self, params):
        ret = self.model.set_param(params)
        return ret

    def set_weight(self):
        weight_tensors = []
        num_weight = 87
        for i in range(num_weight):
            match i:
                case 0:
                    weight_tensors.append(torch.rand(129281, 3584).to(torch.float16).npu())
                case 1|2|45|58:
                    weight_tensors.append(torch.rand(7168).to(torch.float16).npu())
                case 5:
                    weight_tensors.append(torch.rand(2112, 7168).to(torch.int8).npu())
                case 6:
                    weight_tensors.append(torch.rand(2112).to(torch.int32).npu())
                case 7:
                    weight_tensors.append(torch.rand(2112).to(torch.int64).npu())
                case 11|12:
                    weight_tensors.append(torch.rand(1536).to(torch.float16).npu())
                case 13:
                    weight_tensors.append(torch.rand(12288, 1536).to(torch.int8).npu())
                case 14:
                    weight_tensors.append(torch.rand(12288).to(torch.int32).npu())
                case 15:
                    weight_tensors.append(torch.rand(12288).to(torch.int64).npu())
                case 25:
                    weight_tensors.append(torch.rand(512).to(torch.float16).npu())
                case 27:
                    weight_tensors.append(torch.rand(64, 128, 512).to(torch.float16).npu())
                case 33:
                    weight_tensors.append(torch.rand(64, 512, 128).to(torch.float16).npu())
                case 39:
                    weight_tensors.append(torch.rand(7168, 8192).to(torch.int8).npu())
                case 40:
                    weight_tensors.append(torch.rand(7168).to(torch.int32).npu())
                case 41:
                    weight_tensors.append(torch.rand(7168).to(torch.int64).npu())
                case 49:
                    weight_tensors.append(torch.rand(18432, 7168).to(torch.int8).npu())
                case 52:
                    weight_tensors.append(torch.rand(18432).to(torch.float16).npu())
                case 53:
                    weight_tensors.append(torch.rand(18432).to(torch.float32).npu())
                case 55:
                    weight_tensors.append(torch.rand(7168, 9216).to(torch.int8).npu())
                case 59:
                    weight_tensors.append(torch.rand(7168).to(torch.float32).npu())
                case 86:
                    weight_tensors.append(torch.rand(8080, 7168).to(torch.float16).npu())
                case 8|16|42:
                    weight_tensors.append(torch.zeros(1).to(torch.int8).npu())
                case _:
                    weight_tensors.append(torch.zeros(1).to(torch.float16).npu())
        self.model.set_weight(weight_tensors)
    
    def set_kv_cache(self):
        kcache_tensors = []
        vcache_tensors = []
        num_kv = random.randint(10, 50)
        for _ in range(num_kv):
            kcache_tensors.append(torch.rand(9, 128, 1, 512).to(torch.float16).npu())
            vcache_tensors.append(torch.rand(9, 128, 1, 64).to(torch.float16).npu())
        self.model.set_kv_cache(kcache_tensors, vcache_tensors)

    def execute(self, acl_params, speculate_enable=None):
        input_tensors = []
        input_tensors.append(torch.randint(1, 100, (64,)).npu()) # input_ids
        input_tensors.append(torch.randint(1, 100, (64,)).npu()) # position_ids
        input_tensors.append(torch.rand(128, 64).to(torch.float16).npu()) # cos_embed
        input_tensors.append(torch.rand(128, 64).to(torch.float16).npu()) # sin_enbed
        input_tensors.append(torch.rand(128, 128).to(torch.float16).npu()) # attention mask
        input_tensors.append(torch.randint(1, 124, (1, 1)).to(torch.int32).npu()) # block_tables
        input_tensors.append(torch.randint(1, 1024, (64,)).to(torch.int32).npu()) # slots
        input_tensors.append(torch.rand(1).to(torch.float16).npu()) # placeholder
        input_tensors.append(torch.rand(512, 7168).to(torch.float16).npu()) # final_hidden_states
        input_tensors.append(torch.rand(1).to(torch.float16).npu()) # placeholder
        input_tensors.append(torch.rand(1).to(torch.float16).npu()) # placeholder
        input_tensors.append(torch.randint(1, 1024, (1,)).to(torch.int32).npu()) # input_length
        input_tensors.append(torch.randint(1, 1024, (8,)).npu()) # lm_head_indices
        input_tensors.append(torch.randint(1, 1024, (4096,1)).to(torch.float16).npu()) # expert_array
        input_tensors.append(torch.randint(1, 1024, (1024,)).to(torch.int32).npu()) # expert_group
        input_tensors.append(torch.rand(1).to(torch.int32).npu()) # one_hot
        input_tensors.append(torch.rand(1).to(torch.int32).npu()) # zero_hot
        input_tensors.append(torch.rand(1).to(torch.float16).npu()) # placeholder
        if speculate_enable:
            input_tensors.append(torch.randint(1, 1024, (random.randint(10, 1024),)).npu())
        input_tensors.append(torch.randint(1, 1024, (64,)).to(torch.int32).npu())
        input_tensors.append(torch.randint(1, 1024, (32,)).npu())
        input_tensors.append(torch.randint(1, 1024, (32,)).npu())
        input_tensors.append(torch.randint(1, 1024, (64,)).to(torch.int32).npu())
        input_tensors.append(torch.randint(1, 1024, (512,)).npu())
        input_tensors.append(torch.randint(1, 1024, (32,)).to(torch.int32).npu())
        input_tensors.append(torch.randint(1, 1024, (1,)).npu())
        input_tensors.append(torch.randint(1, 1024, (1,)).npu())
        input_tensors.append(torch.randint(1, 1024, (64,)).to(torch.int32).npu())
        input_tensors.append(torch.randint(1, 1024, (1,)).to(torch.float16).npu())
        input_tensors.append(torch.randint(1, 1024, (256,)).to(torch.int32).npu())
        input_tensors.append(torch.randint(1, 1024, (4096,)).to(torch.int32).npu())
        input_tensors.append(torch.randint(1, 1024, (1,)).to(torch.float16).npu())
        self.model.execute(input_tensors, acl_params)