# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import torch
from torch import nn
import numpy as np
import torch_npu

from ..layers.linear.linear_utils import LinearUtils
from .pack_type import TransposeType
from .quant_type import LinearTypeV2


WEIGHT = 'weight'


class W4A8LinearDynamic(nn.Module, LinearUtils):
    def __init__(self, weight, scale, scale_second=None, bias=None, is_sharded=False):
        super().__init__()
        super(nn.Module, self).__init__()

        self.weight_quant_name = 'w4a8_dynamic'
        self.linear_desc = LinearTypeV2.W4A8_DYNAMIC
        # per group 推荐不Transpose，per channel转置
        self.trans_flag = TransposeType.NOT_TRANSPOSE
        self.has_bias = True

        if is_sharded:
            self.register_buffer(WEIGHT, weight)
            self.register_buffer('weight_scale', scale)
            self.register_buffer('bias', bias)
            return

        is_dense = weight.dim() == 2
        scale_second_in_k_n = None
        weight_in_k_n = weight.transpose(-1, -2).contiguous()  # k, n
        if scale_second is not None:
            scale_in_k_n, scale_second_in_k_n = scale.transpose(-1, -2), scale_second.transpose(-1, -2) # k, n
        else:
            scale_in_k_n = scale.transpose(-1, -2)

        if is_dense:
            scale_multi = (scale_in_k_n * scale_second_in_k_n).to(torch.float32)
            group_num, n = scale_multi.shape
            scale_uint32 = torch_npu.npu_trans_quant_param(scale_multi.npu().reshape([group_num * n, ])).cpu()
            scale_uint32 = scale_uint32.reshape([group_num, n])
            weight_compact = torch.from_numpy(np.frombuffer(weight_in_k_n.numpy().tobytes(), dtype=np.int32))
            weight_compact = weight_compact.reshape(-1, n // 8)
            bias = bias.sum(axis=1).to(torch.float32)
        else:
            weight_compact = weight_in_k_n
            bias = bias.transpose(-1, -2).sum(axis=1)
            scale_uint32 = self.process_scale(weight_in_k_n, scale_in_k_n, scale_second_in_k_n)
        
        if is_dense:
            self.register_buffer(WEIGHT, weight_compact)
        else:
            self.register_buffer(WEIGHT, weight_compact.to(torch.int8))

        self.register_buffer('weight_scale', scale_uint32)
        if bias.dtype == torch.bfloat16:
            bias = bias.to(torch.float32)
        self.register_buffer('bias', bias)

    @classmethod
    def weight_format_cast(cls, tensor, enable_nz=False, nd_weight=False):
        need_nz = enable_nz or cls.soc_info.need_nz or cls.soc_info.matmul_nd_nz
        if need_nz and not nd_weight:
            torch.npu.config.allow_internal_format = True
            torch_npu.npu_format_cast_(tensor, 29)
        return tensor

    def process_scale(self, weight: torch.Tensor, scale, per_group_scale):
        group_num, k, n = weight.shape
        n = n * 2
        if per_group_scale is not None:
            per_group_scale = per_group_scale.reshape(group_num, -1, n)
            group_num, quantgroup_num, n = per_group_scale.shape   
        else:
            quantgroup_num = 1
        
        if per_group_scale is not None:
            scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
        else:
            scale_fp32 = scale.to(torch.float16).to(torch.float32)
        scale_fp32_np = scale_fp32.numpy()
        scale_fp32_np.dtype = np.uint32
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2), dtype=np.uint32) 
        
        sscale_uint64[..., ::2] = scale_fp32_np
        
        sscale_uint64_buffer = np.frombuffer(sscale_uint64.tobytes(), dtype=np.int64).copy()
        sscale_uint64_tensor = torch.from_numpy(sscale_uint64_buffer).reshape(group_num, quantgroup_num, n)
        return sscale_uint64_tensor