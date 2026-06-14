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
from torch import nn

from ..layers.linear.linear_utils import LinearUtils
from .pack_type import TransposeType
from .quant_type import LinearTypeV2
from .low_bit_utils import int42int8


class W4A16LinearStatic(nn.Module, LinearUtils):
    def __init__(self, weight, weight_scale, weight_offset, bias=None):
        super().__init__()
        super(nn.Module, self).__init__()

        self.weight_quant_name = 'w4a16'
        self.linear_desc = LinearTypeV2.W4A16

        # per group 推荐不Transpose，per channel转置
        self.trans_flag = TransposeType.TRANSPOSE if weight_scale.shape[-1] == 1 else TransposeType.NOT_TRANSPOSE

        weight_in_k_n = weight.transpose(-1, -2).contiguous()  # k, n

        weight_trans = weight_in_k_n if self.trans_flag == TransposeType.NOT_TRANSPOSE \
            else weight_in_k_n.transpose(-1, -2).contiguous()

        weight_compact = weight_trans
        if self.quant_version == "0.0.0":
            weight_compact = int42int8(weight_trans)  # [k, n // 2] or [n, k // 2]
        self.register_buffer('weight', weight_compact.to(torch.int8))

        self.register_buffer('weight_scale', weight_scale
                             if self.trans_flag == TransposeType.TRANSPOSE
                             else weight_scale.transpose(-1, -2).contiguous())

        if weight_offset is not None:
            self.register_buffer('weight_offset', (-weight_offset)
                                 if self.trans_flag == TransposeType.TRANSPOSE
                                 else (-weight_offset).transpose(-1, -2).contiguous())
        else:
            self.weight_offset = None

        if bias is not None:
            if bias.dtype == torch.bfloat16:
                bias = bias.to(torch.float32)
            self.register_buffer('bias', bias)
            self.has_bias = True