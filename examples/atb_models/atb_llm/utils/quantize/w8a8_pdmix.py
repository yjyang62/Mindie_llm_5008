#!/usr/bin/env python
# coding=utf-8
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

from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.quantize.quant_type import LinearTypeV2


class W8A8PDMixLinear(nn.Module, LinearUtils):
    def __init__(self, weight, weight_scale, weight_offset, deq_scale, quant_bias,
                 input_scale, input_offset, bias=None, need_flatten=True):
        super().__init__()
        super(nn.Module, self).__init__()
        self.weight_quant_name = 'per_channel'
        self.trans_flag = self.check_transpose(weight)
        self.linear_desc = LinearTypeV2.W8A8_PDMIX

        self.register_buffer('weight', weight.to(torch.int8)
                             if self.trans_flag == TransposeType.TRANSPOSE
                             else weight.transpose(-1, -2).contiguous().to(torch.int8))

        weight_scale_dtype = weight_scale.dtype if weight_scale.dtype == torch.bfloat16 else torch.float32
        self.register_buffer('weight_scale', weight_scale.to(weight_scale_dtype).flatten()
                            if need_flatten else weight_scale.to(weight_scale_dtype))
        self.register_buffer('deq_scale', deq_scale)
        if quant_bias is not None:
            self.register_buffer('quant_bias', quant_bias)
            self.has_bias = True
        else:
            self.quant_bias = None

        self.register_buffer('input_scale', input_scale)
        if input_offset is not None:
            self.register_buffer('input_offset', input_offset.to(torch.int8))
        else:
            self.register_buffer('input_offset', torch.tensor([], dtype=torch.int8))
        if weight_offset is not None:
            self.register_buffer('weight_offset', -(weight_offset.flatten())
                                if need_flatten else -(weight_offset))
        else:
            self.weight_offset = None
        if bias is not None:
            self.register_buffer('bias', bias)
            self.has_bias = True