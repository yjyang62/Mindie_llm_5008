# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
from .quant_type import LinearTypeV2


class W8A8SparseCompressedLinear(nn.Module, LinearUtils):
    def __init__(self, weight, deq_scale, input_scale, quant_bias=None, input_offset=None, index=None):
        super().__init__()
        super(nn.Module, self).__init__()
        self.linear_desc = LinearTypeV2.W8A8SC

        self.register_buffer('weight', weight.to(torch.int8))

        self.register_buffer('input_scale', input_scale.to(torch.float16))

        self.register_buffer('input_offset', input_offset.to(torch.int8))

        self.register_buffer('deq_scale', deq_scale)

        self.register_buffer('quant_bias', quant_bias)
        self.has_bias = True

        self.register_buffer('index', index)