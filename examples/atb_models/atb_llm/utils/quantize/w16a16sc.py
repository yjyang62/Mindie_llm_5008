# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from torch import nn

from ..layers.linear.linear_utils import LinearUtils
from .quant_type import LinearTypeV2


class W16A16SparseCompressedLinear(nn.Module, LinearUtils):
    def __init__(self, weight, index, quant_bias):
        super().__init__()
        super(nn.Module, self).__init__()
        self.linear_desc = LinearTypeV2.W16A16SC

        self.register_buffer('weight', weight)

        self.register_buffer('quant_bias', quant_bias)
        self.has_bias = True
 
        self.register_buffer('index', index)