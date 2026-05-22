# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from torch import nn


class ReduceQuant(nn.Module):
    def __init__(self, scale: list):
        super().__init__()

        reduce_scale, gather_scale = scale
        self.reduce_quant_scale = nn.Parameter(reduce_scale, requires_grad=False)
        self.gather_quant_scale = nn.Parameter(gather_scale, requires_grad=False)

    @classmethod
    def load(cls, prefix, weights):
        reduce_scale = weights.get_tensor(f"{prefix}.reduce_scale")
        gather_scale = weights.get_tensor(f"{prefix}.gather_scale")
        return cls([reduce_scale, gather_scale])