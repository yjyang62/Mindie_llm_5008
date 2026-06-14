#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

from .linear_utils import LinearUtils
from ....utils.quantize.quant_type import LinearTypeV2


class FastLinear(nn.Module, LinearUtils):
    def __init__(
            self,
            weight,
            bias,
            is_norm=False,
            nd_weight=False
    ) -> None:
        super().__init__()
        super(nn.Module, self).__init__()
        if not isinstance(weight, torch.Tensor) or weight.dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("linear type not matched, please check `config.json` `quantize` parameter")
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(weight.dtype))
            self.has_bias = True
        else:
            self.bias = None

        self.transpose_weight_as_need()
        self.linear_desc = LinearTypeV2.FLOAT16 if weight.dtype == torch.float16 else LinearTypeV2.BFLOAT16
        self.is_norm_head = is_norm
        self.first_flag = True
        self.nd_weight = nd_weight

    @classmethod
    def load(cls, prefix: str, weights, bias: bool, bias_name: str = "bias", module_name: str = ""):
        if weights.sharded:
            prefix = module_name
            bias_name = "bias"
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.{bias_name}")
        else:
            bias = None
        return cls(weight, bias)

    def get_weights(self, prefix):
        self.prefix = prefix
        weight_dict = OrderedDict()
        weight_dict[f"{prefix}.weight"] = self.weight_format_cast(self.weight.data)
        if self.bias is not None:
            weight_dict[f"{prefix}.bias"] = self.weight_format_cast(self.bias.data)
        return weight_dict
