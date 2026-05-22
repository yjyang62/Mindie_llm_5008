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

from atb_llm.nn.tensor import Tensor
from .module import Module
from ..parameter import Parameter
import atb_llm.nn.functional as F


class RmsNorm(Module):
    def __init__(self, prefix: str, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(prefix=prefix, suffix="weight")

    def __call__(self, input_: Tensor):
        return self._forward(input_)

    def _forward(self, input_: Tensor) -> Tensor:
        return F.rms_norm(input_, self.weight.get_tensor(), self.eps)
