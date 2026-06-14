# Copyright (c) Alibaba Cloud.
# LICENSE file in https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
#
# Implement part of this file based on Qwen/Qwen-7B
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
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from atb_llm.utils.layers import (
    RMSNorm,
    RMSNormBias,
    RMSNormWrapper,
    RMSNormAntiOutlierWrapper,
)


class QwenRMSNorm(RMSNorm):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__(prefix, weights, eps)

    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states, residual


class QwenRMSNormBias(RMSNormBias):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__(prefix, weights, eps)


class QwenRMSNormWrapper(RMSNormWrapper):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__(prefix, weights, eps)


class QwenRMSNormAntiOutlierWrapper(RMSNormAntiOutlierWrapper):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__(prefix, weights, eps)