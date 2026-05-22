# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# Implement part of this file based on vllm-project/vllm
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import ABC, abstractmethod

import torch
from torch import nn


class QuantizationMethodBase(ABC):
    """Base class for quantized methods."""

    INPUT_DIM: str = "input_dim"
    OUTPUT_DIM: str = "output_dim"
    BIAS: str = "bias"

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs) -> None:
        """Create layer weights."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply layer weights to input tensor."""
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process weights after loading, e.g. transpose weights for computation."""
        return
