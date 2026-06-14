# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import abstractmethod
from typing import List
import ctypes

import torch

from .. import nn
from ..nn.tensor import Tensor
from ..models.base.config import BaseConfig
from ..utils.loader.safetensor_file_loader import SafetensorFileLoader
from ..utils.memory_utils import check_npu_mem


class BaseLayer(nn.Module):
    def __init__(self, config: BaseConfig, file_loader: SafetensorFileLoader, **kwargs):
        super().__init__()
        self.config = config
        self.file_loader = file_loader
        self.create_module(**kwargs)
        self.load_weight(**kwargs)

    @abstractmethod
    def create_module(self, **kwargs):
        pass

    @abstractmethod
    def weight_loader(self, parameter: nn.Parameter, **kwargs) -> torch.Tensor:
        # Warning: Instead of calling this function directly, use `load_parameter` instead
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor | List[Tensor]:
        pass

    def load_weight(self, **kwargs):
        for _, parameter in self.named_parameters():
            if isinstance(parameter, nn.Parameter):
                self.load_parameter(parameter, **kwargs)

    def load_parameter(self, parameter: nn.Parameter, **kwargs):
        # Warning: Child classes should not override this method.
        weight_tensor = self.weight_loader(parameter, **kwargs)
        parameter.data = weight_tensor
        llm_config = kwargs.get("llm_config")
        if llm_config is not None:
            weights_options = llm_config.get("llm").get("weights_options")
            if weights_options is not None and weights_options.low_cpu_memory_mode:
                total_weight_size = parameter.numel() * parameter.element_size()
                check_npu_mem(rank=self.file_loader.mapping.rank, total_weight_size=total_weight_size)
                parameter.data = parameter.data.npu()
                torch.npu.synchronize()
                torch.npu.empty_cache()
                del weight_tensor
                ctypes.CDLL("libc.so.6").malloc_trim(0)
