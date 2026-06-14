# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch

from ...models.base.config import BaseConfig
from ...layers.base_layer import BaseLayer
from ... import nn
from ...nn.tensor import Tensor
from ...nn.network_manager import get_default_net
from ...utils.loader.safetensor_file_loader import SafetensorFileLoader
from ...utils.loader.weight_loader import replicated_loader, check_weight_exists, is_all_zero


class RmsNorm(BaseLayer):
    def __init__(self, config: BaseConfig, file_loader: SafetensorFileLoader, prefix: str, **kwargs):
        super().__init__(config, file_loader, prefix=prefix, **kwargs)

    def create_module(self, prefix: str, **kwargs) -> nn.Module:
        self.module = nn.modules.RmsNorm(prefix, self.config.epsilon)
        # Bias used for anti-outlier
        if check_weight_exists(self.file_loader, f"{prefix}.bias") and not is_all_zero(self.file_loader, f"{prefix}.bias"):
            self.bias = nn.Parameter(prefix=prefix, suffix="bias")
            self.bias.register_processor(lambda tensor: tensor.to(self.config.torch_dtype))
        else:
            self.bias = None

    def weight_loader(self,
        parameter: nn.Parameter, prefix: str, **kwargs
    ) -> torch.Tensor:
        tensor = replicated_loader(parameter, self.file_loader, [prefix], **kwargs)
        tensor = tensor.to(self.config.torch_dtype)
        return tensor

    def forward(self, input_tensor: Tensor) -> Tensor:
        out = self.module(input_tensor)
        if self.bias is not None:
            out = out + self.bias.get_tensor()
        return out
