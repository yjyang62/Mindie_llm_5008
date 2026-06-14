# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from atb_llm import nn
from atb_llm.models.base.mindie_llm_config import MindIELLMConfig
from .lora_layers import ColumnParallelLinearWithLoRA, RowParallelLinearWithLoRA


_support_lora_classes = {
    ColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA
}


def from_layer(layer: nn.Module,
               mindie_llm_config: MindIELLMConfig,
               device) -> nn.Module:
    for lora_cls in _support_lora_classes:
        if lora_cls.can_replace_layer(layer):
            instance_layer = lora_cls(layer)
            instance_layer.create_lora_weights(mindie_llm_config, device)
            return instance_layer
    return layer


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module