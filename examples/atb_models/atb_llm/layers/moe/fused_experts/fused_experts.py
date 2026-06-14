# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List, Tuple

import torch

from atb_llm import nn
from atb_llm.layers import QuantTypeV3
from atb_llm.layers.base_layer import BaseLayer
from atb_llm.layers.moe.fused_experts.fused_experts_method import FUSED_EXPERTS_METHOD_ROUTER
from atb_llm.models.base.config import BaseConfig
from atb_llm.nn import Module
from atb_llm.nn.functional import ActType
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.utils.loader.weight_loader import get_linear_quant_type, stack_sharded_loader


PREFIXES = "prefixes"
DIM = "dim"


class FusedExperts(BaseLayer):
    def __init__(self,
        config: BaseConfig, file_loader: SafetensorFileLoader, prefixes: List[str], bias=False, **kwargs
    ):
        self.get_weight_prefixes(config, file_loader, prefixes, **kwargs)
        self.weight_configs = {
            "gate_up_weight": {PREFIXES: self.gate_up_prefixes, DIM: 0},
            "gate_up_bias": {PREFIXES: self.gate_up_prefixes, DIM: 0},
            "down_weight": {PREFIXES: self.down_prefixes, DIM: 1},
            "down_bias": {PREFIXES: self.down_prefixes, DIM: 1},
        }
        quant_types = []
        for prefix in self.gate_up_prefixes[0]:
            quant_types.append(
                get_linear_quant_type(file_loader.model_weight_path, config.torch_dtype, f"{prefix}.weight"))
        if len(set(quant_types) - {QuantTypeV3.INVALID, }) != 1:
            raise ValueError("Weights have different quant type, so they cannot be packed together for calculation.")
        self.quant_types = quant_types

        super().__init__(config, file_loader, prefixes=prefixes, bias=bias, **kwargs)

    def create_module(self, prefixes: List[str], bias=False, **kwargs):
        self.module = FusedExpertsModule(
            "_".join(self.gate_up_prefixes[0]), self.config.torch_dtype, bias=bias,
            quant_type=list(set(self.quant_types))[0], **kwargs
        )

    def load_weight(self, **kwargs):
        for name, parameter in self.module.named_parameters():
            if isinstance(parameter, nn.Parameter):
                self.weight_loader(
                    parameter, self.weight_configs.get(name).get(PREFIXES),
                    self.weight_configs.get(name).get(DIM)
                )

    def weight_loader(self,
        parameter: nn.Parameter, prefixes: List[str], dim: int = 0, **kwargs
    ) -> torch.Tensor:
        return stack_sharded_loader(parameter, self.file_loader, prefixes, dim=dim, **kwargs)

    def get_weight_prefixes(self,
        config: BaseConfig, file_loader: SafetensorFileLoader, prefixes: List[str], **kwargs
    ) -> Tuple[List[List[str]], List[List[str]]]:
        routing_expert_prefix = prefixes[0]
        rank = file_loader.mapping.mlp_tp.rank
        world_size = file_loader.mapping.mlp_tp.group_size
        expert_lists = [[i for i in range(config.n_routed_experts)] for _ in range(world_size)]
        self.gate_up_prefixes = [[f"{routing_expert_prefix}.{i}.gate_proj", f"{routing_expert_prefix}.{i}.up_proj"] \
                            for i in expert_lists[rank]]
        self.down_prefixes = [[f"{routing_expert_prefix}.{i}.down_proj"] for i in expert_lists[rank]]


class FusedExpertsModule(Module):
    def __init__(self, prefix: str, dtype: torch.dtype, bias=False, quant_type: QuantTypeV3 = None, **kwargs):
        super().__init__()
        if quant_type is None and dtype == torch.float16:
            quant_type = QuantTypeV3.FLOAT16
        if quant_type is None and dtype == torch.bfloat16:
            quant_type = QuantTypeV3.BFLOAT16
        fused_experts_method_cls = FUSED_EXPERTS_METHOD_ROUTER.get(quant_type)
        if fused_experts_method_cls is None:
            raise ValueError("Quant type `quant_type` doesn't match any existing implementation.")
        self._quant_method = fused_experts_method_cls()
        self._quant_method.create_weights(self, prefix, dtype, bias)
        self.act_type = kwargs.get("act_type", ActType.SWIGLU)

    def __call__(
            self,
            sorted_hidden_states: Tensor,
            group_list: Tensor
        ) -> Tensor:
        return self._forward(sorted_hidden_states, group_list)

    def _forward(
            self,
            sorted_hidden_states: Tensor,
            group_list: Tensor
        ) -> Tensor:
        return self._quant_method.apply(self, sorted_hidden_states, group_list)
