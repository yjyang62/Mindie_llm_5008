# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List

import torch
import numpy as np

from .linear_method import LINEAR_METHOD_ROUTER
from ...models.base.config import BaseConfig
from ...layers import InferenceMode, QuantTypeV3
from ...layers.base_layer import BaseLayer
from ... import nn
from ...nn.tensor import Tensor
from ...utils.loader.safetensor_file_loader import SafetensorFileLoader
from ...utils.loader.weight_loader import replicated_loader, sharded_loader, get_linear_quant_type, check_weight_exists, is_all_zero
from ...utils.initial import NPUSocInfo
from ...utils.log import message_filter


BIAS = "bias"


class BaseLinear(BaseLayer):
    def __init__(self,
        config: BaseConfig, file_loader: SafetensorFileLoader, prefixes: List[str], bias=False, **kwargs
    ):
        # initialize linear module
        self.prefixes = prefixes
        linear_quant_types = []
        for prefix in prefixes:
            linear_quant_types.append(
                get_linear_quant_type(file_loader.model_weight_path, config.torch_dtype, f"{prefix}.weight"))
        if len(set(linear_quant_types) - {QuantTypeV3.INVALID, }) != 1:
            raise ValueError("Weights have different quant type, so they cannot be packed together for calculation.")
        self.linear_quant_type = list(set(linear_quant_types))[0]

        # bias exist either origin model has bias or mix per tensor has bias
        bias = bias or (check_weight_exists(file_loader, f"{prefixes[0]}.bias") and not is_all_zero(file_loader, f"{prefixes[0]}.bias"))

        self._quant_method = {}
        linear_method_cls = LINEAR_METHOD_ROUTER.get(self.linear_quant_type)
        if self.linear_quant_type == QuantTypeV3.W8A8_MIX:
            self._quant_method[InferenceMode.PREFILL] = LINEAR_METHOD_ROUTER.get(QuantTypeV3.W8A8_DYNAMIC)()
            self._quant_method[InferenceMode.DECODE] = LINEAR_METHOD_ROUTER.get(QuantTypeV3.W8A8)()
        else:
            if linear_method_cls is None:
                error_msg = f"Quant type {self.linear_quant_type} doesn't match any existing implementation."
                raise ValueError(message_filter(error_msg))
            self._quant_method[InferenceMode.ANY] = linear_method_cls()

        super().__init__(config, file_loader, prefixes=prefixes, bias=bias, **kwargs)

    def create_module(self, prefixes: List[str], bias=False, **kwargs):
        prefix = "_".join(prefixes)
        for quant_method in self._quant_method.values():
            quant_method.create_weights(self, prefix, bias)

    def load_weight(self, **kwargs):
        if self.linear_quant_type != QuantTypeV3.W8A8_MIX:
            super().load_weight(**kwargs)
            return

        cache = {}
        for name, param in self.named_parameters():
            if param.name not in cache:
                self.load_parameter(param, **kwargs)
                cache[param.name] = param
            else:
                self.register_parameter(name, cache[param.name])

    def forward(self, input_tensor: Tensor, inference_mode=InferenceMode.ANY, **kwargs) -> Tensor:
        """
        Args:
            kwargs:
                enable_lora: exists when LoRA is enabled
                group_list: exists when multi-LoRA is activated
        """
        return self._quant_method.get(inference_mode, self._quant_method.get(InferenceMode.ANY)).apply(self, input_tensor)

    def _process_deq_scale(self, deq_scale: torch.Tensor):
        if NPUSocInfo().soc_version in (100, 101, 102, 103, 104):
            deq_scale = self._transform_deqscale_dtype_to_float(deq_scale)
        elif self.config.torch_dtype == torch.float16 and deq_scale.dtype != torch.int64:
            deq_scale = torch.from_numpy(
                np.frombuffer(deq_scale.to(torch.float32).numpy().tobytes(), dtype=np.int32).astype(np.int64))
        return deq_scale

    def _transform_deqscale_dtype_to_float(self, deq_scale_int64):
        deq_scale_cpu_int64 = deq_scale_int64.cpu().numpy()
        deq_scale_int32 = np.uint32(deq_scale_cpu_int64)
        original_deq_scale = np.frombuffer(deq_scale_int32.tobytes(), dtype=np.float32).copy()
        tmp_deq_scale = torch.from_numpy(original_deq_scale)
        return tmp_deq_scale.to(deq_scale_int64.device)


class RowParallelLinear(BaseLinear):
    def weight_loader(self,
        parameter: nn.Parameter, prefixes: List[str], **kwargs
    ) -> torch.Tensor:
        if "deq_scale" in parameter.name:
            tmp_tensor = replicated_loader(parameter, self.file_loader, prefixes, **kwargs)
            return self._process_deq_scale(tmp_tensor)

        if "weight_scale" in parameter.name:
            return replicated_loader(parameter, self.file_loader, prefixes, **kwargs)

        if "quant_bias" in parameter.name:
            tmp_tensor = replicated_loader(parameter, self.file_loader, prefixes, **kwargs)
            if self.file_loader.mapping.rank == 0:
                return tmp_tensor
            else:
                return torch.zeros_like(tmp_tensor, dtype=tmp_tensor.dtype, device=tmp_tensor.device)

        if "input_scale" in parameter.name or "input_offset" in parameter.name:
            tensor = replicated_loader(parameter, self.file_loader, prefixes, is_uniform=True, **kwargs)
            if "input_offset" in parameter.name:
                tensor = tensor.to(torch.int8)
            return tensor

        return sharded_loader(parameter, self.file_loader, prefixes, dim=1, **kwargs)


class ColumnParallelLinear(BaseLinear):
    def weight_loader(self,
        parameter: nn.Parameter, prefixes: List[str], **kwargs
    ) -> None:
        if "input_scale" in parameter.name or "input_offset" in parameter.name:
            tensor = replicated_loader(parameter, self.file_loader, prefixes, is_uniform=True, **kwargs)
            if "input_offset" in parameter.name:
                tensor = tensor.to(torch.int8)
            return tensor

        return sharded_loader(parameter, self.file_loader, prefixes, dim=0, **kwargs)


class ReplicatedLinear(BaseLinear):
    def weight_loader(self,
        parameter: nn.Parameter, prefixes: List[str], **kwargs
    ) -> None:
        return replicated_loader(parameter, self.file_loader, prefixes, **kwargs)


class MergedColumnParallelLinear(nn.ModuleList):
    def __init__(self,
        config: BaseConfig, file_loader: SafetensorFileLoader, prefixes: List[str], bias=False, **kwargs
    ):
        linear_quant_types = []
        for prefix in prefixes:
            linear_quant_types.append(
                get_linear_quant_type(file_loader.model_weight_path, config.torch_dtype, f"{prefix}.weight"))
        self.linear_quant_types = linear_quant_types

        if len(set(self.linear_quant_types) - {QuantTypeV3.INVALID, }) == 1:
            module = [ColumnParallelLinear(config, file_loader, prefixes, bias, **kwargs)]
        else:
            linear_module_list = []
            for prefix in prefixes:
                linear_module_list.append(ColumnParallelLinear(config, file_loader, [prefix], bias, **kwargs))
            module = linear_module_list
        
        super().__init__(module)
        self.module = module

    def forward(self, input_tensor: Tensor, inference_mode=InferenceMode.ANY, **kwargs):
        """
        Args:
            kwargs:
                enable_lora: exists when LoRA is enabled
                group_list: exists when multi-LoRA is activated
        """
        outputs = []
        for linear_module in self.module:
            outputs.append(linear_module(input_tensor, inference_mode, **kwargs))
        return outputs
