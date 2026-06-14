# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import types
from abc import ABC, abstractmethod
from typing import Optional, List

import torch
import torch_npu
import torch.nn.functional as F

from atb_llm import nn
from atb_llm.nn.parameter import Parameter
from atb_llm.nn.functional.math import linear, grouped_matmul
from atb_llm.nn.modules.linear import TransposeType
from atb_llm.layers.linear.linear import BaseLinear, ColumnParallelLinear, RowParallelLinear
from atb_llm.models.base.mindie_llm_config import MindIELLMConfig
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader


class BaseLayerWithLoRA(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError

    @abstractmethod
    def create_lora_weights(self, mindie_llm_config: MindIELLMConfig, device):
        """Initializes lora weights stack"""
        ...

    @abstractmethod
    def load_lora_a(self, file_loader: SafetensorFileLoader, prefixes: List[str]) -> torch.Tensor:
        """Load lora a"""
        ...

    @abstractmethod
    def load_lora_b(self, file_loader: SafetensorFileLoader, prefixes: List[str], scales: List[int]) -> torch.Tensor:
        """Load lora b"""
        ...

    @abstractmethod
    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor):
        """Overwrites lora weight at index"""
        ...

    @abstractmethod
    def reset_lora(self, index: int):
        """Clear lora weight at index"""
        ...


class ParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: BaseLinear):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer_prefixes = base_layer.prefixes
        self.pack_num = len(self.base_layer_prefixes)
        self.base_weight_shape = self.get_base_weight_shape(base_layer)
        self.need_nz = False
        self.lora_a_stacked = Parameter(prefix=("_".join(self.base_layer_prefixes) + '.lora_A'), suffix="weight")
        self.lora_b_stacked = Parameter(prefix=("_".join(self.base_layer_prefixes) + '.lora_B'), suffix="weight")
        self.wrap_forward()

    @property
    def weight(self) -> torch.Tensor:
        return getattr(self.base_layer, "weight", None)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.base_layer, "bias", None)

    @staticmethod
    def get_base_weight_shape(linear_module) -> tuple:
        """Get the weight shape of a linear layer, accounting for transposition.
        Args:
            linear_module: the linear layer module
        Returns:
            Tuple of (n, k) shape
        """
        weight_shape = linear_module.weight.shape
        if len(weight_shape) <= 1:
            return 1, 1
        if linear_module.weight.trans_flag == TransposeType.TRANSPOSE:
            n, k = weight_shape
        else:
            k, n = weight_shape
        return n, k

    @staticmethod
    def weight_format_cast(tensor: torch.Tensor):
        """Transdata to Fractal_NZ format"""
        torch_npu.npu_format_cast_(tensor, 29)
        return tensor

    @staticmethod
    def get_padding_size(x: int, need_nz: bool = False):
        """Align up num to 16 or 64 according the soc version"""
        if need_nz:
            padding_size = math.ceil(x / 16) * 16 # Align up 16
        else:
            padding_size = math.ceil(x / 64) * 64 # Align up 64
        return padding_size

    @staticmethod
    def padding_tensor(padding_size, tensor):
        """Apply padding to a tensor along its first dimension"""
        padding_shape = [0, 0] * len(tensor.shape)
        padding_shape[-1] = (padding_size - tensor.shape[0] % padding_size) % padding_size
        return F.pad(tensor, padding_shape)

    def wrap_forward(self):
        ori_forward = self.base_layer.forward

        def forward_decorator(func):
            def wrapper(self_instance, *args, **kwargs):
                enable_lora = kwargs.get("enable_lora", False)
                group_list = kwargs.get("group_list", None)
                input_tensor = args[0]
                base_out = func(*args)
                if not enable_lora:
                    return base_out
                else:
                    if group_list is not None:
                        shrink = grouped_matmul(input_tensor,
                                                self.lora_a_stacked.get_tensor(),
                                                group_list=group_list,
                                                transpose_b=True)
                        expand = grouped_matmul(shrink,
                                                self.lora_b_stacked.get_tensor(),
                                                group_list=group_list,
                                                transpose_b=False)
                    else:
                        shrink = linear(input_tensor,
                                        self.lora_a_stacked.get_tensor(),
                                        transpose_b=True)
                        expand = linear(shrink,
                                        self.lora_b_stacked.get_tensor(),
                                        transpose_b=False)
                    return base_out + expand
            return wrapper
        self.base_layer.forward = types.MethodType(forward_decorator(ori_forward), self.base_layer)

    def create_lora_weights(self, mindie_llm_config: MindIELLMConfig, device):
        self.dtype = mindie_llm_config.hf_config.torch_dtype
        self.device = device
        if self.dtype not in [torch.float16, torch.bfloat16]:
            raise RuntimeError("LoRA only supports float16 and bfloat16 currently.")
        self.need_nz = mindie_llm_config.soc_info.need_nz
        max_loras = mindie_llm_config.lora_config.max_loras
        max_lora_rank = mindie_llm_config.lora_config.max_lora_rank
        padding_size = self.get_padding_size(max_lora_rank * self.pack_num, self.need_nz)
        n, k = self.base_weight_shape
        lora_a = torch.zeros(max_loras + 1, padding_size, k, dtype=self.dtype, device=self.device) # Num of LoRA, r, k
        lora_b = torch.zeros(max_loras + 1, padding_size, n, dtype=self.dtype, device=self.device) # Num of LoRA, r, n
        self.lora_a_stacked.data = self.weight_format_cast(lora_a) if self.need_nz else lora_a
        self.lora_b_stacked.data = self.weight_format_cast(lora_b) if self.need_nz else lora_b
        self.lora_b_stacked.trans_flag = TransposeType.NOT_TRANSPOSE

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor):
        dim_r = lora_a.shape[0]
        padding_size = self.get_padding_size(dim_r, self.need_nz)
        lora_a = self.padding_tensor(padding_size, lora_a).to(self.device)
        lora_b = self.padding_tensor(padding_size, lora_b).to(self.device)
        if self.need_nz:
            lora_a = self.weight_format_cast(lora_a)
            lora_b = self.weight_format_cast(lora_b)
        self.lora_a_stacked.data[index, :padding_size].copy_(lora_a)
        self.lora_b_stacked.data[index, :padding_size].copy_(lora_b)

    def reset_lora(self, index: int):
        self.lora_a_stacked.data[index].zero_()
        self.lora_b_stacked.data[index].zero_()

    def forward(self, *args, **kwargs):
        return self.base_layer(*args, **kwargs)


class ColumnParallelLinearWithLoRA(ParallelLinearWithLoRA):

    def __init__(self, base_layer: ColumnParallelLinear):
        super().__init__(base_layer)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
    ) -> bool:
        return isinstance(source_layer, ColumnParallelLinear)

    def load_lora_a(self, file_loader: SafetensorFileLoader, prefixes: List[str]) -> torch.Tensor:
        # not TP on LoRA_A weight
        weight_tensors = [file_loader.get_tensor(f"{p}.weight") for p in prefixes]
        if len(weight_tensors) > 1:
            weight_tensors = [torch.cat(weight_tensors)]
        return weight_tensors[0].to(self.dtype)

    def load_lora_b(self, file_loader: SafetensorFileLoader, prefixes: List[str], scales: List[int]) -> torch.Tensor:
        # only perform n-dim TP on LoRA_B weight
        weight_tensors = [file_loader.get_sharded(
            f"{p}.weight", dim=0, chunk_id=file_loader.mapping.rank,
            num_chunk=file_loader.mapping.world_size) for p in prefixes]
        if len(weight_tensors) == 1:
            weight_tensors = [(weight_tensors[0] * scales[0])]
        if len(weight_tensors) > 1:
            scaled_lora_weights = []
            for weight_tensor, scale in zip(weight_tensors, scales):
                scaled_lora_weights.append(weight_tensor * scale)
            weight_tensors = [torch.block_diag(*scaled_lora_weights)]
        weight_tensors = weight_tensors[0].T.contiguous()
        return weight_tensors.to(self.dtype)


class RowParallelLinearWithLoRA(ParallelLinearWithLoRA):

    def __init__(self, base_layer: RowParallelLinear):
        super().__init__(base_layer)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
    ) -> bool:
        return isinstance(source_layer, RowParallelLinear)

    def load_lora_a(self, file_loader: SafetensorFileLoader, prefixes: List[str]) -> torch.Tensor:
        # only perform n-dim TP on LoRA_A weight
        weight_tensors = [file_loader.get_sharded(
            f"{p}.weight", dim=1, chunk_id=file_loader.mapping.rank,
            num_chunk=file_loader.mapping.world_size) for p in prefixes]
        return weight_tensors[0].to(self.dtype)

    def load_lora_b(self, file_loader: SafetensorFileLoader, prefixes: List[str], scales: List[int]) -> torch.Tensor:
        # not TP on LoRA_B weight
        weight_tensors = [file_loader.get_tensor(f"{p}.weight") for p in prefixes]
        weight_tensors = weight_tensors[0] * scales[0]
        weight_tensors = weight_tensors.T.contiguous()
        return weight_tensors.to(self.dtype)