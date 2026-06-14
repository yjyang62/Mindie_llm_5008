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

from typing import List, Callable
from functools import partial

import torch
from torch import nn
import torch_npu

from .modules.linear import TransposeType
from .tensor import Format, Tensor
from ..nn.network_manager import get_default_net
from ..utils.env import ENV
from ..utils.initial import NPUSocInfo
from ..utils.log import print_log, logger


class BaseParameter(nn.Parameter):
    def __new__(cls, **kwargs):
        return super().__new__(cls, data=None, requires_grad=False)

    def __init__(self):
        super().__init__()
        self._processor: Callable = None
        self.data: torch.Tensor = torch.tensor([])

    def __setattr__(self, name: str, value) -> None:
        if name == "data" and value.device == torch.device("cpu"):
            if self._processor is not None:
                value = self._processor(value)
        super().__setattr__(name, value)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def get_tensor(self) -> Tensor:
        if self.name not in get_default_net().weights_keys:
            get_default_net().push_weight_key(self.name)
        return Tensor(self.name)

    def register_processor(self, data_processor: Callable) -> None:
        self._processor = data_processor


class Parameter(BaseParameter):
    soc_info = None

    def __init__(self, prefix: str, suffix: str, enable_auto_transpose=False, enable_nd_nz=False, **kwargs):
        super().__init__()
        self.prefix = prefix
        self.suffix = suffix
        self.enable_nd_nz = enable_nd_nz
        self.format_ = Format.ND
        self.enable_auto_transpose = enable_auto_transpose
        if self.enable_auto_transpose:
            self._processor = self._transpose_weight_as_need
        self.trans_flag = TransposeType.TRANSPOSE
        if not self.soc_info:
            self.set_soc_info()

    @property
    def name(self) -> str:
        return f"{self.prefix}.{self.suffix}"

    @classmethod
    def set_soc_info(cls) -> None:
        cls.soc_info = NPUSocInfo()

    def register_processor(self, data_processor: Callable) -> None:
        self._processor = partial(self._apply_functions, functions=(self._transpose_weight_as_need, data_processor))

    def weight_format_cast(self, enable_nd_nz: bool) -> None:
        if enable_nd_nz and self.enable_nd_nz:
            tensor = self.data
            self.format_ = Format.NZ
            torch_npu.npu_format_cast_(tensor, 29)
            self.data = tensor
            print_log(ENV.rank, logger.debug, f"Tensor trans to {torch_npu.get_npu_format(self.data)}")

    def _check_transpose(self, weight: torch.Tensor) -> TransposeType:
        if len(weight.shape) == 1:
            return TransposeType.INVALID

        if self.soc_info.need_nz or not self.enable_auto_transpose:
            return TransposeType.TRANSPOSE
        
        if not ENV.auto_transpose_enable:
            if self.soc_info.matmul_nd_nz:
                logger.warning("NZ weight format is enabled. To ensure hardware compatibility, "
                               "weights must be transposed to [k, n]. The environment variable "
                               "`ATB_LLM_ENABLE_AUTO_TRANSPOSE=0` is being ignored.")
                ENV.auto_transpose_enable = True
                return TransposeType.NOT_TRANSPOSE
            return TransposeType.TRANSPOSE
        
        if self.soc_info.matmul_nd_nz:
            # transpose weights to [k, n] when using nz format
            return TransposeType.NOT_TRANSPOSE

        is_k_divisible = weight.shape[1] % 256 == 0
        is_n_divisible = weight.shape[0] % 256 == 0
        if not is_k_divisible and is_n_divisible:
            return TransposeType.NOT_TRANSPOSE
        return TransposeType.TRANSPOSE

    def _transpose_weight_as_need(self, tensor: torch.Tensor) -> torch.Tensor:
        trans_flag = self._check_transpose(tensor)
        if trans_flag != TransposeType.INVALID and self.trans_flag != trans_flag:
            if trans_flag != TransposeType.TRANSPOSE:
                self.trans_flag = trans_flag
            if trans_flag == TransposeType.NOT_TRANSPOSE:
                return torch.transpose(tensor, 0, 1).contiguous()
        return tensor

    def _apply_functions(self, tensor: torch.Tensor, functions: List[Callable]):
        for func in functions:
            tensor = func(tensor)
        return tensor
