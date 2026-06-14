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

import os
import json
from typing import List

import torch

from .safetensor_file_loader import SafetensorFileLoader
from .. import file_utils
from ... import nn
from ...layers import QuantTypeV3


def replicated_loader(parameter: nn.Parameter, file_loader: SafetensorFileLoader, prefixes: List[str], **kwargs) -> torch.Tensor:
    is_uniform = kwargs.get("is_uniform", False)
    weight_tensors = [file_loader.get_tensor(f"{p}.{parameter.suffix}") for p in prefixes]
    if len(weight_tensors) > 1:
        weight_tensors = [torch.cat(weight_tensors)]
    weight_tensor = weight_tensors[0]
    if is_uniform:
        if torch.allclose(weight_tensor, weight_tensor[0]):
            weight_tensor = weight_tensor[:1]
        else:
            raise ValueError(f"Tensors with `{prefixes}` are not equal.")
    return weight_tensor


def sharded_loader(parameter: nn.Parameter, file_loader: SafetensorFileLoader, prefixes: List[str], dim=0, **kwargs) -> torch.Tensor:
    weight_tensors = [file_loader.get_sharded(
        f"{p}.{parameter.suffix}", dim=dim,
        chunk_id=file_loader.mapping.rank, num_chunk=file_loader.mapping.world_size) for p in prefixes]
    if len(weight_tensors) > 1:
        weight_tensors = [torch.cat(weight_tensors)]
    return weight_tensors[0]


def stack_sharded_loader(parameter: nn.Parameter, file_loader: SafetensorFileLoader,
                         prefixes: List[List[str]], dim=0, **kwargs) -> torch.Tensor:
    weight_list = []
    for prefix in prefixes:
        weight_tensors = [file_loader.get_sharded(
            f"{p}.{parameter.suffix}", dim=dim,
            chunk_id=file_loader.mapping.rank, num_chunk=file_loader.mapping.world_size) for p in prefix]
        if len(weight_tensors) > 1:
            weight_tensors = [torch.cat(weight_tensors)]
        weight_list.append(weight_tensors[0])
    return torch.stack(weight_list, dim=0)


def get_linear_quant_type(model_weight_path, dtype, key) -> QuantTypeV3:
    quant_descs = None
    filename = os.path.join(model_weight_path, 'quant_model_description.json')
    if os.path.exists(filename):
        with file_utils.safe_open(filename, 'r') as f:
            quant_descs = json.load(f)

    if quant_descs is None:
        return QuantTypeV3.FLOAT16 if dtype == torch.float16 else QuantTypeV3.BFLOAT16
    if quant_descs.get(key, "INVALID") == "FLOAT":
        return QuantTypeV3.FLOAT16 if dtype == torch.float16 else QuantTypeV3.BFLOAT16
    return QuantTypeV3[quant_descs.get(key, "INVALID")]


def is_all_zero(file_loader: SafetensorFileLoader, tensor_name: str) -> bool:
    try:
        tensor = file_loader.get_tensor(tensor_name)
        return torch.all(tensor == 0)
    except ValueError:
        return False


def check_weight_exists(file_loader: SafetensorFileLoader, tensor_name: str) -> bool:
    try:
        filename, _ = file_loader.get_filename(tensor_name)
        return filename is not None
    except ValueError:
        return False
