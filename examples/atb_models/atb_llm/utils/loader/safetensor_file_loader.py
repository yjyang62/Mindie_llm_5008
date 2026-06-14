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
# Part of this file was copied from project text-generation-inference 0.9.1

from typing import Tuple, Any

from safetensors import safe_open

from .. import file_utils
from .file_loader import get_weight_filenames, BaseFileLoader


class SafetensorFileLoader(BaseFileLoader):
    def __init__(self, model_weight_path: str, device: Any, mapping=None):
        super().__init__(model_weight_path)
        self._filenames = get_weight_filenames(self.model_weight_path, self.extension)
        self._routing = self._load_weight_file_routing()
        self.mapping = mapping
        self.device = device

    @property
    def extension(self) -> str:
        return ".safetensors"

    def release_file_handler(self) -> None:
        if self._handlers:
            del self._handlers
            self._handlers = {}

    def get_handler(self, filename: str) -> Any:
        if filename not in self._handlers:
            f = safe_open(filename, framework="pytorch")
            self._handlers[filename] = f
            return f
        return self._handlers[filename]

    def get_shape(self, tensor_name: str) -> Any:
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str) -> Any:
        filename, tensor_name = self.get_filename(tensor_name)
        f = self.get_handler(filename)
        tensor = f.get_tensor(tensor_name)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int, chunk_id: int, num_chunk: int) -> Any:
        if dim not in [0, 1]:
            raise AssertionError(f"Dimension {dim} is invalid in `get_sharded`.")
        slice_ = self._get_slice(tensor_name)
        group_size = slice_.get_shape()[dim]
        if group_size < num_chunk:
            raise ValueError(
                f"The size of the tensor (i.e. {group_size}) intended for parallel processing is smaller than "
                f"the number of groups (i.e. {num_chunk}) designated for parallelism.")

        block_size = group_size // num_chunk
        start = chunk_id * block_size
        stop = (chunk_id + 1) * block_size

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        return tensor

    def get_filename(self, tensor_name: str) -> Tuple[str, str]:
        filename = self._routing.get(tensor_name)
        if filename is None:
            raise ValueError(f"Weight file was not found for tensor named with {tensor_name}.")
        return str(filename), tensor_name

    def _load_weight_file_routing(self) -> dict:
        routing = {}
        for filename in self._filenames:
            filename = file_utils.standardize_path(str(filename), check_link=False)
            file_utils.check_path_permission(filename)
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        raise AssertionError("Weight was found in multiple files.")
                    routing[k] = filename
        return routing

    def _get_slice(self, tensor_name: str) -> Any:
        filename, tensor_name = self.get_filename(tensor_name)
        f = self.get_handler(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_
