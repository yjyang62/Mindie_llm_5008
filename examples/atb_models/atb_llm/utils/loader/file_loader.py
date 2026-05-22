# Part of this file was copied from project text-generation-inference 0.9.1
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List, Any
from pathlib import Path
from abc import abstractmethod


class BaseFileLoader:
    def __init__(self, model_weight_path: str):
        self.model_weight_path = model_weight_path
        self._handlers = {}

    @property
    def extension(self) -> str:
        raise NotImplementedError("The `extension` property is not defined in `BaseFileLoader`.")

    @abstractmethod
    def release_file_handler(self) -> None:
        raise NotImplementedError("The `release_file_handler` function is not defined in `BaseFileLoader`.")

    @abstractmethod
    def get_handler(self, filename: str) -> Any:
        raise NotImplementedError("The `get_handler` function is not defined in `BaseFileLoader`.")

    @abstractmethod
    def get_shape(self, tensor_name: str) -> Any:
        raise NotImplementedError("The `get_shape` function is not defined in `BaseFileLoader`.")

    @abstractmethod
    def get_tensor(self, tensor_name: str) -> Any:
        raise NotImplementedError("The `get_tensor` function is not defined in `BaseFileLoader`.")

    @abstractmethod
    def get_sharded(self, tensor_name: str, dim: int, chunk_id: int, num_chunk: int) -> Any:
        raise NotImplementedError("The `get_sharded` function is not defined in `BaseFileLoader`.")


def get_weight_filenames(model_weight_path: str, extension: str) -> List[Path]:
    """Get the local files"""
    if Path(model_weight_path).exists() and Path(model_weight_path).is_dir():
        local_files = list(Path(model_weight_path).glob(f"*{extension}"))
        if not local_files:
            raise FileNotFoundError(
                f"No local weights found with extension {extension};"
                f"Only safetensor format is supported. Please refer to model's README for more details."
            )
        return local_files
    raise FileNotFoundError("The input model id is not exists or not a directory")
