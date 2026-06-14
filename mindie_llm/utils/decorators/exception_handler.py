# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from functools import wraps
import inspect

import torch

from mindie_llm.utils.log.error_code import ErrorCode
from mindie_llm.utils.log.logging import logger


def _torch_oom_handler(func):
    """Handler specifically for PyTorch OOM errors."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Handle PyTorch OOM (Only supports Torch >= 2.6 native exception)
            # If torch version is 2.1 or lower, please check exception message directly.
            if hasattr(torch, "OutOfMemoryError") and isinstance(e, torch.OutOfMemoryError):
                error_msg = (
                    "Device out of memory (OOM) reported by PyTorch, but it can possibly triggered by HCCL. "
                    "Enable logs: export ASCEND_SLOG_PRINT_TO_STDOUT=1, "
                    "export ASCEND_GLOBAL_LOG_LEVEL=3 to check if there's HCCL error messages"
                )
                error_code = ErrorCode.ACL_GRAPH_OUT_OF_MEMORY
                logger.error(error_msg, error_code)
                raise RuntimeError(f"{error_msg}. Error_code: {error_code}") from e
            raise

    return wrapper


def _is_target_method(name):
    """Filter methods that need OOM handling."""
    if name == "generate_position_ids":
        return False
    elif name.startswith("__"):
        return False
    return name.startswith("forward") or name.startswith("compile") or name in ["load_weights"]


def _apply_handlers(func):
    """Apply the chain of handlers to a function."""
    return _torch_oom_handler(func)


def exception_handler(cls):
    """Class decorator that applies OOM handlers to target methods."""
    for name, method in list(cls.__dict__.items()):
        if not _is_target_method(name):
            continue

        if inspect.isfunction(method):
            setattr(cls, name, _apply_handlers(method))
        elif isinstance(method, classmethod):
            setattr(cls, name, classmethod(_apply_handlers(method.__func__)))
        elif isinstance(method, staticmethod):
            setattr(cls, name, staticmethod(_apply_handlers(method.__func__)))
    return cls
