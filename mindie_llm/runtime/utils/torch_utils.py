# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from contextlib import contextmanager
import torch


@contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """
    Temporarily set the default torch data type within a context.

    This context manager saves the current default dtype, sets it to the specified dtype,
    and restores the original dtype upon exiting the context. It is thread-safe for the
    current Python thread (but not for multiple threads sharing the same torch state).

    Args:
        dtype (torch.dtype): The new default data type to set (e.g., `torch.float16`, `torch.bfloat16`).

    Yields:
        None: Control returns to the caller inside the context block.

    Raises:
        RuntimeError: If the provided dtype is invalid.
    """
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)
