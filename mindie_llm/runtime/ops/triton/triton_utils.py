# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any

import torch
from mindie_llm.utils.log.logging import logger

_HAS_TRITON = None
_NUM_AICORE = -1
_NUM_VECTORCORE = -1


def is_triton_available() -> bool:
    global _HAS_TRITON
    if _HAS_TRITON is None:
        try:
            import triton  # noqa
            import triton.language as tl  # noqa

            _HAS_TRITON = True
            logger.info("Detected Triton installation. Triton-based operators are available.")
        except ImportError:
            _HAS_TRITON = False
            logger.info("Triton not found. Falling back to non-Triton implementations.")
    return _HAS_TRITON


def init_device_properties_triton():
    global _NUM_AICORE, _NUM_VECTORCORE
    if _NUM_AICORE == -1 and is_triton_available():
        import triton

        device_properties: dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        _NUM_AICORE = device_properties.get("num_aicore", -1)
        _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        assert _NUM_AICORE > 0 and _NUM_VECTORCORE > 0, "Failed to detect device properties."


def get_aicore_num():
    global _NUM_AICORE
    assert _NUM_AICORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_AICORE


def get_vectorcore_num():
    global _NUM_VECTORCORE
    assert _NUM_VECTORCORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_VECTORCORE
