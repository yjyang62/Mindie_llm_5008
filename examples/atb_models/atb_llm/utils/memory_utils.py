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

import time
import torch
from ..utils.log import logger


def check_npu_mem(rank: int, total_weight_size: int, retry_times=24) -> None:
        
    # First check remaining memory on npu, then load weights
    is_mem_enough = False
    for i in range(retry_times):
        free_mem, _ = torch.npu.mem_get_info()
        if free_mem < total_weight_size:
            warning_msg = f"Remaining memory of rank-{rank} NPU: {free_mem} Byte " \
                f"is not enough to load weights: {total_weight_size} Byte. " \
                f"Now wait 10 seconds to retry loading, retry times: {i + 1} / {retry_times}"
            logger.warning(warning_msg)
            time.sleep(10)
        else:
            is_mem_enough = True
            break

    if not is_mem_enough:
        err_msg = f"Remaining memory of rank-{rank} NPU is not enough to load weights: " \
                f"{total_weight_size} Byte. " \
                f"Please make sure that your NPU has enough memory."
        logger.error(err_msg)
        raise RuntimeError(err_msg)