# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from datetime import timedelta
import json
import time

import torch

from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open
from .log import logger, print_log


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    @staticmethod
    def allreduce(*args, **kwargs):
        return FakeBarrier()

    @staticmethod
    def allgather(inputs, local_tensor, **kwargs):
        return FakeBarrier()

    @staticmethod
    def barrier(*args, **kwargs):
        return FakeBarrier()

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def get_rank_table_file():
    return ENV.rank_table_file


def set_device_from_ranktable(rank, rank_table):
    device_found_flag = False
    logger.info(f"Selecting device from rank table for rank {rank}.")
    with safe_open(rank_table, 'r', encoding='utf-8') as device_file:
        data = json.load(device_file)

        for server in data["server_list"]:
            for device in server["device"]:
                if int(device["rank_id"]) == rank:
                    device_id = int(device["device_id"])
                    device = torch.device(f"npu:{device_id}")
                    device_found_flag = True
                    break
            if device_found_flag:
                break
    if not device_found_flag:
        raise ValueError(
            f"ERROR: Rank id is not in the rankTableFile, the input rank is "
            f" {rank}"
        )
    return device


def initialize_distributed(rank, npu_id, world_size):
    if npu_id is None:
        npu_id = rank
    rank_table = ENV.rank_table_file
    if rank_table:
        device = set_device_from_ranktable(rank, rank_table)
    else:
        device = torch.device(f"npu:{npu_id}")
    logger.info(f"Current process is setting device {device} for rank {rank}.")

    # Try to set device, it will retry 12 times when it failed.
    retry_max = 12
    for i in range(retry_max):
        try:
            torch.npu.set_device(device)
        except Exception as e:
            if i == retry_max - 1:
                err_msg = f"Set device {device} for rank {rank} fails, {str(e)}"
                logger.error(err_msg)
                raise RuntimeError(err_msg) from e

            warning_msg = f"Set device {device} for rank {rank} fails." \
                f"Now wait 5 seconds to retry setting, retry times: {i + 1} / 12"
            logger.warning(warning_msg)
            time.sleep(5) # Wait 5s to retry setting
            continue
        break

    logger.info(f"Current process finished setting device {device} for rank {rank}.")
    return FakeGroup(rank, world_size), device


def initialize_torch_distributed(rank, npu_id, world_size):
    if not torch.npu.is_available():
        raise NotImplementedError("NPU is not available, please check device and torch_npu library")

    from torch_npu._C._distributed_c10d import ProcessGroupHCCL

    rank_table = ENV.rank_table_file
    if rank_table:
        device = set_device_from_ranktable(rank, rank_table)
        torch.npu.set_device(device)
        return FakeGroup(rank, world_size), device

    device = torch.device(f"npu:{npu_id}")
    torch.npu.set_device(device)

    backend = "hccl"
    options = ProcessGroupHCCL.Options()
    logger.info("ProcessGroupHCCL has been Set")

    if world_size == 1:
        return FakeGroup(rank, world_size), device
    else:
        if not torch.distributed.is_initialized():
            # Call the init process.
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=300),
                pg_options=options,
            )
            logger.info(f"rank {rank} init {torch.distributed.is_initialized()}, init_process_group has been activated")
        else:
            logger.info("torch.distributed is already initialized.")

        return torch.distributed.group.WORLD, device