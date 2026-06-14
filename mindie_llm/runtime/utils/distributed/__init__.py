# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch.distributed as dist

from mindie_llm.runtime.utils.helpers.env import ENV
from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelInfoManager

_PARALLEL_INFO_MANAGER = None


def set_parallel_info_manager(parallel_info_manager: ParallelInfoManager) -> None:
    """Sets the global parallel info manager instance.

    Args:
        parallel_info_manager (ParallelInfoManager): The parallel info manager instance to set globally.
    """
    global _PARALLEL_INFO_MANAGER
    _PARALLEL_INFO_MANAGER = parallel_info_manager


def get_parallel_info_manager() -> ParallelInfoManager:
    """Retrieves the global parallel info manager instance.

    Returns:
        ParallelInfoManager: The current parallel info manager instance,
        or None if not initialized.
    """
    return _PARALLEL_INFO_MANAGER


def init_distributed(rank: int, world_size: int, local_rank: int, llm_config=None, server_config=None) -> None:
    """Initializes the distributed training environment and parallel info manager.

    This function sets up the PyTorch distributed process group using HCCL backend
    and initializes the global parallel info manager.

    Args:
        rank (int): Global rank of the current process.
        world_size (int): Total number of processes in the distributed setup.
        local_rank (int): The rank (e.g., device index) of the current process.
        llm_config : Configuration for the LLM model. Defaults to None.
        server_config : Configuration for the serving system. Defaults to None.
    """
    if dist.is_initialized():
        return

    master_ip = ENV.master_ip
    if not master_ip:
        raise ValueError("Master IP address is not set, use export MASTER_IP=xxx.xxx.xxx.xxx to solve")
    master_port = ENV.master_port
    if not master_port:
        raise ValueError("Master port is not set, use export MASTER_PORT=xxxx to solve")
    init_method = f"tcp://{master_ip}:{master_port}"
    logger.info(f"rank: {rank}, world_size: {world_size}, init_method: {init_method}, start to init distributed")
    dist.init_process_group(backend="hccl", init_method=init_method, world_size=world_size, rank=rank)

    # initialize parallel info manager
    global _PARALLEL_INFO_MANAGER
    _PARALLEL_INFO_MANAGER = ParallelInfoManager(local_rank, llm_config, server_config)


def reset_distributed_comm_state_after_reinit(model=None) -> None:
    """Drop cached HCCL process groups and MoE comm names after device reinit."""
    parallel_info_manager = get_parallel_info_manager()
    ParallelInfoManager.clear_process_group_cache()

    if parallel_info_manager is not None:
        seen_parallel_infos = set()
        for parallel_info in getattr(parallel_info_manager, "_parallel_type_map", {}).values():
            parallel_info_id = id(parallel_info)
            if parallel_info_id in seen_parallel_infos:
                continue
            seen_parallel_infos.add(parallel_info_id)
            parallel_info._process_group = None
            parallel_info._cpu_process_group = None

    try:
        from mindie_llm.runtime.layers.fused_moe.token_dispatcher import (
            TokenDispatcherWithAll2AllV,
            TokenDispatcherWithAllGather,
            TokenDispatcherWithMC2,
        )
        from mindie_llm.runtime.utils.singleton import Singleton

        for dispatcher_cls in (TokenDispatcherWithAllGather, TokenDispatcherWithMC2, TokenDispatcherWithAll2AllV):
            Singleton._instances.pop(dispatcher_cls, None)
    except ImportError as exc:
        logger.warning(f"Skip resetting MoE dispatcher singletons after NPU reinit: {exc}")

    if model is None or not hasattr(model, "modules"):
        return

    for module in model.modules():
        if hasattr(module, "_moe_ep_group"):
            module._moe_ep_group = None
