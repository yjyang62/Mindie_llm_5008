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
import sys
from enum import Enum
from typing import Optional, List

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.node import Node
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.log import logger
from atb_llm.utils.env import ENV

path = ENV.atb_speed_home_path
sys.path.append(os.path.join(path, 'lib'))
import _libatb_torch as atb


default_process_group = '-1'
process_group_set = None
pg_ops_map = None
global_rank = None


def init_process_group(
    backend: Optional[str] = 'hccl',
    init_method: Optional[str] = "",
    world_size: int = -1,
    rank: int = -1,
    buffer_size: int = 128,
) -> None:
    '''
    Initialize the default distributed process group.

    This will also initialize the ATB comm manager.

    Args:
        backend (str, optional): Communication backend.
        init_method (str, optional): The method th initialize the communication,
                                    giving a ranktable filepath here for multi-
                                    node hccl communication. 
        world_size (int): The global worldsize.
        rank (int): The global rank id.
        buffer_size (int, optional): The hccl buffersize.
    '''
    global default_process_group
    if world_size <= 1:
        return
    if not is_initialized():
        device_rankids = list(range(world_size))
        atb.init_process_group(
            backend,
            world_size,
            rank,
            init_method
        )
        default_process_group = atb.new_group(
            device_rankids,
            rank,
            backend,
            buffer_size,
        )
        global process_group_set
        process_group_set = set((default_process_group))
        global global_rank
        global_rank = rank
        global pg_ops_map
        pg_ops_map = {default_process_group: []}
    else:
        warning_msg = "Default process group is initialized yet, " \
            f"and the default process group id is {default_process_group}"
        logger.warning(warning_msg)


def is_initialized() -> bool:
    """
    Check if the default process group has been initialized.
    """
    return atb.is_process_group_initialized()


def new_group(
    ranks: List[int],
    backend: Optional[str] = 'hccl',
    buffer_size: int = 128
):
    """
    Create a new distributed group.

    Args:
        ranks (List[int]): The global rank ids list for local communication.
        backend (str, optional): Communication backend.
        buffer_size (int, optional): The hccl buffersize.

    Returns:
        process_group (str): Process group handle.
    """
    if len(ranks) <= 1:
        return '-1'
    if is_initialized():
        _check_group_ranks(ranks)
        pg = atb.new_group(ranks, global_rank, backend, buffer_size)
        global process_group_set
        process_group_set.add(pg)
        _update_pg_ops_map(pg)
    else:
        warning_msg = "Default process group is not initialized yet, please apply `init_process_group`" \
            "before allocate non-default Process Group"
        logger.error(warning_msg)
        raise ValueError(warning_msg)
    return pg


def get_backend(group: str):
    if is_initialized():
        backend = atb.get_backend(group)
        return backend
    else:
        error_msg("Process group is not initialized.")
        logger.error(error_msg)
        raise ValueError(error_msg)


def _check_group_ranks(ranks: List[int]):
    """
    Check whether the group ranks is valid.
    """
    global global_rank
    if global_rank not in ranks:
        error_msg = f"Global_rank is not in given ranks, the gloable rank is {global_rank}, " \
            f"and communication group ranks are {ranks}. " \
            "please verify the input ranks when creating a new process group"
        logger.error(error_msg)
        raise ValueError(error_msg)


def _update_pg_ops_map(op_process_group: str, ops_name: Optional[str] = None):
    """
    Update the process groups map for reusing existing process groups.
    """
    global pg_ops_map
    if not isinstance(pg_ops_map, dict):
        error_msg = "Default process group is initialized yet, " \
            "please initialize the default process group first."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if op_process_group not in pg_ops_map.keys():
        pg_ops_map[op_process_group] = []
    if ops_name is not None:
        _check_group_reuse(op_process_group, ops_name)
        pg_ops_map[op_process_group].append(ops_name)


def _check_group_reuse(op_process_group, ops_name):
    """
    Check the process group reusing.
    """
    global pg_ops_map
    if len(pg_ops_map[op_process_group]) == 0:
        return
    # MoEDistribute is the prefix
    moe_distribute_prefix = "MoEDistribute"
    moedistribute_ops = [moe_distribute_prefix in cache_op for cache_op in pg_ops_map[op_process_group]]
    if moe_distribute_prefix in ops_name and not all(moedistribute_ops):
        error_msg = f"Process group {op_process_group} is used by non-MoEDistribute operation. " \
            "please apply another individual prcess group for MoEDistribute operation."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if moe_distribute_prefix not in ops_name and any(moedistribute_ops):
        error_msg = f"Process group {op_process_group} is used by MoEDistribute operation. " \
            "please apply another individual prcess group for non-MoEDistribute operation."
        logger.error(error_msg)
        raise ValueError(error_msg)


class AllReduceType(str, Enum):
    SUM = 'sum'
    PROD = 'prod'
    MAX = 'max'
    MIN = 'min'


def all_reduce(
    send_tensor: Tensor,
    process_group: str = None,
    all_reduce_type: AllReduceType = AllReduceType.SUM
) -> Tensor:
    '''
    Reduces the tensor data across all devices in a way that all get the final result.

    Args:
        send_tensor (Tensor): Input of communication.
        process_group (str, optional): Process group handle.
        all_reduce_type (int, optional): The computation type of all_reduce, and default is sum.

    Returns:
        recv_tensor (Tensor):  Output of communication.

    '''
    global default_process_group
    op_process_group = default_process_group if process_group is None else process_group
    if op_process_group == '-1':
        return send_tensor
    recv_tensor = Tensor()
    _update_pg_ops_map(op_process_group, 'AllReduce')
    param = {
        "allReduceType": all_reduce_type,
        "processGroup": op_process_group
    }
    node = Node('AllReduce', param, [send_tensor], [recv_tensor])
    get_default_net().push_node(node)
    return recv_tensor


def all_gather(
    send_tensor: Tensor,
    process_group: str = None,
) -> Tensor:
    '''
    Gathers tensors from the whole group in a list.

    Args:
        send_tensor (Tensor): Input of communication.
        process_group (str, optional): Process group handle.

    Returns:
        recv_tensor (Tensor):  Output of communication.

    '''
    global default_process_group
    op_process_group = default_process_group if process_group is None else process_group
    if op_process_group == '-1':
        return send_tensor
    recv_tensor = Tensor()
    _update_pg_ops_map(op_process_group, 'AllGather')
    param = {
        "processGroup": op_process_group
    }
    node = Node('AllGather', param, [send_tensor], [recv_tensor])
    get_default_net().push_node(node)
    return recv_tensor