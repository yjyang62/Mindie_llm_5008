# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional, Dict, Type

from mindie_llm.runtime.layers.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
    TokenDispatcherWithAll2AllV,
)
from mindie_llm.runtime.layers.fused_moe.moe_comm_strategy import (
    MOE_COMM_STRATEGIES,
    MoECommType,
)
from mindie_llm.utils.log.logging import logger


# Communication type to dispatcher class mapping
_COMM_TYPE_TO_DISPATCHER: Dict[MoECommType, Type[MoETokenDispatcher]] = {
    MoECommType.ALLGATHER: TokenDispatcherWithAllGather,
    MoECommType.MC2: TokenDispatcherWithMC2,
    MoECommType.ALLTOALL: TokenDispatcherWithAll2AllV,
}


def select_moe_comm_method(
    num_experts_per_ep_rank: int,
) -> Optional[MoECommType]:
    """Select optimal MoE communication method.

    Args:
        num_experts_per_ep_rank: num of routed experts per ep rank.

    Returns:
        Selected MoECommType.

    Raises:
        RuntimeError: No applicable strategy found.
    """
    # Strategy traversal: return first applicable strategy
    for strategy_cls in MOE_COMM_STRATEGIES:
        if strategy_cls.is_applicable(
            num_experts_per_ep_rank=num_experts_per_ep_rank,
        ):
            selected = strategy_cls.get_comm_type()
            return selected
    error_msg = "MoE strategy selection failed, please check the config."
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def get_cached_dispatcher(
    moe_comm_type: Optional[MoECommType],
) -> Optional[MoETokenDispatcher]:
    """Get dispatcher instance for given communication type.

    Args:
        moe_comm_type: Communication type enum.

    Returns:
        Dispatcher instance or None if type unsupported.
    """
    if moe_comm_type is None or moe_comm_type not in _COMM_TYPE_TO_DISPATCHER:
        return None
    dispatcher_cls = _COMM_TYPE_TO_DISPATCHER[moe_comm_type]
    return dispatcher_cls()
