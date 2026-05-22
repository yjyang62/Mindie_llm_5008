# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
    MoECommType,
    select_moe_comm_method,
    get_cached_dispatcher,
)
from mindie_llm.runtime.layers.fused_moe.token_dispatcher import (
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
    TokenDispatcherWithAll2AllV,
)
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.utils.npu.device_utils import DeviceType


class MockParallelInfo:
    def __init__(self, group_size: int):
        self.group_size = group_size
        self.rank = 0
        self.process_group = MagicMock()

    def is_enabled(self) -> bool:
        return self.group_size > 1


class MockParallelInfoManager:
    def __init__(self, *, world_size: int = 8, moe_ep: int = 1, attn_dp: int = 1, moe_tp: int = 1):
        self.world_size = world_size
        self.moe_ep = MockParallelInfo(moe_ep)
        self.attn_dp = MockParallelInfo(attn_dp)
        self.moe_tp = MockParallelInfo(moe_tp)
        self.moe_ep_mc2 = MockParallelInfo(moe_ep)

    def get(self, parallel_type):
        return {
            ParallelType.MOE_EP: self.moe_ep,
            ParallelType.ATTN_DP: self.attn_dp,
            ParallelType.MOE_TP: self.moe_tp,
        }.get(parallel_type, None)


@pytest.fixture
def mock_platform_and_parallel():
    """
    封装平台信息和并行信息的mock配置
    """
    with patch("mindie_llm.runtime.layers.fused_moe.moe_comm_method.get_npu_node_info") as mock_npu_node, \
            patch("mindie_llm.runtime.layers.fused_moe.moe_comm_method.get_parallel_info_manager") as mock_parallel, \
            patch("mindie_llm.runtime.layers.fused_moe.moe_comm_method.get_forward_context") as mock_forward_ctx:
        # 默认配置: 910B + decode阶段(is_prefill=False)
        mock_npu_node.return_value.get_device_type.return_value = DeviceType.ASCEND_910B
        mock_forward_ctx.return_value.is_prefill = False

        yield mock_npu_node, mock_parallel, mock_forward_ctx


@pytest.fixture
def mock_dist_env():
    """
    封装分布式环境的mock配置
    """
    mock_dist = MagicMock()
    mock_dist.get_rank = MagicMock(return_value=0)
    mock_dist.is_initialized = MagicMock(return_value=True)

    with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.get_parallel_info_manager") as mock_parallel, \
            patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.dist", mock_dist):
        mock_parallel.return_value = MockParallelInfoManager()
        yield


@dataclass(frozen=True)
class MoeCommTestCase:
    device_type: DeviceType
    moe_ep: int
    world_size: int
    attn_dp: int
    moe_tp: int
    quant_type: Optional[str]
    is_prefill: bool
    expected_comm: MoECommType


# ====================== 参数化的 select_moe_comm 测试 ======================
@pytest.mark.parametrize(
    "case",
    [
        # 基础场景
        MoeCommTestCase(DeviceType.ASCEND_910B, 1, 8, 1, 1, None, False, MoECommType.ALLGATHER),  # 无EP，默认ALLGATHER
        # 910B+EP开启 场景
        MoeCommTestCase(DeviceType.ASCEND_910B, 16, 16, 1, 1, None, False, MoECommType.MC2),
        MoeCommTestCase(DeviceType.ASCEND_910B, 8, 8, 1, 1, None, False, MoECommType.ALLGATHER),
        MoeCommTestCase(DeviceType.ASCEND_910B, 16, 16, 1, 1, None, True, MoECommType.ALLGATHER),
        MoeCommTestCase(DeviceType.ASCEND_910B, 16, 8, 1, 1, "w4a8_dynamic", False, MoECommType.ALLTOALL),
        # 910_93 场景
        MoeCommTestCase(DeviceType.ASCEND_910_93, 16, 8, 1, 1, None, False, MoECommType.MC2),
        MoeCommTestCase(DeviceType.ASCEND_910_93, 16, 8, 1, 1, None, True, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_parametrized(
        mock_platform_and_parallel,
        case: MoeCommTestCase
):
    """
    参数化测试 select_moe_comm_method 函数，整合所有场景
    """
    mock_npu_node, mock_parallel, mock_forward_ctx = mock_platform_and_parallel

    # 设置设备类型
    mock_npu_node.return_value.get_device_type.return_value = case.device_type
    # 设置并行信息
    mock_parallel.return_value = MockParallelInfoManager(
        moe_ep=case.moe_ep,
        world_size=case.world_size,
        attn_dp=case.attn_dp,
        moe_tp=case.moe_tp
    )
    mock_forward_ctx.return_value.is_prefill = case.is_prefill

    comm_type = select_moe_comm_method(quant_type=case.quant_type)

    assert comm_type == case.expected_comm


def test_select_moe_comm_unsupported_device(mock_platform_and_parallel):
    """测试：传入不支持的设备类型，抛出ValueError异常场景"""
    mock_npu_node, mock_parallel, _ = mock_platform_and_parallel
    mock_npu_node.return_value.get_device_type.return_value = "ASCEND_UNKNOWN"

    with pytest.raises(ValueError) as exc_info:
        select_moe_comm_method()
    assert "Unsupported soc_version" in str(exc_info.value)


def test_get_cached_dispatcher_valid_type(mock_dist_env):
    """测试：传入合法的MoECommTYpe，返回对应实例"""
    assert isinstance(get_cached_dispatcher(MoECommType.ALLGATHER), TokenDispatcherWithAllGather)
    assert isinstance(get_cached_dispatcher(MoECommType.MC2), TokenDispatcherWithMC2)
    assert isinstance(get_cached_dispatcher(MoECommType.ALLTOALL), TokenDispatcherWithAll2AllV)


def test_get_cached_dispatcher_singleton(mock_dist_env):
    """测试dispatcher是单例模式"""
    dispatcher1 = get_cached_dispatcher(MoECommType.ALLGATHER)
    dispatcher2 = get_cached_dispatcher(MoECommType.ALLGATHER)
    dispatcher3 = get_cached_dispatcher(MoECommType.MC2)

    assert dispatcher1 is dispatcher2
    assert dispatcher1 is not dispatcher3


def test_get_cached_dispatcher_invalid_none():
    """测试：传入None/不支持的类型，返回None"""
    assert get_cached_dispatcher(None) is None
    assert get_cached_dispatcher(MoECommType.FUSED_ALLTOALL) is None
    assert get_cached_dispatcher("INVALID_TYPE") is None