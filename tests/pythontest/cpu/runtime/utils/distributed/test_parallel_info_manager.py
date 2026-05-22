# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from unittest import mock
import pytest
from mindie_llm.runtime.utils.distributed.parallel_info_manager import (
    ParallelInfoManager,
    ParallelType,
    ParallelInfo
)


class FakeManagerForInit:
    """Fake manager that provides world_size, rank, and static method access."""
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank

    # Expose the static method so self._get_current_group_id works
    _get_current_group_id = staticmethod(ParallelInfoManager._get_current_group_id)
    _create_npu_process_group = staticmethod(ParallelInfoManager._create_npu_process_group)
    _create_cpu_process_group = staticmethod(ParallelInfoManager._create_cpu_process_group)


@pytest.fixture(autouse=True)
def mock_torch_distributed():
    """Mock torch.distributed to avoid real initialization in UT."""
    with mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.get_world_size", return_value=8), \
         mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.get_rank", return_value=3), \
         mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.new_group") as mock_new_group:
        yield mock_new_group


@pytest.fixture
def mock_server_config():
    # world_size = 8, so moe_tp * moe_ep must = 8
    return {
        "tp": 2,      # TP: 4 groups of 2
        "dp": 4,      # DP: 2 groups of 4 (strided)
        "cp": 2,      # CP: 4 groups of 2 (strided)
        "moe_tp": 2,  # MoE TP: group_size=2
        "moe_ep": 4,  # MoE EP: group_size=4 → 2*4=8
        "sp": 2       # SP: 4 groups of 2
    }


def test_get_current_group_id():
    rank_per_group = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert ParallelInfoManager._get_current_group_id(rank_per_group, 4) == 1
    assert ParallelInfoManager._get_current_group_id(rank_per_group, 0) == 0
    assert ParallelInfoManager._get_current_group_id(rank_per_group, 8) == 2
    assert ParallelInfoManager._get_current_group_id(rank_per_group, 9) is None


def test_init_tp_parallel_info_world_size(mock_torch_distributed):
    """Test TP with group_size = world_size (single group)."""
    manager = FakeManagerForInit(world_size=4, rank=2)
    info = ParallelInfoManager._init_tp_parallel_info(manager, group_size=4)

    assert info.group_size == 4
    assert info.num_group == 1
    assert info.rank_per_group == [[0, 1, 2, 3]]
    assert info.current_group_id == 0
    assert info.rank == 2
    assert info.process_group is not None


def test_init_tp_parallel_info_group_size_2(mock_torch_distributed):
    """Test TP with group_size=2, world_size=8."""
    manager = FakeManagerForInit(world_size=8, rank=5)
    info = ParallelInfoManager._init_tp_parallel_info(manager, group_size=2)

    assert info.group_size == 2
    assert info.num_group == 4
    assert info.rank_per_group == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert info.current_group_id == 2  # rank 5 is in group [4,5] → index 2
    assert info.rank == 1  # local rank within [4,5]


def test_init_tp_parallel_info_disabled(mock_torch_distributed):
    """Test TP with group_size=1 (disabled)."""
    manager = FakeManagerForInit(world_size=8, rank=3)
    info = ParallelInfoManager._init_tp_parallel_info(manager, group_size=1)

    assert info.group_size == 1
    assert info.num_group == 8
    assert info.rank_per_group == [[i] for i in range(8)]
    assert info.current_group_id == 3
    assert info.rank == 0


def test_init_dp_parallel_info_group_size_4(mock_torch_distributed):
    """Test DP with group_size=4, world_size=8 → num_group=2, strided groups."""
    manager = FakeManagerForInit(world_size=8, rank=5)
    info = ParallelInfoManager._init_dp_parallel_info(manager, group_size=4)

    # With world_size=8, group_size=4 → num_group = 8/4 = 2
    # Groups: [0,2,4,6], [1,3,5,7]
    assert info.group_size == 4
    assert info.num_group == 2
    assert info.rank_per_group == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert info.current_group_id == 1  # rank 5 in second group
    assert info.rank == 2  # [1,3,5,7] → 5 is at index 2
    # Check both CPU and NPU groups created
    assert info.cpu_process_group is not None
    assert info.process_group is not None


def test_parallel_info_manager_initialization(mock_torch_distributed, mock_server_config):
    """Test full initialization with server config."""
    llm_config = None
    local_rank = 0

    with mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.new_group"):
        manager = ParallelInfoManager(local_rank, llm_config, mock_server_config)

    # Check world
    assert manager.world.group_size == 8
    assert manager.world.num_group == 1

    # Check attn_tp (tp=2)
    tp = manager.attn_tp
    assert tp.group_size == 2
    assert tp.num_group == 4
    assert tp.rank_per_group == [[0,1],[2,3],[4,5],[6,7]]

    # Check attn_dp (dp=4 → 2 groups of 4)
    dp = manager.attn_dp
    assert dp.group_size == 4
    assert dp.num_group == 2
    assert dp.rank_per_group == [[0,2,4,6], [1,3,5,7]]

    # Check aliases
    assert manager.word_embed_tp is manager.attn_tp
    assert manager.lm_head_tp is manager.mlp_tp

    # Check buffer sizes for MoE
    assert manager.moe_tp.buffer_size == 64  # default when llm_config is None
    assert manager.moe_ep.buffer_size == 512


def test_get_method_valid_types(mock_torch_distributed, mock_server_config):
    """Test get() returns correct ParallelInfo for valid types."""
    with mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.new_group"):
        manager = ParallelInfoManager(0, None, mock_server_config)

    # Test a few
    assert manager.get(ParallelType.ATTN_TP) is manager.attn_tp
    assert manager.get(ParallelType.ATTN_DP) is manager.attn_dp
    assert manager.get(ParallelType.MOE_TP) is manager.moe_tp
    assert manager.get(ParallelType.WORLD) is manager.world


def test_get_method_invalid_type(mock_torch_distributed, mock_server_config):
    """Test get() raises KeyError for unsupported type."""
    with mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.new_group"):
        manager = ParallelInfoManager(0, None, mock_server_config)

    with pytest.raises(KeyError, match="Unsupported ParallelType"):
        manager.get("invalid_type")


def test_deprecated_has_methods(mock_torch_distributed, mock_server_config):
    """Ensure deprecated has_* methods still work (delegate to .get().is_enabled())."""
    with mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.new_group"):
        manager = ParallelInfoManager(0, None, mock_server_config)

    # tp=2 → enabled
    assert manager.has_attn_tp() is True
    # dp=4 → enabled
    assert manager.has_dp() is True
    # sp=2 → enabled
    assert manager.has_attn_inner_sp() is True
    # pp not implemented → False
    assert manager.has_pp() is False


def test_even_divide_assumption():
    """Ensure even_divide is used → group_size must divide world_size."""
    with mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.get_world_size", return_value=9), \
         mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.get_rank", return_value=0), \
         mock.patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.new_group"), \
         mock.patch("mindie_llm.runtime.utils.distributed.utils.even_divide", return_value=3):
        manager = ParallelInfoManager(0, None, {"tp": 3})
        assert manager.attn_tp.num_group == 3