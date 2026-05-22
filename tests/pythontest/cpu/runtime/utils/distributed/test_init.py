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

from mindie_llm.runtime.utils.distributed import (
    set_parallel_info_manager,
    get_parallel_info_manager,
    init_distributed,
    _PARALLEL_INFO_MANAGER
)
from mindie_llm.runtime.utils.helpers.env import ENV


def reset_manager():
    global _PARALLEL_INFO_MANAGER
    _PARALLEL_INFO_MANAGER = None


def test_set_and_get_parallel_info_manager():
    reset_manager()
    
    class FakeManager: 
        pass
    manager = FakeManager()
    set_parallel_info_manager(manager)
    assert get_parallel_info_manager() is manager


def test_init_distributed_success():
    reset_manager()
    with mock.patch.object(ENV, 'master_ip', '192.168.1.100'), \
         mock.patch.object(ENV, 'master_port', '29500'), \
         mock.patch("mindie_llm.runtime.utils.distributed.dist.init_process_group") as mock_init_pg, \
         mock.patch("mindie_llm.runtime.utils.distributed.ParallelInfoManager") as mock_pim:

        fake_manager = mock.Mock()
        fake_manager.local_rank = 3
        mock_pim.return_value = fake_manager

        llm_config = {"any": "config"}
        server_config = {"tp": 2}

        init_distributed(
            rank=1,
            world_size=8,
            local_rank=3,
            llm_config=llm_config,
            server_config=server_config
        )

        mock_init_pg.assert_called_once_with(
            backend='hccl',
            init_method='tcp://192.168.1.100:29500',
            world_size=8,
            rank=1
        )

        mock_pim.assert_called_once_with(3, llm_config, server_config)

        assert get_parallel_info_manager() is fake_manager


def test_init_distributed_missing_master_ip():
    reset_manager()
    with mock.patch.object(ENV, 'master_ip', None), \
         mock.patch.object(ENV, 'master_port', '29500'):
        with pytest.raises(ValueError, match="Master IP address is not set"):
            init_distributed(rank=0, world_size=1, local_rank=0)


def test_init_distributed_missing_master_port():
    reset_manager()
    with mock.patch.object(ENV, 'master_ip', '127.0.0.1'), \
         mock.patch.object(ENV, 'master_port', None):
        with pytest.raises(ValueError, match="Master port is not set"):
            init_distributed(rank=0, world_size=1, local_rank=0)


def test_init_distributed_missing_both():
    reset_manager()
    with mock.patch.object(ENV, 'master_ip', None), \
         mock.patch.object(ENV, 'master_port', None):
        with pytest.raises(ValueError, match="Master IP address is not set"):
            init_distributed(rank=0, world_size=1, local_rank=0)


def test_init_distributed_with_none_configs():
    reset_manager()
    with mock.patch.object(ENV, 'master_ip', '127.0.0.1'), \
         mock.patch.object(ENV, 'master_port', '12345'), \
         mock.patch("mindie_llm.runtime.utils.distributed.dist.init_process_group") as mock_init_pg, \
         mock.patch("mindie_llm.runtime.utils.distributed.ParallelInfoManager") as mock_pim:

        fake_manager = mock.Mock()
        fake_manager.local_rank = 0
        mock_pim.return_value = fake_manager

        init_distributed(rank=0, world_size=4, local_rank=0)

        mock_init_pg.assert_called_once_with(
            backend='hccl',
            init_method='tcp://127.0.0.1:12345',
            world_size=4,
            rank=0
        )

        mock_pim.assert_called_once_with(0, None, None)

        assert get_parallel_info_manager() is fake_manager