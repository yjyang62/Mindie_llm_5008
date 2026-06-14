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

import torch
import torch.distributed as torch_dist

from mindie_llm.runtime.utils.distributed.communication_op import all_gather, gather_tensor, allgather_and_reorder


def test_gather_tensor_none_index():
    """测试gather_tensor边界场景：入参index=None"""
    input_tensor = torch.randn(3, 5)
    output = gather_tensor(input_tensor, index=None)
    assert torch.equal(output, input_tensor)


def test_gather_tensor_normal():
    """测试gather_tensor正常场景：索引重排逻辑正确性"""
    input_tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    gather_index = torch.tensor([3, 0, 1, 2])
    output = gather_tensor(input_tensor, gather_index)
    expected = torch.tensor([[7, 8], [1, 2], [3, 4], [5, 6]])
    assert torch.equal(output, expected)


def test_all_gather_basic():
    """测试all_gather核心逻辑"""
    with patch.dict('sys.modules', {'torch_npu': MagicMock()}):
        mock_process_group = MagicMock()
        mock_process_group.size.return_value = 2
        with patch.object(torch_dist, "all_gather_into_tensor"):
            input_tensor = torch.tensor([[10], [20]])
            output = all_gather(input_tensor, mock_process_group)
            assert output.shape == (4, 1)
            torch_dist.all_gather_into_tensor.assert_called_once()
            mock_process_group.size.assert_called_once()


def test_allgather_and_reorder_full():
    """完整测试allgather_and_reorder组合逻辑"""
    with patch.dict('sys.modules', {'torch_npu': MagicMock()}):
        mock_process_group = MagicMock()
        mock_process_group.size.return_value = 4
        with patch.object(torch_dist, "all_gather_into_tensor"):
            input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
            gather_index = torch.tensor([3, 0, 1, 2])
            output = allgather_and_reorder(input_tensor, mock_process_group, gather_index)
            assert output.shape == (4, 3)
            torch_dist.all_gather_into_tensor.assert_called_once()
            mock_process_group.size.assert_called_once()

if __name__ == "__main__":
    test_gather_tensor_none_index()
    test_gather_tensor_normal()
    test_all_gather_basic()
    test_allgather_and_reorder_full()