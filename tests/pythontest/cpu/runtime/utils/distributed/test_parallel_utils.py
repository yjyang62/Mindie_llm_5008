# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import tempfile
import os
from unittest import mock
import pytest
import torch

# Import the module under test
from mindie_llm.runtime.utils.distributed.utils import (
    get_device_from_ranktable,
    even_divide,
    set_device
)
from mindie_llm.runtime.utils.helpers.env import ENV


def test_even_divide_success():
    assert even_divide(12, 3) == 4
    assert even_divide(100, 1) == 100


def test_even_divide_not_divisible():
    with pytest.raises(ValueError, match="10 is not evenly divisible by 3"):
        even_divide(10, 3)


def test_even_divide_zero_denominator():
    # Python raises ZeroDivisionError for x % 0
    with pytest.raises(ZeroDivisionError):
        even_divide(5, 0)


@pytest.fixture
def valid_rank_table_data():
    return {
        "server_list": [
            {
                "device": [
                    {"rank_id": "0", "device_id": "0"},
                    {"rank_id": "1", "device_id": "1"}
                ]
            },
            {
                "device": [
                    {"rank_id": "2", "device_id": "2"},
                    {"rank_id": "3", "device_id": "3"}
                ]
            }
        ]
    }


def test_get_device_from_ranktable_found(valid_rank_table_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_rank_table_data, f)
        f.flush()
        try:
            device = get_device_from_ranktable(rank=2, rank_table=f.name)
            assert device == torch.device("npu:2")
        finally:
            os.unlink(f.name)


def test_get_device_from_ranktable_rank_not_found(valid_rank_table_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_rank_table_data, f)
        f.flush()
        try:
            with pytest.raises(ValueError, match="Rank id is not in the rankTableFile.*5"):
                get_device_from_ranktable(rank=5, rank_table=f.name)
        finally:
            os.unlink(f.name)


def test_get_device_from_ranktable_invalid_json():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        f.flush()
        try:
            with pytest.raises(json.JSONDecodeError):
                get_device_from_ranktable(rank=0, rank_table=f.name)
        finally:
            os.unlink(f.name)


def test_get_device_from_ranktable_file_not_exist():
    with pytest.raises(FileNotFoundError):
        get_device_from_ranktable(rank=0, rank_table="/non/existent/path.json")


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure ENV.rank_table_file is reset after each test."""
    original = ENV.rank_table_file
    yield
    ENV.rank_table_file = original


@mock.patch("torch.npu.set_device")
def test_set_device_without_rank_table(mock_set_device):
    """Test set_device when rank_table_file is NOT set."""
    ENV.rank_table_file = None
    device = set_device(rank=3, npu_id=5)
    assert device == torch.device("npu:5")
    mock_set_device.assert_called_once_with(device)


@mock.patch("torch.npu.set_device")
def test_set_device_default_npu_id(mock_set_device):
    """Test npu_id defaults to rank when not provided."""
    ENV.rank_table_file = None
    device = set_device(rank=7)
    assert device == torch.device("npu:7")
    mock_set_device.assert_called_once_with(device)


@mock.patch("torch.npu.set_device")
def test_set_device_with_rank_table(mock_set_device, valid_rank_table_data):
    """Test set_device uses rank table when ENV.rank_table_file is set."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_rank_table_data, f)
        f.flush()
        try:
            ENV.rank_table_file = f.name
            # npu_id should be ignored
            device = set_device(rank=1, npu_id=99)
            assert device == torch.device("npu:1")
            mock_set_device.assert_called_once_with(device)
        finally:
            ENV.rank_table_file = None


@mock.patch("torch.npu.set_device")
def test_set_device_rank_not_in_table(mock_set_device):
    """Test set_device raises error if rank not in rank table."""
    rank_table = {"server_list": [{"device": [{"rank_id": "0", "device_id": "0"}]}]}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(rank_table, f)
        f.flush()
        try:
            ENV.rank_table_file = f.name
            with pytest.raises(ValueError, match="Rank id is not in the rankTableFile.*5"):
                set_device(rank=5)
        finally:
            ENV.rank_table_file = None