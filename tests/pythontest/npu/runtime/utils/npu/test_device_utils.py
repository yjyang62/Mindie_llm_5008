# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR
# FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import patch, MagicMock

import torch

from mindie_llm.runtime.utils.npu.device_utils import (
    _DeviceInfo,
    _NPUNodeInfo,
    _NPUHbmInfo,
    get_npu_node_info,
    get_npu_hbm_info,
    DeviceType,
    Topo,
)
from mindie_llm.runtime.utils.helpers.env import ENV


# Mock torch.npu if it doesn't exist
if not hasattr(torch, 'npu'):
    torch.npu = MagicMock()
    # Set a default return value for get_device_properties
    torch.npu.get_device_properties = MagicMock(
        return_value=MagicMock(name="Ascend910B")
    )


class TestDeviceInfo(unittest.TestCase):
    """Test _DeviceInfo parsing."""

    def test_device_info_valid_line(self):
        """Test parsing a valid npu-smi info -m line."""
        line = "0 1 0 Ascend910B"
        device = _DeviceInfo(line)
        self.assertEqual(device.npu_id, 0)
        self.assertEqual(device.chip_id, 1)
        self.assertEqual(device.chip_logic_id, 0)
        self.assertEqual(device.chip_name, "Ascend910B")

    def test_device_info_non_numeric_logic_id(self):
        """Test parsing line with non-numeric chip_logic_id (e.g., 'N/A')."""
        line = "0 1 N/A Ascend310P"
        device = _DeviceInfo(line)
        self.assertEqual(device.chip_logic_id, "N/A")

    def test_device_info_invalid_line(self):
        """Test parsing invalid line raises ValueError."""
        with self.assertRaises(ValueError):
            _DeviceInfo("invalid line")


class TestNPUNodeInfo(unittest.TestCase):
    """Test _NPUNodeInfo singleton and methods."""

    def setUp(self):
        """Reset ENV and clear singleton instance."""
        ENV.visible_devices = None
        # Clear singleton to ensure clean state
        _NPUNodeInfo._instance = None
        _NPUNodeInfo._initialized = False
        # Mock torch.npu if not present
        if not hasattr(torch, 'npu'):
            torch.npu = MagicMock()
            torch.npu.get_device_properties.return_value = MagicMock(name="Ascend910B")

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    def test_singleton_creation(self, mock_execute):
        """Test _NPUNodeInfo is a singleton."""
        mock_execute.return_value = "Header\n0 1 0 Ascend910B\n"

        info1 = get_npu_node_info()
        info2 = get_npu_node_info()
        self.assertIs(info1, info2)

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    @patch("mindie_llm.runtime.utils.npu.device_utils.torch.npu.get_device_properties")
    def test_init_soc_name_and_flags(self, mock_get_device_properties, mock_execute):
        mock_get_device_properties.return_value.name = "Ascend910B"
        node_info = _NPUNodeInfo()
        self.assertEqual(node_info.soc_name, "Ascend910B")
        self.assertTrue(node_info.only_supports_nz)
        self.assertTrue(node_info.need_nz)

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    def test_get_device_info_map_success(self, mock_execute):
        """Test parsing npu-smi info -m output."""
        mock_execute.return_value = (
            "NPU ID Chip ID Logic ID Chip Name\n"
            "0 1 0 Ascend910B\n"
            "1 2 1 Ascend910B\n"
            "2 3 N/A Ascend310P\n"  # Should be skipped
        )

        info = get_npu_node_info()
        device_map = info.get_device_info_map()
        self.assertIn(0, device_map)
        self.assertIn(1, device_map)
        self.assertNotIn(2, device_map)  # Non-numeric logic ID skipped
        self.assertEqual(len(device_map), 2)

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    def test_visible_device_ids_from_env(self, mock_execute):
        """Test visible_device_ids uses ENV.visible_devices."""
        mock_execute.return_value = "Header\n0 1 0 Ascend910B\n1 2 1 Ascend910B\n"
        ENV.visible_devices = [0, 1]

        info = get_npu_node_info()
        self.assertEqual(info.visible_device_ids, [0, 1])

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    def test_visible_device_ids_auto_detect(self, mock_execute):
        """Test visible_device_ids auto-detects from npu-smi."""
        mock_execute.return_value = (
            "Header\n"
            "0 1 0 Ascend910B\n"
            "1 2 1 Ascend910B\n"
        )
        ENV.visible_devices = None

        info = get_npu_node_info()
        self.assertEqual(info.visible_device_ids, [0, 1])

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    def test_get_pcie_info_success(self, mock_execute):
        """Test get_pcie_info parses PCIeBusInfo correctly."""
        # First call: npu-smi info -m
        # Second call: npu-smi info -t board
        mock_execute.side_effect = [
            "Header\n0 1 0 Ascend910B\n",
            "PCIeBusInfo:0000:1a:00.0\nOther:info\n"
        ]

        info = get_npu_node_info()
        pcie_map = info.get_pcie_info([0])
        self.assertEqual(pcie_map, {0: "0000:1a:00.0"})

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    def test_get_pcie_info_missing_device(self, mock_execute):
        """Test get_pcie_info raises RuntimeError for unknown device."""
        mock_execute.return_value = "Header\n0 1 0 Ascend910B\n"

        info = get_npu_node_info()
        with self.assertRaises(RuntimeError):
            info.get_pcie_info([999])  # Device 999 not in map

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    def test_is_support_hccs_true(self, mock_execute):
        """Test is_support_hccs detects HCCS topology."""
        mock_execute.return_value = "hccs connection\nLegend: ..."

        self.assertTrue(_NPUNodeInfo.is_support_hccs())

    @patch("mindie_llm.runtime.utils.npu.device_utils.execute_command")
    def test_is_support_hccs_false(self, mock_execute):
        """Test is_support_hccs returns False when no HCCS/XLink."""
        mock_execute.return_value = "pcie connection\nLegend: ..."

        self.assertFalse(_NPUNodeInfo.is_support_hccs())

    @patch("mindie_llm.runtime.utils.npu.device_utils.torch.npu.get_device_properties")
    def test_get_device_type_ascend_910b(self, mock_get_device_properties):
        # Set SoC name via torch.npu mock
        mock_get_device_properties.return_value.name = "Ascend910B1"
        
        node_info = _NPUNodeInfo()
        device_type = node_info.get_device_type()
        self.assertEqual(device_type, DeviceType.ASCEND_910B)

    @patch("mindie_llm.runtime.utils.npu.device_utils.torch.npu.get_device_properties")
    def test_get_device_type_ascend_310p(self, mock_get_device_properties):
        mock_get_device_properties.return_value.name = "Ascend310P1"
        node_info = _NPUNodeInfo()
        self.assertEqual(node_info.get_device_type(), DeviceType.ASCEND_310P)


class TestNPUHbmInfo(unittest.TestCase):
    """Test _NPUHbmInfo memory querying."""

    def setUp(self):
        """Reset class cache."""
        _NPUHbmInfo._hbm_capacity = None
        _NPUHbmInfo._instance = None

    @patch("mindie_llm.runtime.utils.npu.device_utils.acl.rt.get_mem_info")
    def test_get_hbm_capacity_usage(self, mock_get_mem_info):
        """Test get_hbm_capacity_usage returns correct tuple."""
        mock_get_mem_info.return_value = (4000, 10000, 0)  # free, total, _ 
        hbm_info = get_npu_hbm_info()
        total, usage = hbm_info.get_hbm_capacity_usage()
        self.assertEqual(total, 10000)
        self.assertAlmostEqual(usage, 0.6)  # (10000 - 4000) / 10000 = 0.6

    @patch("mindie_llm.runtime.utils.npu.device_utils.acl.rt.get_mem_info")
    def test_get_hbm_capacity_cached(self, mock_get_mem_info):
        """Test get_hbm_capacity caches result."""
        mock_get_mem_info.return_value = (5000, 20000, 0)
        hbm_info = get_npu_hbm_info()

        cap1 = hbm_info.get_hbm_capacity()
        cap2 = hbm_info.get_hbm_capacity()
        self.assertEqual(cap1, 20000)
        self.assertEqual(cap2, 20000)
        mock_get_mem_info.assert_called_once()  # Only called once

    @patch("mindie_llm.runtime.utils.npu.device_utils.acl.rt.get_mem_info")
    def test_get_hbm_usage(self, mock_get_mem_info):
        """Test get_hbm_usage returns usage ratio."""
        mock_get_mem_info.return_value = (3000, 10000, 0)
        hbm_info = get_npu_hbm_info()
        usage = hbm_info.get_hbm_usage()
        self.assertAlmostEqual(usage, 0.7)


if __name__ == "__main__":
    unittest.main()