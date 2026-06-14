# Copyright (c) Huawei Technologies Co., Ltd. 2023-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import patch, MagicMock, Mock
from mindie_llm.runtime.utils.cpu.affinity import (
    _get_lscpu,
    _get_numa_info_by_pci,
    _get_balanced_numa_info,
    _get_numa_cpu_affinity,
    bind_cpus,
    _LSCPU_STRING,
)
from mindie_llm.runtime.utils.helpers.env import ENV


class TestCPUAffinityUtils(unittest.TestCase):
    """Test cases for CPU affinity utilities."""

    def setUp(self):
        """Reset global state before each test."""
        import mindie_llm.runtime.utils.cpu.affinity as affinity_module
        affinity_module._LSCPU_STRING = None
        ENV.cpu_binding_num = None
        ENV.visible_devices = None

    @patch("mindie_llm.runtime.utils.cpu.affinity.execute_command")
    def test_get_lscpu_cached(self, mock_execute):
        """Test _get_lscpu with caching."""
        mock_execute.return_value = "lscpu output\n"

        # First call
        result1 = _get_lscpu()
        self.assertEqual(result1, "lscpu output\n")
        mock_execute.assert_called_once_with(["lscpu"])

        # Second call should use cache
        result2 = _get_lscpu()
        self.assertEqual(result2, "lscpu output\n")
        mock_execute.assert_called_once()  # still only once

    @patch("mindie_llm.runtime.utils.cpu.affinity.execute_command")
    def test_get_numa_info_by_pci_success(self, mock_execute):
        """Test _get_numa_info_by_pci with valid lspci output."""
        mock_execute.return_value = "NUMAnode: 1\nother info"
        device2pcie = {0: "0000:1a:00.0", 1: "0000:1b:00.0"}
        device2numa, numa2devices = _get_numa_info_by_pci(device2pcie)

        expected_device2numa = {0: 1, 1: 1}
        expected_numa2devices = {1: [0, 1]}
        self.assertEqual(device2numa, expected_device2numa)
        self.assertEqual(numa2devices, expected_numa2devices)

    @patch("mindie_llm.runtime.utils.cpu.affinity.execute_command")
    def test_get_numa_info_by_pci_no_keyword(self, mock_execute):
        """Test _get_numa_info_by_pci when NUMA info is missing."""
        mock_execute.return_value = "some output without keyword\n"
        device2pcie = {0: "0000:1a:00.0"}
        device2numa, numa2devices = _get_numa_info_by_pci(device2pcie)
        self.assertEqual(device2numa, {})
        self.assertEqual(numa2devices, {})

    @patch("mindie_llm.runtime.utils.cpu.affinity._get_lscpu")
    def test_get_balanced_numa_info_4_devices_2_numa(self, mock_lscpu):
        """Test balanced assignment: 4 devices, 2 NUMA nodes."""
        mock_lscpu.return_value = "NUMAnode(s):2\nOther:info"
        devices = [0, 1, 2, 3]
        device2numa, numa2devices = _get_balanced_numa_info(devices)

        self.assertEqual(device2numa, {0: 0, 1: 0, 2: 1, 3: 1})
        self.assertEqual(numa2devices, {0: [0, 1], 1: [2, 3]})

    @patch("mindie_llm.runtime.utils.cpu.affinity._get_lscpu")
    def test_get_balanced_numa_info_5_devices_3_numa(self, mock_lscpu):
        """Test balanced assignment with remainder: 5 devices, 3 NUMA nodes."""
        mock_lscpu.return_value = "NUMAnode(s):3\n"
        devices = [0, 1, 2, 3, 4]
        device2numa, numa2devices = _get_balanced_numa_info(devices)

        self.assertEqual(device2numa, {0: 0, 1: 0, 2: 1, 3: 1, 4: 2})
        self.assertEqual(numa2devices, {0: [0, 1], 1: [2, 3], 2: [4]})

    @patch("mindie_llm.runtime.utils.cpu.affinity._get_lscpu")
    def test_get_numa_cpu_affinity_success(self, mock_lscpu):
        """Test _get_numa_cpu_affinity with valid lscpu output."""
        mock_lscpu.return_value = (
            "NUMAnode0CPU(s):0-31,64-95\n"
            "NUMAnode1CPU(s):32-63,96-127\n"
        )
        numa_ids = [0, 1]
        result = _get_numa_cpu_affinity(numa_ids)

        expected = {
            0: list(range(0, 32)) + list(range(64, 96)),
            1: list(range(32, 64)) + list(range(96, 128)),
        }
        self.assertEqual(result, expected)

    @patch("mindie_llm.runtime.utils.cpu.affinity._get_lscpu")
    def test_get_numa_cpu_affinity_invalid_range(self, mock_lscpu):
        """Test _get_numa_cpu_affinity raises RuntimeError on invalid range."""
        mock_lscpu.return_value = "NUMAnode0CPU(s):0\n"  # single number, not range
        with self.assertRaises(RuntimeError):
            _get_numa_cpu_affinity([0])

    @patch("mindie_llm.runtime.utils.cpu.affinity.get_npu_node_info")
    @patch("mindie_llm.runtime.utils.cpu.affinity.get_parallel_info_manager")
    @patch("mindie_llm.runtime.utils.cpu.affinity._get_lscpu")
    @patch("mindie_llm.runtime.utils.cpu.affinity.execute_command")
    @patch("psutil.Process")
    def test_bind_cpus_success_ratio_path(
        self, mock_psutil_process, mock_execute, mock_lscpu, mock_pim, mock_npu_info
    ):
        """Test bind_cpus using ratio (CPU_BINDING_NUM not set)."""
        ENV.cpu_binding_num = None
        ENV.visible_devices = [0, 1, 2, 3]

        mock_npu = MagicMock()
        mock_npu.visible_device_ids = [0, 1, 2, 3]
        # PCIe info only for 0,1 — but that's OK, rest will be handled by balanced assignment
        mock_npu.get_pcie_info.return_value = {0: "0000:1a:00.0", 1: "0000:1b:00.0"}
        mock_npu_info.return_value = mock_npu

        # Make PCIe-based NUMA fail (return empty) → fallback to balanced
        mock_execute.return_value = ""  # lspci returns nothing → device2numa={}

        # Mock _get_lscpu: first call for NUMA count, second for CPU ranges
        mock_lscpu.side_effect = [
            "NUMAnode(s):2\n",  # First call in _get_balanced_numa_info
            "NUMAnode0CPU(s):0-15\nNUMAnode1CPU(s):16-31\n"  # Second call in _get_numa_cpu_affinity
        ]

        mock_pim_instance = MagicMock()
        mock_pim_instance.rank = 0
        mock_pim.return_value = mock_pim_instance

        mock_process = MagicMock()
        mock_psutil_process.return_value = mock_process

        # Run
        bind_cpus(ratio=1.0)

        # Verify: rank 0 → device 0 → NUMA 0 → shard_devices [0,1] → gets first half of 16 cores = 8 cores
        expected_cpus = list(range(0, 8))  # 16 cores / 2 devices = 8 per device
        mock_process.cpu_affinity.assert_any_call(expected_cpus)

    @patch("mindie_llm.runtime.utils.cpu.affinity.get_npu_node_info")
    @patch("mindie_llm.runtime.utils.cpu.affinity.get_parallel_info_manager")
    @patch("mindie_llm.runtime.utils.cpu.affinity._get_lscpu")
    @patch("mindie_llm.runtime.utils.cpu.affinity.execute_command")
    @patch("psutil.Process")
    def test_bind_cpus_with_cpu_binding_num(
        self, mock_psutil_process, mock_execute, mock_lscpu, mock_pim, mock_npu_info
    ):
        """Test bind_cpus using ENV.cpu_binding_num."""
        ENV.cpu_binding_num = 4
        ENV.visible_devices = [0, 1]

        mock_npu = MagicMock()
        mock_npu.visible_device_ids = [0, 1]
        mock_npu.get_pcie_info.return_value = {0: "xx", 1: "yy"}
        mock_npu_info.return_value = mock_npu

        # Make PCIe-based NUMA work
        mock_execute.side_effect = lambda cmd: "NUMAnode:0\n" if "lspci" in cmd else ""

        mock_lscpu.return_value = "NUMAnode0CPU(s):0-15\n"

        mock_pim_instance = MagicMock()
        mock_pim_instance.rank = 1
        mock_pim.return_value = mock_pim_instance

        mock_process = MagicMock()
        mock_psutil_process.return_value = mock_process

        bind_cpus(ratio=0.5)  # ratio ignored because CPU_BINDING_NUM is set

        # Rank 1 → device 1 → NUMA 0 → shard_devices [0,1] → idx=1 → cores 4-7
        expected_cpus = [4, 5, 6, 7]
        mock_process.cpu_affinity.assert_any_call(expected_cpus)

    @patch("mindie_llm.runtime.utils.cpu.affinity.get_npu_node_info")
    @patch("mindie_llm.runtime.utils.cpu.affinity.get_parallel_info_manager")
    @patch("mindie_llm.runtime.utils.cpu.affinity._get_lscpu")
    @patch("mindie_llm.runtime.utils.cpu.affinity.execute_command")
    def test_bind_cpus_cpu_binding_num_too_large(
        self, mock_execute, mock_lscpu, mock_pim, mock_npu_info
    ):
        """Test bind_cpus raises ValueError when CPU_BINDING_NUM exceeds available cores."""
        ENV.cpu_binding_num = 100
        ENV.visible_devices = [0, 1]

        mock_npu = MagicMock()
        mock_npu.visible_device_ids = [0, 1]
        mock_npu.get_pcie_info.return_value = {0: "xx", 1: "yy"}
        mock_npu_info.return_value = mock_npu

        mock_execute.side_effect = lambda cmd: "NUMAnode:0\n" if "lspci" in cmd else ""
        mock_lscpu.return_value = "NUMAnode0CPU(s):0-15\n"  # 16 cores

        mock_pim_instance = MagicMock()
        mock_pim_instance.rank = 0
        mock_pim.return_value = mock_pim_instance

        with self.assertRaises(ValueError) as cm:
            bind_cpus()
        self.assertIn("not enough", str(cm.exception).lower())

    @patch("mindie_llm.runtime.utils.cpu.affinity.get_parallel_info_manager")
    @patch("mindie_llm.runtime.utils.cpu.affinity.get_npu_node_info")
    def test_bind_cpus_empty_visible_devices(self, mock_npu_info, mock_pim):
        mock_pim_instance = MagicMock()
        mock_pim_instance.rank = 0
        mock_pim.return_value = mock_pim_instance

        mock_npu = MagicMock()
        mock_npu.visible_device_ids = []
        mock_npu.get_pcie_info.return_value = {}
        mock_npu_info.return_value = mock_npu

        with self.assertRaises(IndexError):  # devices[0] on empty list
            bind_cpus()


if __name__ == "__main__":
    unittest.main()