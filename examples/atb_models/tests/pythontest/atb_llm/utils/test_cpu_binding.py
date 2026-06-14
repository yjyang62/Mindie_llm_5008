# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import patch, MagicMock
from atb_llm.utils.cpu_binding import (
    _get_numa_info, 
    _get_numa_info_v2, 
    execute_command, 
    DeviceInfo,
    _get_pcie_info,
    _get_cpu_info,
    bind_cpus,
    NpuHbmInfo,
)


class TestNumaInfo(unittest.TestCase):
    def setUp(self):
        # 模拟 lspci 的输出
        self.sample_lspci_output = """
        NUMAnode: 0
        """
        # 模拟 lscpu 的输出
        self.sample_lscpu_output = """
        Architecture:          x86_64
        CPU(s):                96
        NUMA node0 CPU(s):     0-23,48-71
        NUMA node1 CPU(s):     24-47,72-95
        """

    @patch("atb_llm.utils.cpu_binding.execute_command")
    def test_get_numa_info(self, mock_exec):
        """测试 _get_numa_info 函数"""
        # 模拟 lspci 的输出
        mock_exec.return_value = self.sample_lspci_output

        # 模拟输入：PCIe 设备表
        pcie_tbl = {
            0: "0000:89:00.0",  # 设备 0 的 PCIe 地址
            1: "0000:89:00.1",  # 设备 1 的 PCIe 地址
        }

        # 调用函数
        device_numa_tbl, numa_devices_tbl = _get_numa_info(pcie_tbl)

        # 验证设备到 NUMA 的映射
        self.assertIn(0, device_numa_tbl)
        self.assertEqual(device_numa_tbl[0], 0)  # 设备 0 应该映射到 NUMA 0
        self.assertIn(1, device_numa_tbl)
        self.assertEqual(device_numa_tbl[1], 0)  # 设备 1 应该映射到 NUMA 0

        # 验证 NUMA 到设备的映射
        self.assertIn(0, numa_devices_tbl)
        self.assertEqual(numa_devices_tbl[0], [0, 1])  # NUMA 0 应该包含设备 0 和 1

    @patch("atb_llm.utils.cpu_binding.execute_command")
    def test_get_numa_info_v2(self, mock_exec):
        """测试 _get_numa_info_v2 函数"""
        # 模拟 lscpu 的输出
        mock_exec.return_value = self.sample_lscpu_output

        # 模拟输入：设备列表
        devices = [0, 1, 2, 3]  # 4 个设备

        # 调用函数
        device_numa_tbl, numa_devices_tbl = _get_numa_info_v2(devices)

        # 验证设备到 NUMA 的映射
        self.assertIn(0, device_numa_tbl)
        self.assertIn(1, device_numa_tbl)
        self.assertIn(2, device_numa_tbl)
        self.assertIn(3, device_numa_tbl)

        # 验证 NUMA 到设备的映射
        self.assertIn(0, numa_devices_tbl)

        # 检查设备是否分配到 NUMA 节点
        self.assertEqual(len(numa_devices_tbl[0]), 4) 

    @patch("atb_llm.utils.cpu_binding.execute_command")
    def test_get_numa_info_v2_uneven_devices(self, mock_exec):
        """测试 _get_numa_info_v2 函数在设备数不均匀时的分配"""
        # 模拟 lscpu 的输出
        mock_exec.return_value = self.sample_lscpu_output

        # 模拟输入：5 个设备
        devices = [0, 1, 2, 3, 4]

        # 调用函数
        device_numa_tbl, numa_devices_tbl = _get_numa_info_v2(devices)

        # 验证设备到 NUMA 的映射
        self.assertIn(0, device_numa_tbl)
        self.assertIn(1, device_numa_tbl)
        self.assertIn(2, device_numa_tbl)
        self.assertIn(3, device_numa_tbl)
        self.assertIn(4, device_numa_tbl)

        # 验证 NUMA 到设备的映射
        self.assertIn(0, numa_devices_tbl)

        # 检查设备是否分配到 NUMA 节点
        self.assertEqual(len(numa_devices_tbl[0]), 5) 


class TestHelpers(unittest.TestCase):
    @patch("subprocess.Popen")
    def test_execute_command(self, mock_popen):
        """测试 execute_command 函数"""
        process_mock = MagicMock()
        process_mock.communicate.return_value = (b"output\n", b"error")
        mock_popen.return_value.__enter__.return_value = process_mock
        
        res = execute_command(["ls"])
        self.assertEqual(res, "output\n")

    def test_device_info(self):
        """测试 DeviceInfo 数据类"""
        # line format: npu_id chip_id chip_logic_id chip_name
        line = "8 0 0 910B"
        info = DeviceInfo(line)
        self.assertEqual(info.npu_id, 8)
        self.assertEqual(info.chip_id, 0)
        self.assertEqual(info.chip_logic_id, 0)
        self.assertEqual(info.chip_name, "910B")

        # Test with string logic id
        line_str = "8 0 0 910B" 
        info_str = DeviceInfo(line_str)
        self.assertEqual(info_str.chip_logic_id, 0)

    def test_device_info_non_numeric_logic_id(self):
        """测试 DeviceInfo with non-numeric chip_logic_id"""
        line = "8 0 N/A 910B"
        info = DeviceInfo(line)
        self.assertEqual(info.npu_id, 8)
        self.assertEqual(info.chip_id, 0)
        self.assertEqual(info.chip_logic_id, "N/A")


class TestCpuBindingLogic(unittest.TestCase):
    @patch("atb_llm.utils.cpu_binding.execute_command")
    def test_get_pcie_info(self, mock_exec):
        """测试 _get_pcie_info 函数"""
        def side_effect(cmd):
            if "info" in cmd and "-m" in cmd:
                return "NPU ID  Chip ID  Chip Logic ID  Chip Name\n8 0 0 910B"
            if "-t" in cmd and "board" in cmd:
                return "Some Header\nPCIeBusInfo: 0000:01:00.0"
            return ""
        
        mock_exec.side_effect = side_effect
        
        res = _get_pcie_info([0])
        self.assertIn(0, res)
        self.assertEqual(res[0], "0000:01:00.0")

    @patch("atb_llm.utils.cpu_binding.execute_command")
    @patch("atb_llm.utils.cpu_binding._get_device_map_info")
    def test_get_pcie_info_missing_device(self, mock_dev_map, mock_exec):
        """测试 _get_pcie_info 函数：设备信息缺失"""
        mock_dev_map.return_value = {} # No devices found
        
        with self.assertRaisesRegex(RuntimeError, "Can not get device info for device"):
            _get_pcie_info([0])

    @patch("atb_llm.utils.cpu_binding.execute_command")
    def test_get_cpu_info(self, mock_exec):
        """测试 _get_cpu_info 函数"""
        mock_exec.return_value = """
        Architecture:          x86_64
        CPU(s):                96
        NUMA node0 CPU(s):     0-23,48-71
        NUMA node1 CPU(s):     24-47,72-95
        """
        # Test for NUMA 0 and 1
        res = _get_cpu_info([0, 1])
        
        expected_0 = list(range(0, 24)) + list(range(48, 72))
        expected_1 = list(range(24, 48)) + list(range(72, 96))
        
        self.assertEqual(res[0], expected_0)
        self.assertEqual(res[1], expected_1)

    @patch("atb_llm.utils.cpu_binding.execute_command")
    def test_get_cpu_info_malformed_range(self, mock_exec):
        """测试 _get_cpu_info 函数：CPU range 格式错误"""
        mock_exec.return_value = """
        Architecture:          x86_64
        CPU(s):                96
        NUMA node0 CPU(s):     0-23-99
        """
        with self.assertRaisesRegex(RuntimeError, "Cannot obtain CPU range for NUMA"):
            _get_cpu_info([0])


class TestBindCpus(unittest.TestCase):
    def setUp(self):
        self.sample_lscpu = """
        Architecture:          x86_64
        CPU(s):                96
        NUMA node0 CPU(s):     0-23,48-71
        NUMA node1 CPU(s):     24-47,72-95
        """

    def test_bind_cpus_basic(self):
        """测试 bind_cpus 基本功能：默认比例分配"""
        with patch("atb_llm.utils.cpu_binding.psutil.Process") as mock_psutil, \
             patch("atb_llm.utils.cpu_binding._get_cpu_info") as mock_cpu, \
             patch("atb_llm.utils.cpu_binding._get_numa_info") as mock_numa, \
             patch("atb_llm.utils.cpu_binding._get_pcie_info") as mock_pcie, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0, 1]
            mock_env.cpu_binding_num = None
            
            mock_pcie.return_value = {0: "pci0", 1: "pci1"}
            
            # Device 0 -> NUMA 0, Device 1 -> NUMA 0
            mock_numa.return_value = (
                {0: 0, 1: 0}, 
                {0: [0, 1]}
            )
            
            # NUMA 0 has 10 CPUs (0-9)
            mock_cpu.return_value = {0: list(range(10))}
            
            # Call bind_cpus for rank 0 (device 0)
            bind_cpus(rank_id=0, ratio=0.5)
            
            process = mock_psutil.return_value
            process.cpu_affinity.assert_any_call([0, 1])

    def test_bind_cpus_explicit_num(self):
        """测试 bind_cpus：指定 CPU 绑定数量"""
        with patch("atb_llm.utils.cpu_binding.psutil.Process") as mock_psutil, \
             patch("atb_llm.utils.cpu_binding._get_cpu_info") as mock_cpu, \
             patch("atb_llm.utils.cpu_binding._get_numa_info") as mock_numa, \
             patch("atb_llm.utils.cpu_binding._get_pcie_info") as mock_pcie, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0]
            mock_env.cpu_binding_num = 4 # Explicitly bind 4 CPUs
            
            mock_pcie.return_value = {0: "pci0"}
            mock_numa.return_value = ({0: 0}, {0: [0]})
            mock_cpu.return_value = {0: list(range(10))}
            
            bind_cpus(rank_id=0)
            
            process = mock_psutil.return_value
            process.cpu_affinity.assert_any_call([0, 1, 2, 3])

    def test_bind_cpus_not_enough_cpus(self):
        """测试 bind_cpus：CPU 资源不足抛出异常"""
        with patch("atb_llm.utils.cpu_binding._get_cpu_info") as mock_cpu, \
             patch("atb_llm.utils.cpu_binding._get_numa_info") as mock_numa, \
             patch("atb_llm.utils.cpu_binding._get_pcie_info") as mock_pcie, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0, 1]
            mock_env.cpu_binding_num = 6 # Need 6 per device, total 12 needed
            
            mock_pcie.return_value = {0: "pci0", 1: "pci1"}
            mock_numa.return_value = ({0: 0, 1: 0}, {0: [0, 1]})
            mock_cpu.return_value = {0: list(range(10))} # Only 10 available
            
            with self.assertRaises(ValueError):
                bind_cpus(rank_id=0)
            
    def test_bind_cpus_fallback_v2(self):
        """测试 bind_cpus：V1 失败回退到 V2"""
        with patch("atb_llm.utils.cpu_binding.psutil.Process") as mock_psutil, \
             patch("atb_llm.utils.cpu_binding._get_cpu_info") as mock_cpu, \
             patch("atb_llm.utils.cpu_binding._get_numa_info_v2") as mock_numa_v2, \
             patch("atb_llm.utils.cpu_binding._get_numa_info") as mock_numa, \
             patch("atb_llm.utils.cpu_binding._get_pcie_info") as mock_pcie, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0]
            mock_env.cpu_binding_num = None
            
            mock_pcie.return_value = {0: "pci0"}
            mock_numa.return_value = ({}, {}) # V1 fails
            
            mock_numa_v2.return_value = ({0: 0}, {0: [0]}) # V2 succeeds
            mock_cpu.return_value = {0: list(range(10))}
            
            bind_cpus(rank_id=0)
            
            mock_numa_v2.assert_called()
            process = mock_psutil.return_value
            process.cpu_affinity.assert_called()
    
    def test_bind_cpus_invalid_num(self):
        """测试 bind_cpus：非法输入（负数）"""
        with patch("atb_llm.utils.cpu_binding._get_cpu_info") as mock_cpu, \
             patch("atb_llm.utils.cpu_binding._get_numa_info") as mock_numa, \
             patch("atb_llm.utils.cpu_binding._get_pcie_info") as mock_pcie, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0]
            mock_env.cpu_binding_num = -1
            
            mock_pcie.return_value = {0: "pci0"}
            mock_numa.return_value = ({0: 0}, {0: [0]})
            mock_cpu.return_value = {0: list(range(10))}
            
            with self.assertRaises(ValueError):
                bind_cpus(rank_id=0)

    def test_bind_cpus_no_visible_devices_env(self):
        """测试 bind_cpus：ENV.visible_devices 为 None"""
        with patch("atb_llm.utils.cpu_binding.psutil.Process") as mock_psutil, \
             patch("atb_llm.utils.cpu_binding._get_cpu_info") as mock_cpu, \
             patch("atb_llm.utils.cpu_binding._get_numa_info") as mock_numa, \
             patch("atb_llm.utils.cpu_binding._get_pcie_info") as mock_pcie, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env, \
             patch("atb_llm.utils.cpu_binding._get_device_map_info") as mock_dev_map:
            mock_env.visible_devices = None
            mock_env.cpu_binding_num = None
            
            # _get_device_map_info returns dict with keys
            mock_dev_map.return_value = {0: MagicMock(), 1: MagicMock()} 
            
            mock_pcie.return_value = {0: "pci0", 1: "pci1"}
            mock_numa.return_value = (
                {0: 0, 1: 0}, 
                {0: [0, 1]}
            )
            mock_cpu.return_value = {0: list(range(10))}
            
            bind_cpus(rank_id=0)
            
            # Should verify that it used devices [0, 1]
            mock_pcie.assert_called_with([0, 1])


class TestNpuHbmInfo(unittest.TestCase):
    def tearDown(self):
        # Reset class state
        NpuHbmInfo.visible_npu_ids = None
        NpuHbmInfo.hbm_capacity = None
        NpuHbmInfo.hbm_usage = None

    def test_get_hbm_capacity(self):
        """测试 NpuHbmInfo.get_hbm_capacity"""
        with patch("atb_llm.utils.cpu_binding.execute_command") as mock_exec, \
             patch("atb_llm.utils.cpu_binding.torch_npu._C._npu_get_soc_version") as mock_soc, \
             patch("atb_llm.utils.cpu_binding._get_device_map_info") as mock_dev_map, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            # Setup
            mock_env.visible_devices = [0]
            mock_dev_map.return_value = {0: MagicMock(npu_id=0)}
            
            mock_soc.return_value = 240 # Example version
            
            # Mock npu-smi output
            # Output format expected by code: key: value
            mock_exec.return_value = "Header\nCapacity(MB): 1024"
            
            cap = NpuHbmInfo.get_hbm_capacity(rank=0, world_size=1, need_nz=False)
            self.assertEqual(cap, 1024 * 1024 * 1024)

    def test_get_hbm_usage(self):
        """测试 NpuHbmInfo.get_hbm_usage"""
        with patch("atb_llm.utils.cpu_binding.execute_command") as mock_exec, \
             patch("atb_llm.utils.cpu_binding.torch_npu._C._npu_get_soc_version") as mock_soc, \
             patch("atb_llm.utils.cpu_binding._get_device_map_info") as mock_dev_map, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0]
            mock_dev_map.return_value = {0: MagicMock(npu_id=0)}
            
            mock_soc.return_value = 240
            
            # Mock usage output
            mock_exec.return_value = "Header\nMemory Usage Rate(%): 50"
            
            usage = NpuHbmInfo.get_hbm_usage(rank=0, world_size=1, need_nz=False)
            # (50 + 1) / 100 = 0.51
            self.assertAlmostEqual(usage, 0.51)

    def test_get_hbm_capacity_cached(self):
        """测试 NpuHbmInfo.get_hbm_capacity: cached"""
        NpuHbmInfo.hbm_capacity = 12345
        self.assertEqual(NpuHbmInfo.get_hbm_capacity(0, 1, False), 12345)

    def test_get_hbm_usage_cached(self):
        """测试 NpuHbmInfo.get_hbm_usage: cached"""
        NpuHbmInfo.hbm_usage = 0.99
        self.assertEqual(NpuHbmInfo.get_hbm_usage(0, 1, False), 0.99)

    def test_set_visible_devices_no_env(self):
        """测试 NpuHbmInfo.set_visible_devices: ENV.visible_devices is None"""
        with patch("atb_llm.utils.cpu_binding._get_device_map_info") as mock_dev_map, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = None
            mock_dev_map.return_value = {0: MagicMock(npu_id=10), 1: MagicMock(npu_id=11)}
            
            NpuHbmInfo.set_visible_devices(2)
            
            self.assertEqual(NpuHbmInfo.visible_npu_ids, [10, 11])

    def test_set_visible_devices_cached(self):
        """测试 NpuHbmInfo.set_visible_devices: already cached"""
        NpuHbmInfo.visible_npu_ids = [99]
        
        NpuHbmInfo.set_visible_devices(1)
        self.assertEqual(NpuHbmInfo.visible_npu_ids, [99])

    def test_get_hbm_capacity_other_soc(self):
        """测试 NpuHbmInfo.get_hbm_capacity: other soc version"""
        with patch("atb_llm.utils.cpu_binding.execute_command") as mock_exec, \
             patch("atb_llm.utils.cpu_binding.torch_npu._C._npu_get_soc_version") as mock_soc, \
             patch("atb_llm.utils.cpu_binding._get_device_map_info") as mock_dev_map, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0]
            mock_dev_map.return_value = {0: MagicMock(npu_id=0)}
            mock_soc.return_value = 100 # Not 240
            
            # Test need_nz=False
            mock_exec.return_value = "Header\nHBM Capacity(MB): 2048"
            cap = NpuHbmInfo.get_hbm_capacity(rank=0, world_size=1, need_nz=False)
            self.assertEqual(cap, 2048 * 1024 * 1024)
            
            # Reset cache
            NpuHbmInfo.hbm_capacity = None
            
            # Test need_nz=True
            mock_exec.return_value = "Header\nDDR Capacity(MB): 4096"
            cap = NpuHbmInfo.get_hbm_capacity(rank=0, world_size=1, need_nz=True)
            self.assertEqual(cap, 4096 * 1024 * 1024)

    def test_get_hbm_capacity_parse_error(self):
        """测试 NpuHbmInfo.get_hbm_capacity: parse error / not found"""
        with patch("atb_llm.utils.cpu_binding.execute_command") as mock_exec, \
             patch("atb_llm.utils.cpu_binding.torch_npu._C._npu_get_soc_version") as mock_soc, \
             patch("atb_llm.utils.cpu_binding._get_device_map_info") as mock_dev_map, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0]
            mock_dev_map.return_value = {0: MagicMock(npu_id=0)}
            mock_soc.return_value = 240
            
            # Invalid format
            mock_exec.return_value = "Header\nInvalidLine"
            
            with self.assertRaisesRegex(ValueError, "not found valid hbm capactiy"):
                NpuHbmInfo.get_hbm_capacity(rank=0, world_size=1, need_nz=False)

            # Valid format but ValueError during int conversion (simulated by malformed value)
            # But the code has try-except ValueError. So it should skip and then raise not found.
            mock_exec.return_value = "Header\nCapacity(MB): NotANumber"
            with self.assertRaisesRegex(ValueError, "not found valid hbm capactiy"):
                NpuHbmInfo.get_hbm_capacity(rank=0, world_size=1, need_nz=False)

    def test_get_hbm_usage_other_soc(self):
        """测试 NpuHbmInfo.get_hbm_usage: other soc version"""
        with patch("atb_llm.utils.cpu_binding.execute_command") as mock_exec, \
             patch("atb_llm.utils.cpu_binding.torch_npu._C._npu_get_soc_version") as mock_soc, \
             patch("atb_llm.utils.cpu_binding._get_device_map_info") as mock_dev_map, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0]
            mock_dev_map.return_value = {0: MagicMock(npu_id=0)}
            mock_soc.return_value = 100
            
            # Test need_nz=False
            mock_exec.return_value = "Header\nHBM Usage Rate(%): 10"
            usage = NpuHbmInfo.get_hbm_usage(rank=0, world_size=1, need_nz=False)
            self.assertAlmostEqual(usage, 0.11)
            
            # Reset cache
            NpuHbmInfo.hbm_usage = None
            
            # Test need_nz=True
            mock_exec.return_value = "Header\nDDR Usage Rate(%): 20"
            usage = NpuHbmInfo.get_hbm_usage(rank=0, world_size=1, need_nz=True)
            self.assertAlmostEqual(usage, 0.21)

    def test_get_hbm_usage_parse_error(self):
        """测试 NpuHbmInfo.get_hbm_usage: parse error / not found"""
        with patch("atb_llm.utils.cpu_binding.execute_command") as mock_exec, \
             patch("atb_llm.utils.cpu_binding.torch_npu._C._npu_get_soc_version") as mock_soc, \
             patch("atb_llm.utils.cpu_binding._get_device_map_info") as mock_dev_map, \
             patch("atb_llm.utils.cpu_binding.ENV") as mock_env:
            mock_env.visible_devices = [0]
            mock_dev_map.return_value = {0: MagicMock(npu_id=0)}
            mock_soc.return_value = 240
            
            mock_exec.return_value = "Header\nInvalidLine"
            
            with self.assertRaisesRegex(ValueError, "not found valid hbm usage"):
                NpuHbmInfo.get_hbm_usage(rank=0, world_size=1, need_nz=False)


if __name__ == "__main__":
    unittest.main()