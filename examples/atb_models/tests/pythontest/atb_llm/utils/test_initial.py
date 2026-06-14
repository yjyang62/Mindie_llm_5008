# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from unittest.mock import patch
from atb_llm.utils.initial import (
    Topo,
    CommunicationLibrary,
    NPUSocInfo,
    load_atb_speed,
    is_lcoc_enable
)


class TestTopo:
    def test_topo_enum(self):
        assert Topo.pcie.value == "pcie"
        assert Topo.hccs.value == "hccs"


class TestCommunicationLibrary:
    def test_communication_library_enum(self):
        assert CommunicationLibrary.hccl.value == "hccl"
        assert CommunicationLibrary.lccl.value == "lccl"


class TestNPUSocInfo:
    @patch("atb_llm.utils.initial.torch_npu._C._npu_get_soc_version")
    def test_post_init_with_need_nz_soc(self, mock_get_soc):
        """Test SOC version that requires NZ."""
        mock_get_soc.return_value = 100
        soc_info = NPUSocInfo()
        assert soc_info.soc_version == 100
        assert soc_info.need_nz is True
        assert soc_info.support_bf16 is False

    @patch("atb_llm.utils.initial.torch_npu._C._npu_get_soc_version")
    def test_post_init_without_need_nz_soc(self, mock_get_soc):
        """Test SOC version that doesn't require NZ."""
        mock_get_soc.return_value = 300
        soc_info = NPUSocInfo()
        assert soc_info.soc_version == 300
        assert soc_info.need_nz is False
        assert soc_info.support_bf16 is True

    @patch("atb_llm.utils.initial.ENV")
    def test_communication_backend_lccl(self, mock_env):
        """Verify LCCL is selected when HCCL is disabled and LCCL is supported."""
        mock_env.hccl_enable = False
        soc_info = NPUSocInfo()
        soc_info.is_support_lccl = lambda: True
        assert soc_info.communication_backend == CommunicationLibrary.lccl

    @patch("atb_llm.utils.initial.ENV")
    def test_communication_backend_hccl(self, mock_env):
        """Verify HCCL is selected when HCCL is enabled or LCCL is not supported."""
        mock_env.hccl_enable = True
        soc_info = NPUSocInfo()
        soc_info.is_support_lccl = lambda: False
        assert soc_info.communication_backend == CommunicationLibrary.hccl

    @patch("atb_llm.utils.initial.execute_command")
    def test_is_support_hccs_true(self, mock_execute):
        mock_execute.return_value = "some info hccs Legend others"
        assert NPUSocInfo.is_support_hccs() is True

    @patch("atb_llm.utils.initial.execute_command")
    def test_is_support_hccs_false(self, mock_execute):
        mock_execute.return_value = "some info pcie Legend others"
        assert NPUSocInfo.is_support_hccs() is False

    @patch("atb_llm.utils.initial.ENV")
    @patch("atb_llm.utils.initial.execute_command")
    def test_is_support_lccl_true(self, mock_execute, mock_env):
        """Test LCCL support when topology includes HCCS and need_nz is False."""
        mock_execute.return_value = "some info hccs Legend others"
        mock_env.npu_vm_support_hccs = False
        soc_info = NPUSocInfo()
        soc_info.need_nz = False
        assert soc_info.is_support_lccl() is True

    @patch("atb_llm.utils.initial.ENV")
    @patch("atb_llm.utils.initial.execute_command")
    def test_is_support_lccl_false_by_need_nz(self, mock_execute, mock_env):
        """Test LCCL is not supported when need_nz is True."""
        mock_execute.return_value = "some info hccs Legend others"
        mock_env.npu_vm_support_hccs = False
        soc_info = NPUSocInfo()
        soc_info.need_nz = True
        assert soc_info.is_support_lccl() is False
        
    @patch("atb_llm.utils.initial.ENV")
    @patch("atb_llm.utils.initial.execute_command")
    def test_is_support_lccl_false_by_topo(self, mock_execute, mock_env):
        """Test LCCL is not supported when topology doesn't include HCCS."""
        mock_execute.return_value = "some info pcie Legend others"
        mock_env.npu_vm_support_hccs = False
        soc_info = NPUSocInfo()
        soc_info.need_nz = False
        assert soc_info.is_support_lccl() is False
        
    @patch("atb_llm.utils.initial.ENV")
    @patch("atb_llm.utils.initial.execute_command")
    def test_is_support_lccl_false_by_env(self, mock_execute, mock_env):
        """Test LCCL is not supported when environment variable is False."""
        mock_execute.return_value = "some info Legend others"
        mock_env.npu_vm_support_hccs = False
        
        soc_info = NPUSocInfo()
        soc_info.need_nz = False
        
        result = soc_info.is_support_lccl()
        
        assert result is False


class TestFunctions:
    @patch("atb_llm.utils.initial.ENV")
    @patch("atb_llm.utils.initial.sys.path")
    @patch("atb_llm.utils.initial.torch.classes.load_library")
    def test_load_atb_speed(self, mock_load_library, mock_sys_path, mock_env):
        """Verify ATB speed library is loaded correctly with proper path configuration."""
        # Configure test environment
        mock_env.atb_speed_home_path = "/test/path"
        
        load_atb_speed()
        
        # Verify library loading and path configuration
        mock_load_library.assert_called_once_with("/test/path/lib/libatb_speed_torch.so")
        mock_sys_path.append.assert_called_once_with("/test/path/lib")

    @patch("atb_llm.utils.initial.ENV")
    def test_is_lcoc_enable_true(self, mock_env):
        mock_env.lcoc_enable = True
        assert is_lcoc_enable(need_nz=False) is True

    @patch("atb_llm.utils.initial.ENV")
    def test_is_lcoc_enable_false_by_env(self, mock_env):
        mock_env.lcoc_enable = False
        assert is_lcoc_enable(need_nz=False) is False

    @patch("atb_llm.utils.initial.ENV")
    def test_is_lcoc_enable_false_by_need_nz(self, mock_env):
        """Verify LCOC is disabled when need_nz is True regardless of environment setting."""
        mock_env.lcoc_enable = True
        assert is_lcoc_enable(need_nz=True) is False