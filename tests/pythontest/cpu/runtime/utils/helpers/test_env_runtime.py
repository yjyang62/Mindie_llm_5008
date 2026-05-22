# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
import os
import json
from unittest.mock import patch, mock_open


class TestEnvVarCheckRanktable(unittest.TestCase):
    """Test EnvVar check_rank_table method"""

    def setUp(self):
        """Setup before each test"""
        # Save original environment variables
        self.original_env = {}
        env_vars = ['MINDIE_LLM_FRAMEWORK_BACKEND']
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
        
    def tearDown(self):
        """Cleanup after each test"""
        # Restore original environment variables
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_check_rank_table_with_none_file(self):
        """Test check_rank_table with None file"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        env_var = EnvVar()
        
        # Should not raise any exception when rank_table_file is None
        try:
            env_var.check_rank_table(None)
        except Exception as e:
            self.fail(f"check_rank_table raised {e} unexpectedly!")

    def test_check_rank_table_with_empty_string(self):
        """Test check_rank_table with empty string"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        env_var = EnvVar()
        
        # Should not raise any exception when rank_table_file is empty string
        try:
            env_var.check_rank_table("")
        except Exception as e:
            self.fail(f"check_rank_table raised {e} unexpectedly!")
            
    @patch('mindie_llm.runtime.utils.helpers.safety.file.safe_open')
    def test_check_rank_table_with_valid_ranktable(self, mock_safe_open):
        """Test check_rank_table with valid ranktable"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Mock valid ranktable JSON
        valid_ranktable = {
            "server_list": [
                {
                    "server_id": "192.168.1.1",
                    "device": [
                        {"rank_id": "0", "device_ip": "192.168.1.10"},
                        {"rank_id": "1", "device_ip": "192.168.1.11"}
                    ]
                },
                {
                    "server_id": "192.168.1.2",
                    "device": [
                        {"rank_id": "2", "device_ip": "192.168.1.20"},
                        {"rank_id": "3", "device_ip": "192.168.1.21"}
                    ]
                }
            ]
        }
        
        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(valid_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value

        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        env_var = EnvVar()
        
        # Should not raise any exception with valid ranktable
        try:
            env_var.check_rank_table("/path/to/valid_ranktable.json")
        except Exception as e:
            self.fail(f"check_rank_table raised {e} unexpectedly!")

    @patch('mindie_llm.runtime.utils.helpers.safety.file.safe_open')
    def test_check_rank_table_with_invalid_rank_id(self, mock_safe_open):
        """Test check_rank_table with invalid rank_id (>= world_size)"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Mock ranktable with invalid rank_id
        invalid_ranktable = {
            "server_list": [
                {
                    "server_id": "192.168.1.1",
                    "device": [
                        {"rank_id": "0", "device_ip": "192.168.1.10"},
                        {"rank_id": "5", "device_ip": "192.168.1.11"}  # rank_id 5 >= world_size 2
                    ]
                }
            ]
        }
        
        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(invalid_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        env_var = EnvVar()
        
        # Should raise ValueError for invalid rank_id
        with self.assertRaises(ValueError) as context:
            env_var.check_rank_table("/path/to/invalid_ranktable.json")
        
        self.assertIn("must be less than world_size", str(context.exception))

    @patch('mindie_llm.runtime.utils.helpers.safety.file.safe_open')
    def test_check_rank_table_with_invalid_device_ip(self, mock_safe_open):
        """Test check_rank_table with invalid device IP"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Mock ranktable with invalid device IP
        invalid_ranktable = {
            "server_list": [
                {
                    "server_id": "192.168.1.1",
                    "device": [
                        {"rank_id": "0", "device_ip": "192.168.1.10"},
                        {"rank_id": "1", "device_ip": "invalid_ip"}  # Invalid IP
                    ]
                }
            ]
        }
        
        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(invalid_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        env_var = EnvVar()
        
        # Should raise ValueError for invalid device IP
        with self.assertRaises(ValueError) as context:
            env_var.check_rank_table("/path/to/invalid_ranktable.json")
        
        self.assertIn("Invalid device_ip", str(context.exception))

    @patch('mindie_llm.runtime.utils.helpers.safety.file.safe_open')
    def test_check_rank_table_with_invalid_server_id(self, mock_safe_open):
        """Test check_rank_table with invalid server_id IP"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Mock ranktable with invalid server_id
        invalid_ranktable = {
            "server_list": [
                {
                    "server_id": "invalid_server_ip",  # Invalid server IP
                    "device": [
                        {"rank_id": "0", "device_ip": "192.168.1.10"},
                        {"rank_id": "1", "device_ip": "192.168.1.11"}
                    ]
                }
            ]
        }
        
        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(invalid_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        env_var = EnvVar()
        
        # Should raise ValueError for invalid server_id
        with self.assertRaises(ValueError) as context:
            env_var.check_rank_table("/path/to/invalid_ranktable.json")
        
        self.assertIn("Invalid server_id", str(context.exception))

    @patch('mindie_llm.runtime.utils.helpers.safety.file.safe_open')
    def test_check_rank_table_with_complex_valid_ranktable(self, mock_safe_open):
        """Test check_rank_table with complex valid ranktable (multiple servers)"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Mock complex valid ranktable JSON
        complex_ranktable = {
            "server_list": [
                {
                    "server_id": "10.0.0.1",
                    "device": [
                        {"rank_id": "0", "device_ip": "10.0.0.10"},
                        {"rank_id": "1", "device_ip": "10.0.0.11"},
                        {"rank_id": "2", "device_ip": "10.0.0.12"}
                    ]
                },
                {
                    "server_id": "10.0.0.2",
                    "device": [
                        {"rank_id": "3", "device_ip": "10.0.0.20"},
                        {"rank_id": "4", "device_ip": "10.0.0.21"}
                    ]
                },
                {
                    "server_id": "10.0.0.3",
                    "device": [
                        {"rank_id": "5", "device_ip": "10.0.0.30"},
                        {"rank_id": "6", "device_ip": "10.0.0.31"},
                        {"rank_id": "7", "device_ip": "10.0.0.32"}
                    ]
                }
            ]
        }
        
        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(complex_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        env_var = EnvVar()
        
        # Should not raise any exception with complex valid ranktable
        try:
            env_var.check_rank_table("/path/to/complex_ranktable.json")
        except Exception as e:
            self.fail(f"check_rank_table raised {e} unexpectedly!")


class TestEnvVarIsValidIp(unittest.TestCase):
    """Test EnvVar is_valid_ip static method"""

    def test_is_valid_ip_with_valid_ipv4(self):
        """Test is_valid_ip with valid IPv4 addresses"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        
        # Test valid IPv4 addresses
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "127.0.0.1",
            "255.255.255.255"
        ]
        
        for ip in valid_ips:
            with self.subTest(ip=ip):
                self.assertTrue(EnvVar.is_valid_ip(ip))

    def test_is_valid_ip_with_valid_ipv6(self):
        """Test is_valid_ip with valid IPv6 addresses"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        
        # Test valid IPv6 addresses
        valid_ipv6s = [
            "::1",
            "2001:db8::1",
            "fe80::1",
            "::ffff:192.168.1.1"
        ]
        
        for ip in valid_ipv6s:
            with self.subTest(ip=ip):
                self.assertTrue(EnvVar.is_valid_ip(ip))

    def test_is_valid_ip_with_invalid_ips(self):
        """Test is_valid_ip with invalid IP addresses"""
        # Set valid framework backend to avoid initialization errors
        os.environ['MINDIE_LLM_FRAMEWORK_BACKEND'] = 'ATB'
        
        # Clear module cache to ensure reload
        import sys
        if 'mindie_llm.runtime.utils.helpers.env' in sys.modules:
            del sys.modules['mindie_llm.runtime.utils.helpers.env']
        
        from mindie_llm.runtime.utils.helpers.env import EnvVar
        
        # Test invalid IP addresses
        invalid_ips = [
            "invalid_ip",
            "256.256.256.256",
            "192.168.1",
            "192.168.1.1.1",
            "192.168.-1.1",
            "",
            "0.0.0.0",
            "localhost",
            "192.168.1.a"
        ]
        
        for ip in invalid_ips:
            with self.subTest(ip=ip):
                self.assertFalse(EnvVar.is_valid_ip(ip))

    def tearDown(self):
        """Cleanup after each test"""
        # Clean up environment variable
        if 'MINDIE_LLM_FRAMEWORK_BACKEND' in os.environ:
            del os.environ['MINDIE_LLM_FRAMEWORK_BACKEND']

if __name__ == '__main__':
    unittest.main()
