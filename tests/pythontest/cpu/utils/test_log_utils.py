# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import argparse
from mindie_llm.utils.log.utils import update_log_file_param, \
    get_component_config, create_log_dir_and_check_permission


class TestCreateLogDirAndCheckPermission(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.log")
        
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_directory_path_raises_error(self):
        """Test error thrown when input is a directory path"""
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            create_log_dir_and_check_permission(self.temp_dir)
        self.assertIn("only supports paths that end with a file", str(cm.exception))

    def test_existing_file_permissions(self):
        """Test permission check for existing files"""
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write("test")
        
        # Set initial permissions
        os.chmod(self.test_file, 0o640)
        
        # File should exist and have correct permissions
        self.assertTrue(os.path.exists(self.test_file))
        
        # Call function (Note: this function won't modify permissions of existing files)
        create_log_dir_and_check_permission(self.test_file)
        
        # Verify file permissions remain unchanged
        actual_mode = os.stat(self.test_file).st_mode & 0o777
        self.assertEqual(actual_mode, 0o640)
        
        # Or use more detailed permission check
        st = os.stat(self.test_file)
        self.assertEqual(st.st_mode & 0o777, 0o640)

    def test_existing_dir_with_write_permission(self):
        """Test case with existing directory and write permissions"""
        # 1. Clean up any existing files and directories
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        dir_path = os.path.dirname(self.test_file)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        
        # 2. Create directory and set permissions
        os.makedirs(dir_path, mode=0o750)
        
        # 3. Verify directory permissions
        actual_mode = os.stat(dir_path).st_mode & 0o777
        self.assertEqual(actual_mode, 0o750,
                        f"Directory permissions {oct(actual_mode)} do not match expected {oct(0o750)}")
        
        # 4. Verify write permissions
        self.assertTrue(os.access(dir_path, os.W_OK),
                    "Directory should have write permission")
        
        # 5. Execute test function
        try:
            create_log_dir_and_check_permission(self.test_file)
        except Exception as e:
            self.fail(f"create_log_dir_and_check_permission raised {e} unexpectedly!")
        
        # 6. Verify directory still exists
        self.assertTrue(os.path.exists(dir_path),
                    "Directory should still exist after function call")
        
        # 7. Verify directory permissions unchanged
        actual_mode_after = os.stat(dir_path).st_mode & 0o777
        self.assertEqual(actual_mode_after, 0o750,
                        f"Directory permissions changed from {oct(0o750)} to {oct(actual_mode_after)}")

    def test_create_new_directory(self):
        """Test case of creating new directory"""
        new_path = os.path.join(self.temp_dir, "new_dir", "test.log")
        dir_path = os.path.dirname(new_path)
        
        # Directory should not exist
        self.assertFalse(os.path.exists(dir_path))
        
        # Create directory
        create_log_dir_and_check_permission(new_path)
        
        # Verify directory created with correct permissions
        self.assertTrue(os.path.exists(dir_path))
        self.assertEqual(os.stat(dir_path).st_mode & 0o777, 0o750)

    def test_create_nested_directories(self):
        """Test creation of nested directories"""
        nested_path = os.path.join(self.temp_dir, "a", "b", "c", "test.log")
        
        create_log_dir_and_check_permission(nested_path)
        
        # Verify all directories are created
        self.assertTrue(os.path.exists(os.path.dirname(nested_path)))


class TestStandardEnv(unittest.TestCase):
    """Unit test class for standard environment variable configuration"""
    
    def setUp(self):
        """Pre-test setup"""
        return super().setUp()
    
    def tearDown(self):
        """Post-test cleanup"""
        return super().tearDown()
    
    def test_get_component_config_with_1(self):
        """Test case with single value '1'"""
        env_variable = "1"
        env_variable = get_component_config(env_variable, "llm")
        self.assertEqual(env_variable, "1")
    
    def test_get_component_config_with_llm0(self):
        """Test case with llm component configured as '0'"""
        env_variable = "llm:0"
        env_variable = get_component_config(env_variable, "llm")
        self.assertEqual(env_variable, "0")

    def test_get_component_config_with_server0(self):
        """Test case with server component configured as '0' but querying llm component"""
        env_variable = "server:0"
        env_variable = get_component_config(env_variable, "llm")
        self.assertEqual(env_variable, "")  # Should return empty string as no llm config exists

    def test_get_component_config_with_servererrorllminfo(self):
        """Test getting llm component configuration from multiple component configurations"""
        env_variable = "server:error;llm:info"
        env_variable = get_component_config(env_variable, "llm")
        self.assertEqual(env_variable, "info")

    def test_get_component_config_with_servererrorllminfollmwarn(self):
        """Test returning last configuration when component configurations are duplicated"""
        env_variable = "server:error;llm:info;llm:warn"
        env_variable = get_component_config(env_variable, "llm")
        self.assertEqual(env_variable, "warn")  # Should return the last llm configuration

    def test_get_component_config_with_servererrorllminfodebug(self):
        """Test getting default value from mixed configuration"""
        env_variable = "server:error;llm:info;debug"
        env_variable = get_component_config(env_variable, "llm")
        self.assertEqual(env_variable, "debug")  # Should return the last value

    def test_get_component_config_with_infollmdebug(self):
        """Test case with spaces in configuration"""
        env_variable = "info;llm : debug"
        env_variable = get_component_config(env_variable, "llm")
        self.assertEqual(env_variable, "debug")  # Should handle spaces correctly

    def test_get_component_config_empty_config(self):
        """Test case with empty configuration"""
        result = get_component_config("", "llm")
        self.assertEqual(result, "")

    def test_get_component_config_malformed(self):
        """Test case with malformed configuration"""
        env_variable = "invalid_format"
        result = get_component_config(env_variable, "llm")
        self.assertEqual(result, "invalid_format")  # Should return the entire string

    def test_get_component_config_special_chars(self):
        """Test case with special characters in configuration"""
        env_variable = "llm:DEBUG-INFO;server:ERROR"
        result = get_component_config(env_variable, "llm")
        self.assertEqual(result, "DEBUG-INFO")  # Should preserve special characters


class TestUpdateLogFileParam(unittest.TestCase):
    def setUp(self):
        """Setup before each test case"""
        self.default_size = 20 * 1024 * 1024
        self.default_files = 10

    def test_empty_config(self):
        """Test empty configuration string"""
        result = update_log_file_param("")
        self.assertEqual(result, (self.default_size, self.default_files))

    def test_valid_file_size_config(self):
        """Test valid file size configuration"""
        result = update_log_file_param("-fs 100")
        self.assertEqual(result, (100 * 1024 * 1024, self.default_files))

    def test_valid_rotation_config(self):
        """Test valid rotation count configuration"""
        result = update_log_file_param("-r 5")
        self.assertEqual(result, (self.default_size, 5))

    def test_valid_both_config(self):
        """Test both file size and rotation count configuration"""
        result = update_log_file_param("-fs 50 -r 8")
        self.assertEqual(result, (50 * 1024 * 1024, 8))

    def test_invalid_file_size_zero(self):
        """Test file size of 0"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-fs 0")
        self.assertIn("should be between 1 and 500 MB", str(context.exception))
        self.assertIn("but got 0 MB", str(context.exception))

    def test_invalid_file_size_negative(self):
        """Test negative file size"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-fs -1")
        self.assertIn("should be between 1 and 500 MB", str(context.exception))
        self.assertIn("but got -1 MB", str(context.exception))

    def test_invalid_file_size_too_large(self):
        """Test file size too large"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-fs 501")
        self.assertIn("should be between 1 and 500 MB", str(context.exception))
        self.assertIn("but got 501 MB", str(context.exception))

    def test_invalid_rotation_zero(self):
        """Test rotation count of 0"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-r 0")
        self.assertIn("should be between 1 and 64", str(context.exception))
        self.assertIn("but got 0", str(context.exception))

    def test_invalid_rotation_negative(self):
        """Test negative rotation count"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-r -1")
        self.assertIn("should be between 1 and 64", str(context.exception))
        self.assertIn("but got -1", str(context.exception))

    def test_invalid_rotation_too_large(self):
        """Test rotation count too large"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-r 65")
        self.assertIn("should be between 1 and 64", str(context.exception))
        self.assertIn("but got 65", str(context.exception))

    def test_non_numeric_file_size(self):
        """Test non-numeric file size"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-fs abc")
        self.assertIn("should be an integer", str(context.exception))
        self.assertIn("but got 'abc'", str(context.exception))

    def test_non_numeric_rotation(self):
        """Test non-numeric rotation count"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-r xyz")
        self.assertIn("should be an integer", str(context.exception))
        self.assertIn("but got 'xyz'", str(context.exception))

    def test_incomplete_file_size_config(self):
        """Test incomplete file size configuration"""
        result = update_log_file_param("-fs")
        self.assertEqual(result, (self.default_size, self.default_files))

    def test_incomplete_rotation_config(self):
        """Test incomplete rotation configuration"""
        result = update_log_file_param("-r")
        self.assertEqual(result, (self.default_size, self.default_files))

    def test_invalid_option(self):
        """Test invalid option"""
        result = update_log_file_param("-x 100 -y 10")
        self.assertEqual(result, (self.default_size, self.default_files))

    def test_boundary_values(self):
        """Test boundary values"""
        # Test minimum values
        result = update_log_file_param("-fs 1 -r 1")
        self.assertEqual(result, (1 * 1024 * 1024, 1))
        
        # Test maximum values
        result = update_log_file_param("-fs 500 -r 64")
        self.assertEqual(result, (500 * 1024 * 1024, 64))

    def test_multiple_configs(self):
        """Test multiple configurations, using only the last valid value"""
        result = update_log_file_param("-fs 100 -fs 200 -r 5 -r 10")
        self.assertEqual(result, (200 * 1024 * 1024, 10))

    def test_config_with_extra_spaces(self):
        """Test configuration with extra spaces"""
        result = update_log_file_param("  -fs  100  -r  5  ")
        self.assertEqual(result, (100 * 1024 * 1024, 5))

    def test_config_with_newlines(self):
        """Test configuration with newline characters"""
        result = update_log_file_param("-fs 100\n-r 5")
        self.assertEqual(result, (100 * 1024 * 1024, 5))

    def test_config_with_tabs(self):
        """Test configuration with tab characters"""
        result = update_log_file_param("-fs\t100\t-r\t5")
        self.assertEqual(result, (100 * 1024 * 1024, 5))

    @patch('builtins.print')
    def test_error_messages(self, mock_print):
        """Test error message format"""
        with self.assertRaises(ValueError):
            update_log_file_param("-fs 0")
        
        # Verify error message format
        try:
            update_log_file_param("-fs 0")
        except ValueError as e:
            error_msg = str(e)
            self.assertIn("Log file size (-fs)", error_msg)
            self.assertIn("should be between 1 and 500 MB", error_msg)
            self.assertIn("but got 0 MB", error_msg)

    def test_large_file_size(self):
        """Test handling of large file size"""
        result = update_log_file_param("-fs 500")
        self.assertEqual(result, (500 * 1024 * 1024, self.default_files))

    def test_large_rotation_count(self):
        """Test handling of large rotation count"""
        result = update_log_file_param("-r 64")
        self.assertEqual(result, (self.default_size, 64))


if __name__ == '__main__':
    unittest.main()
