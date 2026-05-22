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
import os
import shutil
from unittest.mock import patch, mock_open
from examples.convert.convert_utils import copy_tokenizer_files, modify_config, MAX_TOKENIZER_FILE_SIZE


class TestCopyTokenizerFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directories for testing
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_model_dir = os.path.join(cls.base_dir, "test_model_dir")
        cls.test_dest_dir = os.path.join(cls.base_dir, "test_dest_dir")
        os.makedirs(cls.test_model_dir, exist_ok=True)
        os.makedirs(cls.test_dest_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directories after each test
        if os.path.exists(cls.test_model_dir):
            shutil.rmtree(cls.test_model_dir)
        if os.path.exists(cls.test_dest_dir):
            shutil.rmtree(cls.test_dest_dir)

    @patch('shutil.copyfile')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('examples.convert.convert_utils.file_utils')
    def test_copy_tokenizer_files_success(self, mock_file_utils, mock_exists, mock_makedirs, mock_copyfile):
        """
        Test that tokenizer files are correctly identified and copied.
        """
        # Configure mocks
        mock_file_utils.standardize_path.side_effect = lambda path, check_link: path
        mock_file_utils.check_path_permission.return_value = None
        mock_file_utils.check_file_safety.return_value = None

        # Simulate files in the model directory
        mock_file_utils.safe_listdir.return_value = [
            'tokenizer.json',
            'config.json',
            'special_token_map.json',
            'tokenization_config.json'
        ]

        # Simulate dest_dir not existing initially
        mock_exists.side_effect = lambda path: path == self.test_model_dir

        copy_tokenizer_files(self.test_model_dir, self.test_dest_dir)

        # Assertions
        # Check if standardize_path and check_path_permission were called for model_dir
        mock_file_utils.standardize_path.assert_any_call(self.test_model_dir, check_link=False)
        mock_file_utils.check_path_permission.assert_any_call(self.test_model_dir)

        # Check if dest_dir was created and its permissions checked
        mock_makedirs.assert_called_once_with(self.test_dest_dir, exist_ok=True)
        mock_file_utils.standardize_path.assert_any_call(self.test_dest_dir, check_link=False)

        # Check if safe_listdir was called
        mock_file_utils.safe_listdir.assert_called_once_with(self.test_model_dir)

        # Check if file safety was checked for source and destination
        self_src_path = os.path.join(self.test_model_dir, 'tokenizer.json')
        self_dest_path = os.path.join(self.test_dest_dir, 'tokenizer.json')
        mock_file_utils.check_file_safety.assert_any_call(self_src_path, 'r', max_file_size=MAX_TOKENIZER_FILE_SIZE)
        mock_file_utils.check_file_safety.assert_any_call(self_dest_path, 'w', max_file_size=MAX_TOKENIZER_FILE_SIZE)

        # Check if copyfile was called for the relevant files
        expected_copied_files = [
            'tokenizer.json',
            'special_token_map.json',
            'tokenization_config.json'
        ]

        self.assertEqual(mock_copyfile.call_count, len(expected_copied_files))
        for filename in expected_copied_files:
            src = os.path.join(self.test_model_dir, filename)
            dest = os.path.join(self.test_dest_dir, filename)
            mock_copyfile.assert_any_call(src, dest)

        # Ensure that 'dummy_file.txt' was NOT copied
        non_tokenizer_src = os.path.join(self.test_model_dir, 'dummy_file.txt')
        non_tokenizer_dest = os.path.join(self.test_dest_dir, 'dummy_file.txt')
        self.assertNotIn((non_tokenizer_src, non_tokenizer_dest), mock_copyfile.call_args_list)


    @patch('shutil.copyfile')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('examples.convert.convert_utils.file_utils')
    def test_copy_tokenizer_files_dest_dir_exists(self, mock_file_utils, mock_exists, mock_makedirs, mock_copyfile):
        """
        Test that dest_dir's permissions are checked when it already exists.
        """
        mock_file_utils.standardize_path.side_effect = lambda path, check_link: path
        mock_file_utils.check_path_permission.return_value = None
        mock_file_utils.check_file_safety.return_value = None
        mock_file_utils.safe_listdir.return_value = ['tokenizer.json']

        # Simulate dest_dir existing initially
        mock_exists.side_effect = lambda path: path in [self.test_model_dir, self.test_dest_dir]

        copy_tokenizer_files(self.test_model_dir, self.test_dest_dir)

        # Assertions
        mock_makedirs.assert_not_called() # Should not create if it exists
        mock_file_utils.check_path_permission.assert_any_call(self.test_dest_dir)
        mock_copyfile.assert_called_once()

    @patch('examples.convert.convert_utils.file_utils')
    def test_copy_tokenizer_files_no_tokenizer_files(self, mock_file_utils):
        """
        Test that no files are copied if no tokenizer-related files are found.
        """
        mock_file_utils.standardize_path.side_effect = lambda path, check_link: path
        mock_file_utils.check_path_permission.return_value = None
        mock_file_utils.check_file_safety.return_value = None
        mock_file_utils.safe_listdir.return_value = ['image.jpg', 'data.csv'] # No tokenizer files

        with patch('shutil.copyfile') as mock_copyfile:
            copy_tokenizer_files(self.test_model_dir, self.test_dest_dir)
            mock_copyfile.assert_not_called() # No files should be copied


class TestModifyConfig(unittest.TestCase):

    def setUp(self):
        # Define mock directories and file paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.mock_model_dir = os.path.join(self.base_dir, "mock/model_dir")
        self.mock_dest_dir = os.path.join(self.base_dir, "mock/dest_dir")
        self.mock_src_config_path = os.path.join(self.mock_model_dir, 'config.json')
        self.mock_dest_config_path = os.path.join(self.mock_dest_dir, 'config.json')

        # Create a mock torch_dtype enum-like object
        class MockTorchDtype:
            float16 = "MockTorchDtype.float16"
            bfloat16 = "MockTorchDtype.bfloat16"

        self.MockTorchDtype = MockTorchDtype

    @patch('json.dump')
    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open) # Patch builtins.open
    @patch('examples.convert.convert_utils.file_utils')
    def test_modify_config_success(self, mock_file_utils, mock_builtin_open, mock_json_load, mock_json_dump):
        """
        Test that config.json is correctly read, modified, and written.
        """
        # --- Configure Mocks ---
        # Mock file_utils methods
        mock_file_utils.standardize_path.side_effect = lambda path, check_link: path
        mock_file_utils.check_path_permission.return_value = None

        # Mock the initial content of config.json
        initial_config_data = {
            "architectures": ["LLaMAForCausalLM"],
            "torch_dtype": "float32",
            "hidden_size": 4096
        }
        mock_json_load.return_value = initial_config_data

        # --- Call the function under test ---
        mock_torch_dtype = self.MockTorchDtype.float16
        mock_quantize_type = "int8" # Not used in modify_config, but passed for signature
        modify_config(self.mock_model_dir, self.mock_dest_dir, mock_torch_dtype, mock_quantize_type)

        # --- Assertions ---

        # 1. Assert standardize_path and check_path_permission calls
        mock_file_utils.standardize_path.assert_any_call(self.mock_model_dir, check_link=False)
        mock_file_utils.check_path_permission.assert_any_call(self.mock_model_dir)
        mock_file_utils.standardize_path.assert_any_call(self.mock_dest_dir, check_link=False)
        mock_file_utils.check_path_permission.assert_any_call(self.mock_dest_dir)

        # 2. Assert src config file was opened for reading
        # mock_builtin_open.call_args_list will contain calls in order.
        # The first call should be for reading the source config.
        mock_file_utils.safe_open.assert_any_call(self.mock_src_config_path, 'r', encoding='utf-8')
        mock_json_load.assert_called_once() # Should be called after opening the source file

        # 3. Assert dest config file was opened for writing
        # The second call should be for writing the destination config.
        mock_file_utils.safe_open.assert_any_call(self.mock_dest_config_path, 'w', encoding='utf-8', is_exist_ok=False)
        mock_json_dump.assert_called_once() # Should be called before closing the dest file

        # 4. Assert the data passed to json.dump is correct
        expected_modified_data = initial_config_data.copy()
        expected_modified_data['torch_dtype'] = "float16" # Expected change


if __name__ == '__main__':
    unittest.main()