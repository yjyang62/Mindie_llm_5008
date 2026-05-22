#!/usr/bin/env python
# coding=utf-8
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
import sys
import tempfile
import shutil

# Add the root directory to the Python path so we can import modules
# This follows the pattern used in other test files which import from atb_llm
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../../.."))

# Import the decrypt_code_padding module from tests.encrypttest
# We need to adjust the import path since we're running from a different location
# than where the script was originally located
import tests.encrypttest.decrypt_code_padding as decrypt_code_padding

patch_operations = decrypt_code_padding.patch_operations


class TestDecryptCodePadding(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)
        
    def test_process_file_all_operations_successful(self):
        """
        Test that the function only passes when all target content matches and modifications are successful
        """
        # Create a complete test file that matches all operations
        test_file_path = os.path.join(self.test_dir, "test_file.py")
        with open(test_file_path, 'w') as f:
            f.write("# Part of this file was copied from project text-generation-inference 0.9.1\n")
            f.write("from typing import Tuple, Any\n")
            f.write("\n")
            f.write("from safetensors import safe_open\n")
            f.write("\n")
            f.write("from .. import file_utils\n")
            f.write("from .file_loader import get_weight_filenames, BaseFileLoader\n")
            f.write("\n")
            f.write("\n")
            f.write("class SafetensorFileLoader(BaseFileLoader):\n")
            f.write("    def __init__(self, model_weight_path: str, device: Any, mapping=None):\n")
            f.write("        super().__init__(model_weight_path)\n")
            f.write("        self._filenames = get_weight_filenames(self.model_weight_path, self.extension)\n")
            f.write("        self._routing = self._load_weight_file_routing()\n")
            f.write("        self.mapping = mapping\n")
            f.write("        self.device = device\n")
            f.write("\n")
            f.write("    def get_tensor(self, tensor_name: str) -> Any:\n")
            f.write("        filename, tensor_name = self.get_filename(tensor_name)\n")
            f.write("        f = self.get_handler(filename)\n")
            f.write("        tensor = f.get_tensor(tensor_name)\n")
            f.write("        return tensor\n")
            f.write("\n")
            f.write("    def get_sharded(self, tensor_name: str, dim: int, chunk_id: int, num_chunk: int) -> Any:\n")
            f.write("        if dim not in [0, 1]:\n")
            f.write("            raise AssertionError(f\"Dimension {dim} is invalid in `get_sharded`.\")\n")
            f.write("        slice_ = self._get_slice(tensor_name)\n")
            f.write("        group_size = slice_.get_shape()[dim]\n")
            f.write("\n")
            f.write("    def _load_weight_file_routing(self) -> dict:\n")
            f.write("        routing = {}\n")
            f.write("        for filename in self._filenames:\n")
            f.write("            filename = file_utils.standardize_path(str(filename), check_link=False)\n")
            f.write("            file_utils.check_path_permission(filename)\n")
            f.write("            with safe_open(filename, framework=\"pytorch\") as f:\n")
            f.write("                for k in f.keys():\n")
            f.write("                    if k in routing:\n")
            f.write("                        raise AssertionError(f\"Weight was found in multiple files.\")\n")
            f.write("                    routing[k] = filename\n")
            f.write("        return routing\n")
            
        # Process the file
        result = decrypt_code_padding.process_file(test_file_path, patch_operations)
        
        # Check that all modifications were made successfully
        self.assertTrue(result, "The decrypt_code_padding script should successfully process the file when all target content matches")
        
        # Verify file content was modified with all expected changes
        with open(test_file_path, 'r') as f:
            content = f.read()
            
        # Check for inserted import
        self.assertIn("import importlib", content, "The script should add 'import importlib' to the file")
        
        # Check for inserted encrypt_enable property
        self.assertIn("def encrypt_enable(self):", content, "The script should add the encrypt_enable property")
        
        # Check for inserted sf_metadata initialization
        self.assertIn("self.sf_metadata = {}", content, "The script should initialize self.sf_metadata")
        
        # Check for inserted decrypt logic in get_tensor (verify the new code block is there)
        self.assertIn("if self.encrypt_enable:", content, "The script should add encryption check in get_tensor method")
        self.assertIn("tensor = self.decrypt_ins.decrypt(tensor)", content, "The script should add decryption logic")
        
        # Check for inserted metadata update
        self.assertIn("self.sf_metadata.update(f.metadata())", content, "The script should add metadata update code")
        
        # Verify backup file was created
        backup_file_path = test_file_path + ".bak"
        self.assertTrue(os.path.exists(backup_file_path), "The script should create a backup file")
        
    def test_process_real_safetensor_file_loader(self):
        """
        Test that the function works correctly with the real safetensor_file_loader.py
        """
        # Get the path to the real safetensor_file_loader.py
        import atb_llm.utils.loader.safetensor_file_loader as real_loader
        real_file_path = real_loader.__file__
        
        # Verify the file exists
        self.assertTrue(os.path.exists(real_file_path), "The real safetensor_file_loader.py file should exist")
        
        # Create a backup of the original file with a unique name to avoid conflicts
        test_backup_path = real_file_path + ".test_backup"
        shutil.copy2(real_file_path, test_backup_path)
        
        # Store the original backup file path to check if it existed before
        original_backup_path = real_file_path + ".bak"
        backup_existed_before = os.path.exists(original_backup_path)
        
        try:
            # Process the real file
            result = decrypt_code_padding.process_file(real_file_path, patch_operations)
            
            # Check that modifications were made
            self.assertTrue(result, "The decrypt_code_padding script should successfully process the real safetensor_file_loader.py file")
            
            # Verify the file was modified correctly
            with open(real_file_path, 'r') as f:
                content = f.read()
                
            # Check for key modifications
            self.assertIn("import importlib", content, "The script should add 'import importlib' to the real file")
            self.assertIn("def encrypt_enable(self):", content, "The script should add the encrypt_enable property to the real file")
            self.assertIn("self.sf_metadata = {}", content, "The script should initialize self.sf_metadata in the real file")
            self.assertIn("if self.encrypt_enable:", content, "The script should add encryption check in get_tensor method in the real file")
            self.assertIn("tensor = self.decrypt_ins.decrypt(tensor)", content, "The script should add decryption logic to the real file")
            self.assertIn("self.sf_metadata.update(f.metadata())", content, "The script should add metadata update code to the real file")
            
            # Verify backup file was created
            script_backup_path = real_file_path + ".bak"
            self.assertTrue(os.path.exists(script_backup_path), "The script should create a backup file for the real file")
            
        finally:
            # Restore the original file from our test backup
            shutil.move(test_backup_path, real_file_path)
            
            # Only remove the backup file created by the script if it didn't exist before
            # This prevents us from deleting backups created by other processes
            script_backup_path = real_file_path + ".bak"
            if os.path.exists(script_backup_path) and not backup_existed_before:
                os.remove(script_backup_path)


if __name__ == '__main__':
    unittest.main()