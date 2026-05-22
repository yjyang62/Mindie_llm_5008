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
import shutil
import inspect
from typing import get_type_hints


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../../.."))


class TestDecryptCodePaddingV2(unittest.TestCase):
    """
    Test suite for verifying decrypt_code_padding_v2 functionality:
    1. Code modification operations work correctly
    2. CryptedWeightsFileHandler maintains interface compatibility with WeightsFileHandler
    """

    def test_process_real_safetensor_file_loader(self):
        """
        Test that the function works correctly with the real weight_utils.py
        """
        import tests.encrypttest.crypted_weights_handler as real_loader
        import tests.encrypttest.decrypt_code_padding_v2 as dp_code
        patch_operations = dp_code.operations
        real_file_path = real_loader.__file__

        # Verify the file exists
        self.assertTrue(os.path.exists(real_file_path), "The real weight_utils.py file should exist")

        # Create a backup of the original file with a unique name to avoid conflicts
        test_backup_path = real_file_path + ".test_backup"
        shutil.copy2(real_file_path, test_backup_path)

        # Store the original backup file path to check if it existed before
        original_backup_path = real_file_path + ".bak"
        backup_existed_before = os.path.exists(original_backup_path)

        try:
            # Process the real file
            result = dp_code.process_file(real_file_path, patch_operations)

            # Check that modifications were made
            self.assertTrue(result, "The decrypt_code_padding_v2 script should successfully process the real weight_utils.py file")

            # Verify the file was modified correctly
            with open(real_file_path, 'r') as f:
                content = f.read()

            # Check for modifications
            for i in patch_operations:
                for operation, code in i.items():
                    if operation != "insert":
                        continue
                    self.assertIn(code, content, f"The script should add specific code to the real file, but it didn't. Failed code is: \n {code}")

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

    def test_crypted_handler_inherits_from_base(self):
        """
        Test that CryptedWeightsFileHandler properly inherits from WeightsFileHandler
        """
        from mindie_llm.runtime.utils.loader.weight_utils import WeightsFileHandler
        from tests.encrypttest.decrypt_code_padding_v2 import CryptedWeightsFileHandler

        self.assertTrue(issubclass(CryptedWeightsFileHandler, WeightsFileHandler),
                       "CryptedWeightsFileHandler should be a subclass of WeightsFileHandler")

    def test_interface_compatibility(self):
        """
        Test that CryptedWeightsFileHandler implements all public methods from WeightsFileHandler
        with compatible signatures
        """
        from mindie_llm.runtime.utils.loader.weight_utils import WeightsFileHandler
        from tests.encrypttest.decrypt_code_padding_v2 import CryptedWeightsFileHandler

        # Get all public methods from base class
        base_methods = {name: (method, get_type_hints(method))
                       for name, method in inspect.getmembers(WeightsFileHandler, predicate=inspect.isfunction)
                       if not name.startswith('_')}

        # Get all public methods from derived class
        derived_methods = {name: (method, get_type_hints(method))
                          for name, method in inspect.getmembers(CryptedWeightsFileHandler, predicate=inspect.isfunction)
                          if not name.startswith('_')}

        # Check that all base public methods are present in derived class
        for method_name in base_methods:
            self.assertIn(method_name, derived_methods,
                         f"CryptedWeightsFileHandler should implement method '{method_name}' from WeightsFileHandler")

            # Check signature compatibility
            base_method, _ = base_methods[method_name]
            derived_method, _ = derived_methods[method_name]

            base_sig = inspect.signature(base_method)
            derived_sig = inspect.signature(derived_method)

            # Compare parameter names (excluding 'self')
            base_params = [p for p in base_sig.parameters.keys() if p != 'self']
            derived_params = [p for p in derived_sig.parameters.keys() if p != 'self']

            self.assertEqual(base_params, derived_params,
                           f"Method '{method_name}' should have the same parameters as base class")


if __name__ == '__main__':
    unittest.main()