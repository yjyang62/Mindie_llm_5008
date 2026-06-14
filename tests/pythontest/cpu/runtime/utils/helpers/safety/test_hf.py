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
from unittest.mock import patch, MagicMock
from mindie_llm.runtime.utils.helpers.safety.hf import (
    safe_get_tokenizer_from_pretrained,
    safe_get_config_dict
)


class TestHFUtils(unittest.TestCase):

    def test_tokenizer_success(self):
        """Verify normal tokenizer loading flow."""
        with patch('mindie_llm.runtime.utils.helpers.safety.hf.standardize_path') as mock_std, \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.check_file_permission') as mock_chk, \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.AutoTokenizer.from_pretrained') as mock_prep:
            
            mock_std.return_value = "valid_path"
            mock_chk.return_value = None
            mock_prep.return_value = MagicMock()
            
            result = safe_get_tokenizer_from_pretrained("test_path")
            self.assertIsNotNone(result)
            
            mock_std.assert_called_once_with("test_path", check_link=False)
            mock_chk.assert_called_once_with("valid_path")
            mock_prep.assert_called_once_with("valid_path", local_files_only=True)

    def test_config_dict_success(self):
        """Verify normal config dict loading flow."""
        with patch('mindie_llm.runtime.utils.helpers.safety.hf.standardize_path') as mock_std, \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.check_file_permission') as mock_chk, \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.PretrainedConfig.get_config_dict') as mock_cfg:
            
            mock_std.return_value = "valid_path"
            mock_chk.return_value = None
            mock_cfg.return_value = (MagicMock(), None)
            
            result = safe_get_config_dict("test_path")
            self.assertIsNotNone(result)
            
            mock_std.assert_called_once_with("test_path", check_link=False)
            mock_chk.assert_called_once_with("valid_path")
            mock_cfg.assert_called_once_with("valid_path", local_files_only=True)

    def test_tokenizer_exception(self):
        with patch('mindie_llm.runtime.utils.helpers.safety.hf.AutoTokenizer.from_pretrained', side_effect=EnvironmentError("original_error")), \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.standardize_path'), \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.check_file_permission'), \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.logger.error') as mock_logger:
            with self.assertRaises(EnvironmentError) as cm:
                safe_get_tokenizer_from_pretrained("test_path")
            mock_logger.assert_called_once()
            self.assertIn("Please check the input parameters", str(cm.exception))
                
    def test_config_dict_exception(self):
        with patch('mindie_llm.runtime.utils.helpers.safety.hf.PretrainedConfig.get_config_dict', side_effect=ValueError("original_error")), \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.standardize_path'), \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.check_file_permission'), \
             patch('mindie_llm.runtime.utils.helpers.safety.hf.logger.error') as mock_logger:
            with self.assertRaises(ValueError) as cm:
                safe_get_config_dict("test_path")
            mock_logger.assert_called_once()
            self.assertIn("Please check the input parameters", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
