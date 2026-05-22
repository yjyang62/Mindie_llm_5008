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
from unittest.mock import patch, MagicMock, call

from mindie_llm.runtime.models import get_router_ins
from mindie_llm.runtime.config.load_config import LoadConfig


class TestGetRouterIns(unittest.TestCase):

    @patch('mindie_llm.runtime.models.importlib.import_module')
    @patch('mindie_llm.runtime.models.os.path.isdir')
    @patch('mindie_llm.runtime.models.safe_listdir')
    @patch('mindie_llm.runtime.models.os.path.abspath')
    @patch('mindie_llm.runtime.models.os.path.dirname')
    @patch('mindie_llm.runtime.models.safe_get_config_dict')
    @patch('mindie_llm.runtime.models.check_file_permission')
    def test_get_router_ins_success(self, *args):
        mock_check_file_permission, mock_safe_get_config_dict, \
        mock_dirname, mock_abspath, mock_safe_listdir, mock_isdir, mock_import_module = args

        mock_check_file_permission.return_value = None
        mock_safe_get_config_dict.return_value = {'model_type': "llama"}
        mock_safe_listdir.return_value = ["base", "llama", "_hidden", "invalid"]
        mock_dirname.return_value = "/current/path"
        mock_abspath.return_value = "/current/path"
        
        def isdir_side_effect(path):
            if path.endswith("llama") or path.endswith("base"):
                return True
            return False
        mock_isdir.side_effect = isdir_side_effect
        
        mock_router_class = MagicMock()
        mock_router_instance = MagicMock()
        mock_router_class.return_value = mock_router_instance
        
        mock_module = MagicMock()
        mock_module.LlamaRouter = mock_router_class
        mock_import_module.return_value = mock_module
        
        mock_import_module.reset_mock()
        
        load_config = LoadConfig(model_name_or_path="/standardized/path")
        
        result = get_router_ins(load_config)

        mock_check_file_permission.assert_called_once_with("/standardized/path")
        mock_safe_get_config_dict.assert_called_once_with("/standardized/path")
        mock_safe_listdir.assert_called_once_with("/current/path")
        
        expected_call_found = False
        for call_args in mock_import_module.call_args_list:
            if call_args[0] and call_args[0][0] == "mindie_llm.runtime.models.llama.router_llama":
                expected_call_found = True
                break
        
        self.assertTrue(expected_call_found)
        
        mock_router_class.assert_called_once_with(
            {'model_type': 'llama'}, 
            load_config
        )
        
        self.assertEqual(result, mock_router_instance)

    @patch('mindie_llm.runtime.models.importlib.import_module')
    @patch('mindie_llm.runtime.models.os.path.isdir')
    @patch('mindie_llm.runtime.models.safe_listdir')
    @patch('mindie_llm.runtime.models.os.path.abspath')
    @patch('mindie_llm.runtime.models.os.path.dirname')
    @patch('mindie_llm.runtime.models.safe_get_config_dict')
    @patch('mindie_llm.runtime.models.check_file_permission')
    def test_get_router_ins_invalid_model_type(self, *args):
        mock_check_file_permission, mock_safe_get_config_dict, \
        mock_dirname, mock_abspath, mock_safe_listdir, mock_isdir, mock_import_module = args

        mock_check_file_permission.return_value = None
        mock_safe_get_config_dict.return_value = {"model_type": "unknown"}
        mock_safe_listdir.return_value = ["base", "llama", "_hidden"]
        mock_dirname.return_value = "/current/path"
        mock_abspath.return_value = "/current/path"
        
        def isdir_side_effect(path):
            if path.endswith("llama") or path.endswith("base"):
                return True
            return False
        mock_isdir.side_effect = isdir_side_effect
        
        mock_import_module.reset_mock()
        
        load_config = LoadConfig(model_name_or_path="/standardized/path")
        
        with self.assertRaises(NotImplementedError) as cm:
            get_router_ins(load_config)
        
        self.assertIn("unsupported model type: unknown", str(cm.exception))
        self.assertIn("mindie_llm.runtime.models directory", str(cm.exception))

    @patch('mindie_llm.runtime.models.importlib.import_module')
    @patch('mindie_llm.runtime.models.os.path.isdir')
    @patch('mindie_llm.runtime.models.safe_listdir')
    @patch('mindie_llm.runtime.models.os.path.abspath')
    @patch('mindie_llm.runtime.models.os.path.dirname')
    @patch('mindie_llm.runtime.models.safe_get_config_dict')
    @patch('mindie_llm.runtime.models.check_file_permission')
    def test_get_router_ins_missing_model_type(self, *args):
        mock_check_file_permission, mock_safe_get_config_dict, \
        mock_dirname, mock_abspath, mock_safe_listdir, mock_isdir, mock_import_module = args

        mock_check_file_permission.return_value = None
        mock_safe_get_config_dict.return_value = {}
        mock_safe_listdir.return_value = ["base", "llama", "_hidden"]
        mock_dirname.return_value = "/current/path"
        mock_abspath.return_value = "/current/path"
        
        def isdir_side_effect(path):
            if path.endswith("llama") or path.endswith("base"):
                return True
            return False
        mock_isdir.side_effect = isdir_side_effect
        
        mock_import_module.reset_mock()
        
        load_config = LoadConfig(model_name_or_path="/standardized/path")
        
        with self.assertRaises(KeyError) as cm:
            get_router_ins(load_config)
        
        self.assertIn("'model_type'", str(cm.exception))


if __name__ == '__main__':
    unittest.main()
