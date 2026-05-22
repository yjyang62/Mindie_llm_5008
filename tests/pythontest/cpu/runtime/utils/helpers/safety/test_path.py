# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import unittest

from mindie_llm.runtime.utils.helpers.safety.path import (
    standardize_path,
    check_path_is_none,
    check_path_is_link,
    check_path_is_str,
    check_path_has_special_characters,
    check_path_length_lt
)


class TestPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.file_path = os.path.join(cls.test_dir, "test.txt")
        
        cls.symbolic_link = os.path.join(cls.test_dir, "link_to_test.txt")
        if os.path.lexists(cls.symbolic_link):
            os.unlink(cls.symbolic_link)
        os.symlink(cls.file_path, cls.symbolic_link)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.symbolic_link)

    def test_check_path_is_none_result_success(self):
        file_path = self.file_path
        check_path_is_none(file_path)

    def test_check_path_is_none_type_error(self):
        with self.assertRaises(TypeError):
            check_path_is_none(None)

    def test_check_path_is_link_result_success(self):
        file_path = self.file_path
        check_path_is_link(file_path)
    
    def test_check_path_is_link_result_value_error(self):
        with self.assertRaises(ValueError):
            check_path_is_link(self.symbolic_link)

    def test_check_path_is_str_result_success(self):
        file_path = self.file_path
        check_path_is_str(file_path)
    
    def test_check_path_is_str_result_type_error(self):
        with self.assertRaises(TypeError):
            check_path_is_str(56)
    
    def test_check_path_has_special_characters_result_success(self):
        file_path = self.file_path
        check_path_has_special_characters(file_path)
    
    def test_check_path_has_special_characters_result_value_error(self):
        with self.assertRaises(ValueError):
            check_path_has_special_characters("&*%*")
    
    def test_check_path_length_lt_result_success(self):
        file_path = self.file_path
        check_path_length_lt(file_path)

    def test_check_path_length_lt_result_value_error(self):
        with self.assertRaises(ValueError):
            check_path_length_lt(self.file_path, 5)

    def test_standardize_path(self):
        file_path = self.file_path
        result = standardize_path(file_path)
        self.assertEqual(file_path, result)


if __name__ == '__main__':
    unittest.main()
