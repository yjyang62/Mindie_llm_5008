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
from unittest.mock import patch, call, Mock
from pathlib import Path

# 待测函数
from examples.convert.convert_weights import (
    parse_arguments,
    convert_bin2st,
    convert_bin2st_from_pretrained
)


class TestConvertWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path'])
    def test_parse_arguments(self):
        args = parse_arguments()
        self.assertEqual(args.model_path, 'fake_model_path')
    
    @patch("examples.convert.convert_weights.convert_files")
    @patch("examples.convert.convert_weights.weight_files")
    def test_convert_bin2st(self, mock_weight_files, mock_convert_files):
        model_path = 'fake_model_path'
        mock_local_pt_files = [Path('pytorch_model_00001_of_00003.bin'),
                               Path('pytorch_model_00002_of_00003.bin'),
                               Path('pytorch_model_00003_of_00003.bin')]
        mock_local_st_files = [Path('model_00001_of_00003.safetensors'),
                               Path('model_00002_of_00003.safetensors'),
                               Path('model_00003_of_00003.safetensors')]
        mock_weight_files.return_value = mock_local_pt_files

        convert_bin2st(model_path)

        mock_convert_files.assert_called_once_with(mock_local_pt_files, mock_local_st_files, discard_names=[])
        mock_weight_files.assert_has_calls([call(model_path, extension=".bin"), call(model_path)], any_order=False)
        self.assertEqual(mock_weight_files.call_count, 2)
    
    @patch("examples.convert.convert_weights.safe_get_model_from_pretrained")
    def test_convert_bin2st_from_pretrained(self, mock_safe_get_model_from_pretrained):
        model_path = 'fake_model_path'
        mock_model = Mock()
        mock_model.save_pretrained = Mock()
        mock_safe_get_model_from_pretrained.return_value = mock_model

        convert_bin2st_from_pretrained(model_path)
        
        mock_safe_get_model_from_pretrained.assert_called_once_with(model_path,
                                                                    low_cpu_mem_usage=True,
                                                                    torch_dtype="auto")
        mock_model.save_pretrained.assert_called_once_with(model_path, safe_serialization=True)


if __name__ == '__main__':
    unittest.main()
