# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import patch
import argparse

from examples.convert.model_slim.sparse_compressor import parse_arguments


class TestSparseCompressor(unittest.TestCase):
    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path',
                        '--save_directory', 'fake_save_directory'])
    def test_default_value(self):
        args = parse_arguments()
        self.assertEqual(args.multiprocess_num, 8)
        self.assertIsNone(args.save_split_w8a8s_dir)

    @patch('sys.argv', ['test_script.py', '--model_path', '', '--save_directory', 'fake_save_directory'])
    def test_model_path_too_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'A' * 4097, '--save_directory', 'fake_save_directory'])
    def test_model_path_too_long(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', ''])
    def test_save_directory_too_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'A' * 4097])
    def test_save_directory_too_long(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path',
                        '--save_directory', 'fake_save_directory',
                        '--multiprocess_num', '16', '--save_split_w8a8s_dir', 'fake_save_split_w8a8s_dir'])
    def test_valid_param(self):
        args = parse_arguments()
        self.assertEqual(args.model_path, 'fake_model_path')
        self.assertEqual(args.save_directory, 'fake_save_directory')
        self.assertEqual(args.multiprocess_num, 16)
        self.assertEqual(args.save_split_w8a8s_dir, 'fake_save_split_w8a8s_dir')


if __name__ == '__main__':
    unittest.main()