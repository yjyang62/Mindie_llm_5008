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

import os
import unittest
from unittest.mock import MagicMock, patch

import tbe
import torch

from examples.convert.weight_cast import (
    WeightCast,
    check_path_valid,
    parse_arguments,
)


class TestWeightCastArgsCheck(unittest.TestCase):

    def setUp(self):
        # 获取当前脚本文件的绝对路径
        self.current_file_path = os.path.abspath(__file__)

        # 获取当前工作目录（即运行脚本时的工作目录）
        self.current_working_directory = os.getcwd()

        # 计算相对路径
        self.relative_path = os.path.relpath(self.current_file_path, self.current_working_directory)

    def tearDown(self):
        pass

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--index_file', 'fake_index_file'])
    def test_parse_arguments(self):
        args = parse_arguments()
        self.assertEqual(args.model_path, 'fake_model_path')
        self.assertEqual(args.save_directory, 'fake_save_directory')
        self.assertEqual(args.index_file, 'fake_index_file')

    def test_non_absolute_path(self):
        """测试非绝对路径的情况"""
        with self.assertRaises(ValueError) as context:
            check_path_valid(self.relative_path)
        self.assertIn("must be absolute", str(context.exception))

    def test_non_directory_path(self):
        """测试路径不是一个目录的情况"""
        with self.assertRaises(ValueError) as context:
            check_path_valid('/test/test.txt')
        self.assertIn("is not a valid director", str(context.exception))

    def test_valid_absolute_directory_path(self):
        """测试有效的绝对路径（目录）"""
        try:
            check_path_valid(self.current_working_directory)  # 确保这个路径是存在的
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")


class TestWeightCast(unittest.TestCase):

    def setUp(self):
        # 初始化所有mock
        self.mock_exists = patch('examples.convert.weight_cast.os.path.exists', return_value=True).start()
        self.mock_json_load = patch('examples.convert.weight_cast.json.load').start()
        self.mock_safe_open = patch('examples.convert.weight_cast.safetensors.safe_open').start()
        self.mock_save_file = patch('examples.convert.weight_cast.save_file').start()
        self.mock_check_path_valid = patch('examples.convert.weight_cast.check_path_valid').start()
        self.mock_weight_cast_open = patch('examples.convert.weight_cast.safe_open').start()
        self.mock_allow_internal_format = patch("torch.npu.config").start()
        self.mock_format_cast = patch("torch_npu.npu_format_cast_").start()

        # 配置mock行为
        self.mock_json_load.return_value = {'weight_map': {'file1.safetensor': 'weight1'}}
        self.mock_safe_open.return_value.__enter__.return_value.keys.return_value = \
            ['model.layers.0.mlp.experts.0.gate_proj.weight', "model.layers.0.mlp.experts.0.up_proj.weight",
             "model.layers.0.mlp.experts.0.down_proj.weight"]
        self.mock_safe_open.return_value.__enter__.return_value.get_tensor.return_value = \
            torch.randn((2048, 7168)).to(torch.int8)
        self.mock_format_cast.side_effect = lambda *args: args[0]

        # 测试结束后自动清理
        self.addCleanup(patch.stopall)

    def tearDown(self):
        pass

    def test_cast(self):
        # 准备模拟数据
        args = MagicMock()
        args.model_path = '/path/to/model'
        args.save_directory = '/path/to/save'
        args.index_file = 'index.json'

        # 创建WeightCast实例并调用cast方法
        weightcast = WeightCast(args)
        result = weightcast.cast()

        # 验证结果
        self.assertEqual(tbe.__name__, "tbe")
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()