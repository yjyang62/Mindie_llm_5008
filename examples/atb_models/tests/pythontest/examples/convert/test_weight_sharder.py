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
from unittest.mock import patch
import argparse

from examples.convert.weight_sharder import WeightSharder, parse_arguments, main


class TestWeightSharder(unittest.TestCase):
    def setUp(self):
        self.model_path = "/fake_model_path"
        self.save_directory = "/fake_save_directory"
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

        self.mock_env = patch("examples.convert.weight_sharder.ENV").start()
        self.mock_env.deepseek_mtp = 0
        self.mock_env.auto_transpose_enable = True
        self.mock_env.rank = 0
        self.mock_env.world_size = 1
        self.mock_env.local_rank = 0

        self.mock_model_runner = patch("examples.convert.weight_sharder.ModelRunner").start()

        self.mocked_file_utils = patch('examples.convert.weight_sharder.file_utils').start()
        self.mocked_file_utils.standardize_path.return_value = self.save_directory
        self.mocked_file_utils.check_path_permission.return_value = None

        self.mocked_copy_tokenizer_files = patch("examples.convert.weight_sharder.copy_tokenizer_files").start()
        self.mocked_modify_config = patch("examples.convert.weight_sharder.modify_config").start()

        self.addCleanup(patch.stopall)

    def test_init(self):
        kwargs = {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "model_path": self.model_path,
            "save_directory": self.save_directory,
        }

        weight_sharder = WeightSharder(**kwargs)

        self.assertFalse(self.mock_env.auto_transpose_enable)
        weight_sharder.model.load_weights.assert_called_once()

    def test_save_sharded(self):
        mock_model = self.mock_model_runner.return_value
        mock_model.dtype = "flaot16"
        mock_model.quantize = None

        kwargs = {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "model_path": self.model_path,
        }

        weight_sharder = WeightSharder(**kwargs)
        weight_sharder.save_sharded()

        mock_model.save_sharded.assert_called_once_with(save_directory=None)

    @patch('sys.argv', ['weight_sharder.py', '--model_path', 'fake_model_path',
                        '--save_directory', 'fake_save_directory',
                        '--dp', '2', '--tp', '4'])
    def test_valid_param(self):
        args = parse_arguments()
        self.assertEqual(args.model_path, 'fake_model_path')
        self.assertEqual(args.save_directory, 'fake_save_directory')
        self.assertEqual(args.dp, 2)
        self.assertEqual(args.tp, 4)

    @patch('sys.argv', ['weight_sharder.py', '--model_path', 'fake_model_path',
                        '--save_directory', 'fake_save_directory',
                        '--dp', '2', '--tp', '4'])
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_main_execution(self, mock_path_exists, mock_mkdirs):
        mock_path_exists.return_value = False
        main()
        mock_mkdirs.assert_called_once_with("fake_save_directory", exist_ok=True)


if __name__ == '__main__':
    unittest.main()