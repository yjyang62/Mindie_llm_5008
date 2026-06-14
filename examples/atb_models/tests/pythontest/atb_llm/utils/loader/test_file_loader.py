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
from unittest.mock import MagicMock, patch


from atb_llm.utils.loader.file_loader import BaseFileLoader, get_weight_filenames


class TestBaseFileLoader(unittest.TestCase):
    def setUp(self):
        self.model_weight_path = "fake_path"
        self.filename = "fake_file_name"
        self.tensor_name = "fake_tensor_name"

    def test_file_loader_extension(self):
        file_loader = BaseFileLoader(self.model_weight_path)
        with self.assertRaises(NotImplementedError):
            _ = file_loader.extension

    def test_file_loader_release_file_handler(self):
        file_loader = BaseFileLoader(self.model_weight_path)
        self.assertRaises(NotImplementedError, file_loader.release_file_handler)

    def test_file_loader_get_handler(self):
        file_loader = BaseFileLoader(self.model_weight_path)
        self.assertRaises(NotImplementedError, file_loader.get_handler, self.filename)

    def test_file_loader_get_shape(self):
        file_loader = BaseFileLoader(self.model_weight_path)
        self.assertRaises(NotImplementedError, file_loader.get_shape, self.tensor_name)

    def test_file_loader_get_tensor(self):
        file_loader = BaseFileLoader(self.model_weight_path)
        self.assertRaises(NotImplementedError, file_loader.get_tensor, self.tensor_name)

    def test_file_loader_get_sharded(self):
        file_loader = BaseFileLoader(self.model_weight_path)
        self.assertRaises(NotImplementedError, file_loader.get_sharded,
                          self.tensor_name, 0, 0, 2)  # 0, 0, 2: dim, chunk_id, num_chunk

    @patch("atb_llm.utils.loader.file_loader.Path")
    def test_get_weight_filenames(self, mock_path):
        mock_path_ins = mock_path.return_value
        mock_path_ins.exists = MagicMock(return_value=True)
        mock_path_ins.is_dir = MagicMock(return_value=True)
        mock_path_ins.glob = MagicMock(return_value=[])

        self.assertRaises(FileNotFoundError, get_weight_filenames, self.model_weight_path, "safetensors")


if __name__ == '__main__':
    unittest.main()