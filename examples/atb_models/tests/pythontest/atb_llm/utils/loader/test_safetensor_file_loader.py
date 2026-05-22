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

import torch
import torch_npu
from ddt import ddt

from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.utils.mapping import Mapping


KEY = "key"


class FakeSafeTensor:
    def __init__(self, tensor):
        self._data = tensor
    
    def get_shape(self):
        return self._data.shape

    def __getitem__(self, index):
        return self._data[index]


class FakeFileHandler:
    def keys(self):
        return [KEY]


@ddt
class TestSafetensorFileLoader(unittest.TestCase):
    @patch('atb_llm.utils.loader.safetensor_file_loader.file_utils')
    @patch("atb_llm.utils.loader.file_loader.Path")
    @patch("atb_llm.utils.loader.safetensor_file_loader.safe_open")
    def test_file_loader_init(self, _, mock_path, mock_file_utils):
        mock_file_utils.standardize_path = MagicMock(return_value=["path_1", "path_2"])
        mock_file_utils.check_path_permission = MagicMock(return_value=None)
        mock_path_ins = mock_path.return_value
        mock_path_ins.exists = MagicMock(return_value=True)
        mock_path_ins.is_dir = MagicMock(return_value=True)
        mock_path_ins.glob = MagicMock(return_value=["fake_file_1", "fake_file_2"])

        file_loader = SafetensorFileLoader(
            "fake_path",
            torch.device("npu"),
            Mapping(world_size=2, rank=0)
        )

        self.assertEqual(file_loader.model_weight_path, "fake_path")
        self.assertListEqual(file_loader._filenames, ["fake_file_1", "fake_file_2"])

    @patch('atb_llm.utils.loader.safetensor_file_loader.get_weight_filenames')
    @patch("atb_llm.utils.loader.safetensor_file_loader.safe_open", return_value="fake_file_handler")
    def test_file_loader_file_handler(self, _1, _2):
        file_loader = SafetensorFileLoader(
            "fake_path",
            torch.device("npu"),
            Mapping(world_size=2, rank=0)
        )

        file_loader.get_handler("fake_file_name")
        self.assertDictEqual(file_loader._handlers, {"fake_file_name": "fake_file_handler"})

        ret = file_loader.get_handler("fake_file_name")
        self.assertEqual(ret, "fake_file_handler")
        self.assertDictEqual(file_loader._handlers, {"fake_file_name": "fake_file_handler"})

        file_loader.release_file_handler()
        self.assertDictEqual(file_loader._handlers, {})

        file_loader.release_file_handler()
        self.assertDictEqual(file_loader._handlers, {})

    @patch('atb_llm.utils.loader.safetensor_file_loader.get_weight_filenames')
    def test_get_sharded_dim_1(self, _):
        file_loader = SafetensorFileLoader(
            "fake_path",
            torch.device("npu"),
            Mapping(world_size=2, rank=0)
        )

        file_loader._get_slice = MagicMock(return_value=FakeSafeTensor(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])))
        out_tensor = file_loader.get_sharded("fake_tensor", 1, chunk_id=1, num_chunk=2)
        expected_tensor = torch.tensor([[3, 4], [7, 8]])
        self.assertTrue(torch.equal(out_tensor, expected_tensor))

    @patch('atb_llm.utils.loader.safetensor_file_loader.get_weight_filenames')
    def test_get_sharded_dim_0(self, _):
        file_loader = SafetensorFileLoader(
            "fake_path",
            torch.device("npu"),
            Mapping(world_size=2, rank=0)
        )

        file_loader._get_slice = MagicMock(return_value=FakeSafeTensor(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])))
        out_tensor = file_loader.get_sharded("fake_tensor", 0, chunk_id=1, num_chunk=2)
        expected_tensor = torch.tensor([[5, 6, 7, 8]])
        self.assertTrue(torch.equal(out_tensor, expected_tensor))

    @patch('atb_llm.utils.loader.safetensor_file_loader.get_weight_filenames')
    def test_get_sharded_invalid_dim(self, _):
        file_loader = SafetensorFileLoader(
            "fake_path",
            torch.device("npu"),
            Mapping(world_size=2, rank=0)
        )

        file_loader._get_slice = MagicMock(return_value=FakeSafeTensor(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])))
        with self.assertRaises(AssertionError):
            _ = file_loader.get_sharded("fake_tensor", 2, chunk_id=1, num_chunk=2)

    @patch('atb_llm.utils.loader.safetensor_file_loader.get_weight_filenames')
    def test_get_sharded_invalid_num_chunk(self, _):
        file_loader = SafetensorFileLoader(
            "fake_path",
            torch.device("npu"),
            Mapping(world_size=2, rank=0)
        )

        file_loader._get_slice = MagicMock(return_value=FakeSafeTensor(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])))
        with self.assertRaises(ValueError):
            _ = file_loader.get_sharded("fake_tensor", 0, chunk_id=1, num_chunk=3)

    @patch('atb_llm.utils.loader.safetensor_file_loader.get_weight_filenames')
    def test_get_slice(self, _):
        file_loader = SafetensorFileLoader(
            "fake_path",
            torch.device("npu"),
            Mapping(world_size=2, rank=0)
        )

        file_loader._routing = {KEY: "fake_file_name"}
        fake_file_handler = MagicMock()
        file_loader.get_handler = MagicMock(return_value=fake_file_handler)
        fake_file_handler.get_slice.return_value = torch.tensor([])

        out = file_loader._get_slice(KEY)
        self.assertTrue(torch.equal(out, torch.tensor([])))


if __name__ == '__main__':
    unittest.main()