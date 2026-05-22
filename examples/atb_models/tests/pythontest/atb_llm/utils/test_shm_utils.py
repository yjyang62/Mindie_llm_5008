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
from unittest.mock import patch, MagicMock

import torch
import numpy as np

# Function under test
from atb_llm.utils.shm_utils import (
    create_shm, MAX_SHM_SIZE,
    release_shared_memory, 
    encode_shm_name_to_int64, SHM_NAME_MAX_LENGTH, SHM_NAME_PREFIX, SPLIT_PLAN_SET,
    encode_shape_to_int64,
    decode_shm_name_from_int64,
    decode_shape_from_int64,
    get_data_from_shm,
    process_shared_memory
)


class TestShmUtils(unittest.TestCase):

    @patch('atb_llm.utils.file_utils.safe_open', new_callable=MagicMock)
    @patch('multiprocessing.shared_memory.SharedMemory')
    def test_create_shm_success(self, mock_shared_memory, mock_open):
        mock_shm = mock_shared_memory.return_value
        mock_shm.name = 'test_shm_name'
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        size = 1024
        shm_name_save_path = 'shm_path.txt'
        shm = create_shm(size, shm_name_save_path)

        mock_shared_memory.assert_called_once_with(create=True, size=size)
        mock_open.assert_called_once_with(shm_name_save_path, 'a', is_exist_ok=True)
        mock_file.write.assert_called_once_with(f'{mock_shm.name}\n')

        self.assertEqual(shm, mock_shm)

    @patch('atb_llm.utils.file_utils.safe_open')
    @patch('multiprocessing.shared_memory.SharedMemory')
    def test_create_shm_error(self, mock_shared_memory, mock_safe_open):
        shm_name_save_path = 'shm_path.txt'
        with self.assertRaises(ValueError):
            create_shm(-1024, shm_name_save_path)
        with self.assertRaises(ValueError):
            create_shm(1024.0, shm_name_save_path)  # non-integer
        with self.assertRaises(ValueError):
            create_shm(MAX_SHM_SIZE + 1, shm_name_save_path)  # exceeds maximun value

        mock_shared_memory.side_effect = Exception("Failed to create shared memory")
        with self.assertRaises(RuntimeError):
            create_shm(1024, shm_name_save_path)

        mock_shared_memory.side_effect = None
        mock_shared_memory.return_value = MagicMock()
        mock_safe_open.side_effect = Exception("File open failed")
        with self.assertRaises(RuntimeError):
            create_shm(1024, shm_name_save_path)

    @patch('multiprocessing.shared_memory.SharedMemory', new_callable=MagicMock)
    @patch('atb_llm.utils.file_utils.safe_open')
    @patch('os.path.exists', return_value=True)
    def test_release_shared_memory_success(self, mock_exists, mock_safe_open, mock_shared_memory):
        mock_safe_open.return_value.__enter__.return_value.readlines.return_value = ['shm_name_1', 'shm_name_2']
        mock_shm = MagicMock()
        mock_shared_memory.return_value = mock_shm

        release_shared_memory('mock_file.txt')

        mock_safe_open.assert_called_once_with('mock_file.txt', 'r', is_exist_ok=True)
        mock_shared_memory.assert_any_call(name='shm_name_1')
        mock_shared_memory.assert_any_call(name='shm_name_2')
        mock_shm.close.assert_called_with()
        mock_shm.unlink.assert_called_with()

    @patch('os.path.exists')
    @patch('atb_llm.utils.file_utils.safe_open', new_callable=MagicMock)
    @patch('multiprocessing.shared_memory.SharedMemory')
    def test_release_shared_memory_error(self, mock_shared_memory, mock_safe_open, mock_exists):
        shm_name_save_path = 'mock_path.txt'
        # test path does not exists
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            release_shared_memory(shm_name_save_path)
        
        # test file safe open error
        mock_exists.reset_mock()
        mock_exists.return_value = True
        mock_safe_open.side_effect = Exception("File open failed")
        with self.assertRaises(RuntimeError):
            release_shared_memory(shm_name_save_path)

        # test shared memory operation failed
        mock_safe_open.reset_mock()
        mock_safe_open.side_effect = None
        mock_safe_open.return_value.__enter__.return_value.readlines.return_value = ['shm_name_1']
        mock_shared_memory.side_effect = Exception("Shared memory error")
        with self.assertRaises(RuntimeError):
            release_shared_memory(shm_name_save_path)

        mock_safe_open.reset_mock()
        mock_shared_memory.reset_mock()
        mock_safe_open.return_value.__enter__.return_value.readlines.return_value = ['shm_name_1', 'shm_name_2']
        mock_shared_memory.side_effect = FileNotFoundError("Can not find shm name")
        release_shared_memory(shm_name_save_path)
        mock_safe_open.assert_called_once_with(shm_name_save_path, 'r', is_exist_ok=True)
        # verify that shm is created, closed, and unbound
        mock_shared_memory.assert_any_call(name='shm_name_1')
        mock_shared_memory.assert_any_call(name='shm_name_2')
        mock_shared_memory.return_value.close.assert_not_called()
        mock_shared_memory.return_value.unlink.assert_not_called()

    def test_encode_shm_name_to_int64_success(self):
        name = 'abCD123'
        expected_value = -2110989766714458114  # expected value which encoded from name SHM_NAME_PREFIX + 'abCD123'
        result = encode_shm_name_to_int64(name)
        self.assertEqual(result, expected_value)

        # test for valid shm name (with prefix)
        name_with_prefix = SHM_NAME_PREFIX + 'abCD123'
        result = encode_shm_name_to_int64(name_with_prefix)
        self.assertEqual(result, expected_value)

    def test_encode_shm_name_to_int64_error(self):
        # test name length exeeds limit
        name_too_long = 'a' * (SHM_NAME_MAX_LENGTH + 1)
        with self.assertRaises(ValueError):
            encode_shm_name_to_int64(name_too_long)

        # test name contains invalid characters
        invalid_name = 'abcd*&123'
        with self.assertRaises(ValueError):
            encode_shm_name_to_int64(invalid_name)

        # test for encoded values exeeding 64 bits
        name_too_large = 'a' * SHM_NAME_MAX_LENGTH
        with self.assertRaises(ValueError):
            encode_shm_name_to_int64(name_too_large)

    def test_decode_shm_name_from_int64_success(self):
        # encoded value from name SHM_NAME_PREFIX + 'abCD123'
        encoded_value = -2110989766714458114
        expected_name = SHM_NAME_PREFIX + 'abCD123'
        result = decode_shm_name_from_int64(encoded_value)
        self.assertEqual(result, expected_name)

        tensor_value = torch.tensor(encoded_value)
        result = decode_shm_name_from_int64(tensor_value)
        self.assertEqual(result, expected_name)

    def test_decode_shm_name_from_int64_error(self):
        invalid_encoded_value = -9876543210123456789  # random value
        with self.assertRaises(ValueError):
            decode_shm_name_from_int64(invalid_encoded_value)

        # encoded value from name SHM_NAME_PREFIX + 'abCD123' but change the last number which means change the checksum
        incorrect_checksum_value = -2110989766714458115
        with self.assertRaises(ValueError):
            decode_shm_name_from_int64(incorrect_checksum_value)

    def test_encode_decode_shm_name(self):
        name = SHM_NAME_PREFIX + "abCD123"
        encoded_shm_name = encode_shm_name_to_int64(name)
        decoded_shm_name = decode_shm_name_from_int64(encoded_shm_name)
        self.assertEqual(decoded_shm_name, name)

    def test_encode_shape_to_int64_error(self):
        invalid_shape_6d = [1, 2, 3, 4, 5, 6]
        with self.assertRaises(ValueError):
            encode_shape_to_int64(invalid_shape_6d)

        shape_dim_too_large = [1, 128, 512, 256]
        with self.assertRaises(ValueError):
            encode_shape_to_int64(shape_dim_too_large)

    def test_decode_shape_from_int64_error(self):
        # encoded value from shape [128, 128, 128], and change the checksum
        encoded_shape_value = encode_shape_to_int64([128, 128, 128]) - 1
        with self.assertRaises(ValueError):
            decode_shape_from_int64(encoded_shape_value)

        encoded_shape_value = -(0b001 << 60)
        with self.assertRaises(ValueError):
            decode_shape_from_int64(encoded_shape_value)

        encoded_shape_value = -(0b110 << 60)
        with self.assertRaises(ValueError):
            decode_shape_from_int64(encoded_shape_value)

    def test_encode_decode_shape(self):
        for i in range(1, 6):
            shape = [(1 << dim - 1) for dim in SPLIT_PLAN_SET.get(i)]
            encoded_shape = encode_shape_to_int64(shape)
            decoded_shape = decode_shape_from_int64(encoded_shape)
            self.assertEqual(decoded_shape, shape)

    @patch('multiprocessing.shared_memory.SharedMemory')
    @patch('atb_llm.utils.shm_utils.decode_shape_from_int64', return_value=[128, 128])
    @patch('atb_llm.utils.shm_utils.decode_shm_name_from_int64', return_value='shm_name')
    def test_get_data_from_shm_success(self, mock_shm_name, mock_shm_shape, mock_shared_memory):
        mock_shm = MagicMock()
        mock_shared_memory.return_value = mock_shm
        mock_shm.buf = bytearray(np.random.rand(128, 128).astype(np.float32).tobytes())

        # 123456789098765421: an unused number which mocks the first param shm name
        # 987654321334567890: the encoded num from shape [128, 128]
        result = get_data_from_shm(123456789098765421, 987654321334567890)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size(mock_shm_shape.return_value))
        self.assertEqual(result.dtype, torch.float32)

    @patch('atb_llm.utils.shm_utils.encode_shm_name_to_int64')
    @patch('atb_llm.utils.shm_utils.encode_shape_to_int64')
    @patch('atb_llm.utils.shm_utils.create_shm')
    def test_process_shared_memory_with_pixel_values_only(self, mock_create_shm, mock_encode_shape, mock_encode_shm_name):
        mock_shm_name_encoded = 123456789
        mock_shape_encoded = 987654321
        mock_shm = MagicMock()
        mock_shm.name = "test_shm"
        mock_shm.buf = bytearray(np.random.rand(128, 128).astype(np.float32).tobytes())
        mock_create_shm.return_value = mock_shm
        mock_encode_shm_name.return_value = mock_shm_name_encoded
        mock_encode_shape.return_value = mock_shape_encoded
        inputs = {
            'pixel_values': torch.randn(128, 128),
            'image_grid_thw': torch.ones((128, 128), dtype=torch.int32)
        }
        result = process_shared_memory(inputs, "shm_path.txt")
        self.assertEqual(result['pixel_values_shm_name'], mock_shm_name_encoded)
        self.assertEqual(result['pixel_values_shape_value'], mock_shape_encoded)
        self.assertEqual(result['image_grid_thw_shm_name'], mock_shm_name_encoded)
        self.assertEqual(result['image_grid_thw_shape_value'], mock_shape_encoded)
        self.assertEqual(result['pixel_values_videos_shm_name'], 0)
        self.assertEqual(result['pixel_values_videos_shape_value'], 0)
        self.assertEqual(result['video_grid_thw_shm_name'], 0)
        self.assertEqual(result['video_grid_thw_shape_value'], 0)
        self.assertEqual(mock_create_shm.call_count, 2)
        self.assertEqual(mock_encode_shm_name.call_count, 2)
        self.assertEqual(mock_encode_shape.call_count, 2)

    @patch('atb_llm.utils.shm_utils.encode_shm_name_to_int64')
    @patch('atb_llm.utils.shm_utils.encode_shape_to_int64')
    @patch('atb_llm.utils.shm_utils.create_shm')
    def test_process_shared_memory_with_videos_only(self, mock_create_shm, mock_encode_shape, mock_encode_shm_name):
        mock_shm_name_encoded = 123456789
        mock_shape_encoded = 987654321
        mock_shm = MagicMock()
        mock_shm.name = "test_shm"
        mock_shm.buf = bytearray(np.random.rand(128, 128).astype(np.float32).tobytes())
        mock_create_shm.return_value = mock_shm
        mock_encode_shm_name.return_value = mock_shm_name_encoded
        mock_encode_shape.return_value = mock_shape_encoded
        inputs = {
            'pixel_values_videos': torch.randn(128, 128),
            'video_grid_thw': torch.ones((128, 128), dtype=torch.int32)
        }
        result = process_shared_memory(inputs, "shm_path.txt")
        self.assertEqual(result['pixel_values_shm_name'], 0)
        self.assertEqual(result['pixel_values_shape_value'], 0)
        self.assertEqual(result['image_grid_thw_shm_name'], 0)
        self.assertEqual(result['image_grid_thw_shape_value'], 0)
        self.assertEqual(result['pixel_values_videos_shm_name'], mock_shm_name_encoded)
        self.assertEqual(result['pixel_values_videos_shape_value'], mock_shape_encoded)
        self.assertEqual(result['video_grid_thw_shm_name'], mock_shm_name_encoded)
        self.assertEqual(result['video_grid_thw_shape_value'], mock_shape_encoded)
        self.assertEqual(mock_create_shm.call_count, 2)
        self.assertEqual(mock_encode_shm_name.call_count, 2)
        self.assertEqual(mock_encode_shape.call_count, 2)

if __name__ == '__main__':
    unittest.main()