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
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch

import torch
import PIL.Image
import numpy as np

from atb_llm.models.qwen3_vl.input_builder_qwen3_vl import Qwen3vlInputBuilder


@dataclass
class MockVisionConfig:
    """Mock vision configuration for Qwen3vlInputBuilder testing"""
    spatial_merge_size: int = 2


@dataclass
class MockConfig:
    """Mock model configuration for Qwen3vlInputBuilder testing"""
    vision_config: MockVisionConfig = None
    image_token_id: int = 151644
    video_token_id: int = 151645
    vision_start_token_id: int = 151646
    vision_end_token_id: int = 151647
    
    def __post_init__(self):
        """Initialize default vision config if not provided"""
        if self.vision_config is None:
            self.vision_config = MockVisionConfig()


class MockTokenizer:
    """Mock tokenizer for Qwen3vlInputBuilder testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 151643
        self.bos_token_id = 151643


class MockProcessor:
    """Mock processor for Qwen3vlInputBuilder testing"""
    def __init__(self):
        self.tokenizer = MockTokenizer()
    
    def apply_chat_template(self, conversation_list, tokenize=True, add_generation_prompt=True,
                           return_dict=True, return_tensors='pt'):
        """Mock chat template application with dummy input ids"""
        return {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        }


class TestQwen3vlInputBuilder(unittest.TestCase):
    """Unit tests for Qwen3vlInputBuilder class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.tokenizer = MockTokenizer()
        self.processor = MockProcessor()
        self.config = MockConfig()
    
    def test_init(self):
        """Test Qwen3vlInputBuilder initialization
        
        Given: Mock tokenizer, config and processor
        When: Initialize Qwen3vlInputBuilder instance
        Then: Instance attributes should match the input fixtures
        """
        # Given (already set via setUp)
        
        # When
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        
        # Then
        self.assertEqual(input_builder.tokenizer, self.tokenizer)
        self.assertEqual(input_builder.processor, self.processor)
        self.assertEqual(input_builder.config, self.config)
        self.assertEqual(input_builder.spatial_merge_size, self.config.vision_config.spatial_merge_size)
        self.assertEqual(input_builder.image_token_id, self.config.image_token_id)
        self.assertEqual(input_builder.video_token_id, self.config.video_token_id)
        self.assertEqual(input_builder.vision_start_token_id, self.config.vision_start_token_id)
        self.assertEqual(input_builder.vision_end_token_id, self.config.vision_end_token_id)
    
    def test_get_shm_name_save_path(self):
        """Test static method get_shm_name_save_path
        
        Given: Absolute and relative file paths
        When: Call get_shm_name_save_path with the file paths
        Then: Should return correct shm file path in the parent's parent directory
        """
        # Given
        abs_file_path = '/project/data/inputs/current/file.jpg'
        rel_file_path = 'data/inputs/file.jpg'
        expected_abs_result = '/project/data/inputs/shm_name.txt'
        expected_rel_result = os.path.join(os.path.dirname(os.path.dirname(rel_file_path)), 'shm_name.txt')
        
        # When
        abs_result = Qwen3vlInputBuilder.get_shm_name_save_path(abs_file_path)
        rel_result = Qwen3vlInputBuilder.get_shm_name_save_path(rel_file_path)
        
        # Then
        self.assertEqual(abs_result, expected_abs_result)
        self.assertEqual(rel_result, expected_rel_result)
    
    def test_generate_position_ids_without_vision(self):
        """Test generate_position_ids without vision tokens
        
        Given: Input ids without vision tokens
        When: Call generate_position_ids
        Then: Should return sequential position ids matching input length
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        expected_pos_ids = np.arange(len(input_ids), dtype=np.int64)
        
        # When
        result_pos_ids = input_builder.generate_position_ids(input_ids)
        
        # Then
        np.testing.assert_array_equal(result_pos_ids, expected_pos_ids)
    
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.get_data_from_shm')
    def test_generate_position_ids_with_vision(self, mock_get_data_from_shm):
        """Test generate_position_ids with vision tokens
        
        Given: Input ids with vision start token, mocked shm data and helper methods
        When: Call generate_position_ids
        Then: Should return sequential position ids matching input length
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        input_ids = np.array([1, 2, self.config.vision_start_token_id, 4, 5], dtype=np.int64)
        expected_pos_ids = np.arange(len(input_ids), dtype=np.int64)
        mock_get_data_from_shm.return_value = None
        
        # When
        with patch.object(input_builder, '_parse_inputs_ids_with_shm', return_value=(None, None)):
            with patch.object(input_builder, '_compute_llm_pos_delta', return_value=0):
                result_pos_ids = input_builder.generate_position_ids(input_ids)
        
        # Then
        np.testing.assert_array_equal(result_pos_ids, expected_pos_ids)
    
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.process_shared_memory')
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.check_video_path')
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.safe_open_image')
    def test_make_context_with_text_only(self, mock_safe_open_image, mock_check_video_path, mock_process_shm):
        """Test make_context with text-only content
        
        Given: Conversation with text-only user content, mocked shared memory processing
        When: Call make_context
        Then: Should return valid result without calling image or video helper methods
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        conversation = [{"role": "user", "content": [{"text": "Hello, world!"}]}]
        mock_process_shm.return_value = {}
        
        # When
        result = input_builder.make_context(rank=0, conversation=conversation)
        
        # Then
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_not_called()
        mock_check_video_path.assert_not_called()
        mock_process_shm.assert_called_once()
    
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.process_shared_memory')
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.check_video_path')
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.safe_open_image')
    def test_make_context_with_image(self, mock_safe_open_image, mock_check_video_path, mock_process_shm):
        """Test make_context with image content
        
        Given: Conversation with image content, mocked image and shared memory processing
        When: Call make_context
        Then: Should return valid result and call image helper method once
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        mock_image = MagicMock(spec=PIL.Image.Image)
        mock_safe_open_image.return_value = mock_image
        conversation = [
            {"role": "user", "content": [{"text": "Describe this image"}, {"image": "/path/to/image.jpg"}]}
        ]
        mock_process_shm.return_value = {}
        
        # When
        result = input_builder.make_context(rank=0, conversation=conversation)
        
        # Then
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_called_once_with(PIL.Image, "/path/to/image.jpg")
        mock_check_video_path.assert_not_called()
    
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.process_shared_memory')
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.check_video_path')
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.safe_open_image')
    def test_make_context_with_video(self, mock_safe_open_image, mock_check_video_path, mock_process_shm):
        """Test make_context with video content
        
        Given: Conversation with video content, mocked video path check and shared memory processing
        When: Call make_context
        Then: Should return valid result and call video helper method once
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        mock_check_video_path.return_value = "/path/to/video.mp4"
        conversation = [
            {"role": "user", "content": [{"text": "Describe this video"}, {"video": "/path/to/video.mp4"}]}
        ]
        mock_process_shm.return_value = {}
        
        # When
        result = input_builder.make_context(rank=0, conversation=conversation)
        
        # Then
        self.assertIsNotNone(result)
        mock_check_video_path.assert_called_once_with("/path/to/video.mp4")
        mock_safe_open_image.assert_not_called()
    
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.process_shared_memory')
    def test_make_context_with_string_content(self, mock_process_shm):
        """Test make_context with string content (non-list format)
        
        Given: Conversation with string content, mocked shared memory processing
        When: Call make_context
        Then: Should return valid result and call shared memory processing once
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        conversation = [{"role": "user", "content": "Hello, world!"}]
        mock_process_shm.return_value = {}
        
        # When
        result = input_builder.make_context(rank=0, conversation=conversation)
        
        # Then
        self.assertIsNotNone(result)
        mock_process_shm.assert_called_once()
    
    def test_make_context_with_invalid_content_type(self):
        """Test make_context with invalid content type
        
        Given: Conversation with integer content (invalid type)
        When: Call make_context
        Then: Should raise TypeError with relevant error message
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        conversation = [{"role": "user", "content": 123}]
        
        # When & Then
        with self.assertRaises(TypeError) as context:
            input_builder.make_context(rank=0, conversation=conversation)
        
        self.assertIn("content", str(context.exception))
        self.assertIn("List[Dict]", str(context.exception))
    
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.process_shared_memory')
    def test_make_context_with_shm_name_save_path(self, mock_process_shm):
        """Test make_context with custom shm_name_save_path
        
        Given: Conversation with text content, custom shm path and mocked shared memory processing
        When: Call make_context with custom shm path
        Then: Should call shared memory processing with the custom path
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        conversation = [{"role": "user", "content": [{"text": "Hello"}]}]
        mock_process_shm.return_value = {}
        custom_shm_path = "/custom/shm/path.txt"
        
        # When
        result = input_builder.make_context(rank=0, conversation=conversation, shm_name_save_path=custom_shm_path)
        
        # Then
        self.assertIsNotNone(result)
        mock_process_shm.assert_called_once()
        call_args = mock_process_shm.call_args
        self.assertEqual(call_args[0][1], custom_shm_path)
    
    def test_update_token_id_without_vision(self):
        """Test update_token_id without vision tokens
        
        Given: Input ids without vision tokens, empty shm info
        When: Call update_token_id
        Then: Should return flattened input ids matching expected tensor
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}
        shm_info = {}
        expected_tensor = inputs['input_ids'].flatten()
        
        # When
        result_tensor = input_builder.update_token_id(inputs, shm_info)
        
        # Then
        torch.testing.assert_close(result_tensor, expected_tensor)
    
    def test_update_token_id_with_image(self):
        """Test update_token_id with image tokens
        
        Given: Input ids with vision tokens, sufficient padding, valid shm info
        When: Call update_token_id
        Then: Should return valid tensor with shm info in correct positions
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        # Build input ids with sufficient space
        input_ids_list = [1, 2, self.config.vision_start_token_id, self.config.image_token_id]
        input_ids_list.extend([0] * 20)  # Sufficient padding
        input_ids_list.append(self.config.vision_end_token_id)
        input_ids_list.extend([5, 6, 7])
        inputs = {'input_ids': torch.tensor([input_ids_list])}
        shm_info = {
            'pixel_values_shm_name': 1001,
            'pixel_values_shape_value': 1002,
            'image_grid_thw_shm_name': 1003,
            'image_grid_thw_shape_value': 1004,
            'pixel_values_videos_shm_name': 0,
            'pixel_values_videos_shape_value': 0,
            'video_grid_thw_shm_name': 0,
            'video_grid_thw_shape_value': 0,
        }
        
        # When
        result_tensor = input_builder.update_token_id(inputs, shm_info)
        
        # Then
        self.assertIsNotNone(result_tensor)
        self.assertIsInstance(result_tensor, torch.Tensor)
        # Verify shm info is in correct position
        boi_pos = torch.where(result_tensor == self.config.vision_start_token_id)[0][0].item() + 1
        self.assertEqual(result_tensor[boi_pos + 1].item(), shm_info['pixel_values_shm_name'])
    
    def test_update_token_id_insufficient_space(self):
        """Test update_token_id with insufficient space for shm info
        
        Given: Input ids with vision tokens but no padding (insufficient space), valid shm info
        When: Call update_token_id
        Then: Should raise RuntimeError with relevant error message
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        # Build input ids with insufficient space
        input_ids_list = [1, 2, self.config.vision_start_token_id, self.config.image_token_id]
        input_ids_list.append(self.config.vision_end_token_id)
        inputs = {'input_ids': torch.tensor([input_ids_list])}
        shm_info = {
            'pixel_values_shm_name': 1001,
            'pixel_values_shape_value': 1002,
            'image_grid_thw_shm_name': 1003,
            'image_grid_thw_shape_value': 1004,
            'pixel_values_videos_shm_name': 0,
            'pixel_values_videos_shape_value': 0,
            'video_grid_thw_shm_name': 0,
            'video_grid_thw_shape_value': 0,
        }
        
        # When & Then
        with self.assertRaises(RuntimeError) as context:
            input_builder.update_token_id(inputs, shm_info)
        
        self.assertIn("Load share memory info to input ids failed", str(context.exception))
    
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.get_data_from_shm')
    def test_parse_inputs_ids_with_shm_invalid_shm(self, mock_get_data_from_shm):
        """Test _parse_inputs_ids_with_shm with invalid shm names
        
        Given: Input ids with 16 image tokens, replaced with invalid shm values (-1)
        When: Call _parse_inputs_ids_with_shm
        Then: Should return None for both grids and not call get_data_from_shm
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        input_builder.vision_start_token_id = self.config.vision_start_token_id
        input_builder.image_token_id = self.config.image_token_id
        
        # Build base input ids with 16 image tokens
        input_ids_list = [
            1,
            2,
            self.config.vision_start_token_id
        ] + [self.config.image_token_id] * 16 + [
            self.config.vision_end_token_id,
            5,
            6
        ]
        # Calculate base position
        vision_start_idx = input_ids_list.index(self.config.vision_start_token_id)
        boi_pos = vision_start_idx + 1
        # Replace with invalid shm values
        input_ids_list[boi_pos + 3] = 0
        input_ids_list[boi_pos + 4] = 0
        input_ids_list[boi_pos + 7] = 0
        input_ids_list[boi_pos + 8] = 0
        input_ids = np.array(input_ids_list, dtype=np.int64)
        
        # When
        image_grid_thw, video_grid_thw = input_builder._parse_inputs_ids_with_shm(input_ids)
        
        # Then
        self.assertIsNone(image_grid_thw)
        self.assertIsNone(video_grid_thw)
        mock_get_data_from_shm.assert_not_called()
        
    @patch('atb_llm.models.qwen3_vl.input_builder_qwen3_vl.get_data_from_shm')
    def test_parse_inputs_ids_with_shm_valid_shm(self, mock_get_data_from_shm):
        """Test _parse_inputs_ids_with_shm with valid shm names
        
        Given: Input ids with 16 image tokens, replaced with valid shm values, mocked grid data
        When: Call _parse_inputs_ids_with_shm
        Then: Should return valid grids matching mock data and call get_data_from_shm twice
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        input_builder.vision_start_token_id = self.config.vision_start_token_id
        input_builder.image_token_id = self.config.image_token_id
        
        # Mock valid grid data
        mock_image_grid = np.array([[1, 2, 3]], dtype=np.int32)
        mock_video_grid = np.array([[2, 4, 6]], dtype=np.int32)
        mock_get_data_from_shm.side_effect = [mock_image_grid, mock_video_grid]
        
        # Build base input ids with 16 image tokens
        input_ids_list = [
            1,
            2,
            self.config.vision_start_token_id
        ] + [self.config.image_token_id] * 16 + [
            self.config.vision_end_token_id,
            5,
            6
        ]
        # Calculate base position
        vision_start_idx = input_ids_list.index(self.config.vision_start_token_id)
        boi_pos = vision_start_idx + 1
        # Replace with valid shm values
        input_ids_list[boi_pos + 3] = 100
        input_ids_list[boi_pos + 4] = 101
        input_ids_list[boi_pos + 7] = 102
        input_ids_list[boi_pos + 8] = 103
        input_ids = np.array(input_ids_list, dtype=np.int64)
        
        # When
        image_grid_thw, video_grid_thw = input_builder._parse_inputs_ids_with_shm(input_ids)
        
        # Then
        self.assertIsNotNone(image_grid_thw)
        self.assertIsNotNone(video_grid_thw)
        np.testing.assert_array_equal(image_grid_thw, mock_image_grid)
        np.testing.assert_array_equal(video_grid_thw, mock_video_grid)
        self.assertEqual(mock_get_data_from_shm.call_count, 2)
    
    def test_compute_llm_pos_delta_no_vision(self):
        """Test _compute_llm_pos_delta without vision inputs
        
        Given: Input ids without vision tokens, None for both grids
        When: Call _compute_llm_pos_delta
        Then: Should return 0 as position delta
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        image_grid_thw = None
        video_grid_thw = None
        
        # When
        position_delta = input_builder._compute_llm_pos_delta(input_ids, image_grid_thw, video_grid_thw)
        
        # Then
        self.assertEqual(position_delta, 0)
    
    def test_compute_llm_pos_delta_with_image(self):
        """Test _compute_llm_pos_delta with image inputs
        
        Given: Input ids with 16 image tokens, valid image grid, None for video grid
        When: Call _compute_llm_pos_delta
        Then: Should return a negative position delta
        """
        # Given
        input_builder = Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)
        # Build input ids with 16 image tokens
        input_ids_list = [
            1,
            2,
            self.config.vision_start_token_id
        ] + [self.config.image_token_id] * 16 + [
            self.config.vision_end_token_id,
            5,
            6
        ]
        input_ids = np.array(input_ids_list, dtype=np.int64)
        image_grid_thw = np.array([[1, 4, 4]], dtype=np.int32)
        video_grid_thw = None
        
        # When
        position_delta = input_builder._compute_llm_pos_delta(input_ids, image_grid_thw, video_grid_thw)
        
        # Then
        self.assertLess(position_delta, 0)


if __name__ == '__main__':
    unittest.main()
