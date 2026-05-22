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
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import torch
import PIL.Image

from atb_llm.models.qwen3_vl.router_qwen3_vl import Qwen3vlRouter


# Test constants
FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    'model_type': 'qwen3_vl',
    'text_config': {
        'dtype': 'float16',
        'vocab_size': 152064,
        'max_position_embeddings': 32768,
        'num_hidden_layers': 32,
    },
    'vision_config': {
        'image_size': 1120,
        'patch_size': 14,
    }
}

# Default shared memory info structure
DEFAULT_SHM_INFO = {
    'pixel_values_shm_name': 0,
    'pixel_values_shape_value': 0,
    'image_grid_thw_shm_name': 0,
    'image_grid_thw_shape_value': 0,
    'pixel_values_videos_shm_name': 0,
    'pixel_values_videos_shape_value': 0,
    'video_grid_thw_shm_name': 0,
    'video_grid_thw_shape_value': 0,
}


@dataclass
class MockTextConfig:
    dtype: torch.dtype = torch.float16
    vocab_size: int = 152064
    max_position_embeddings: int = 32768


@dataclass
class MockVisionConfig:
    image_size: int = 1120
    patch_size: int = 14


@dataclass
class MockConfig:
    text_config: MockTextConfig = None
    vision_config: MockVisionConfig = None
    model_name_or_path: str = ""
    torch_dtype: torch.dtype = None
    
    def __post_init__(self):
        if self.text_config is None:
            self.text_config = MockTextConfig()
        if self.vision_config is None:
            self.vision_config = MockVisionConfig()


class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()
    
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, 
                           return_dict=True, return_tensors='pt'):
        return {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]])
        }


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 151643
        self.bos_token_id = 151643


class MockInputBuilder:
    def __init__(self, tokenizer, config, processor):
        self.tokenizer = tokenizer
        self.config = config
        self.processor = processor
    
    def get_shm_name_save_path(self, file_path):
        return f"/tmp/shm_{file_path.replace('/', '_')}.txt"
    
    def update_token_id(self, inputs, shm_info):
        return inputs['input_ids'].flatten()


class TestQwen3vlRouter(unittest.TestCase):
    
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.safe_from_pretrained')
    def setUp(self, mock_safe_from_pretrained):
        self.mock_processor = MockProcessor()
        mock_safe_from_pretrained.return_value = self.mock_processor
        self.router = Qwen3vlRouter(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=FAKE_CONFIG_DICT
        )
    
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.process_shared_memory')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.check_video_path')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.safe_open_image')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.Qwen3vlInputBuilder')
    @patch.object(Qwen3vlRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_text_only(self, mock_processor_prop, mock_input_builder_cls, mock_safe_open_image, 
                                     mock_check_video_path, mock_process_shm):
        """
        Test tokenize method with text-only input
        Given: A Qwen3vlRouter instance with mocked dependencies
        When: Tokenize is called with text-only input
        Then: 
        - Tokenization completes successfully (non-null result)
        - Image/video processing functions are not called
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [{'text': 'Hello, world!'}]
        result = self.router.tokenize(inputs)
        
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_not_called()
        mock_check_video_path.assert_not_called()
    
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.process_shared_memory')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.check_video_path')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.safe_open_image')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.Qwen3vlInputBuilder')
    @patch.object(Qwen3vlRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_image(self, mock_processor_prop, mock_input_builder_cls, mock_safe_open_image,
                                  mock_check_video_path, mock_process_shm):
        """
        Test tokenize method with image input
        Given: A Qwen3vlRouter instance with mocked dependencies
        When: Tokenize is called with image input
        Then: 
        - Tokenization completes successfully (non-null result)
        - Image processing function is called once
        - Video processing function is not called
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_image = MagicMock(spec=PIL.Image.Image)
        mock_safe_open_image.return_value = mock_image
        
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [{'image': '/path/to/image.jpg'}]
        result = self.router.tokenize(inputs)
        
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_called_once()
        mock_check_video_path.assert_not_called()
    
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.process_shared_memory')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.check_video_path')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.safe_open_image')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.Qwen3vlInputBuilder')
    @patch.object(Qwen3vlRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_video(self, mock_processor_prop, mock_input_builder_cls, mock_safe_open_image,
                                  mock_check_video_path, mock_process_shm):
        """
        Test tokenize method with video input
        Given: A Qwen3vlRouter instance with mocked dependencies
        When: Tokenize is called with video input
        Then: 
        - Tokenization completes successfully (non-null result)
        - Video check function is called once
        - Image processing function is not called
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_check_video_path.return_value = '/path/to/video.mp4'
        
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [{'video': '/path/to/video.mp4'}]
        result = self.router.tokenize(inputs)
        
        self.assertIsNotNone(result)
        mock_check_video_path.assert_called_once()
        mock_safe_open_image.assert_not_called()
    
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.process_shared_memory')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.check_video_path')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.safe_open_image')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.Qwen3vlInputBuilder')
    @patch.object(Qwen3vlRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_mixed_inputs(self, mock_processor_prop, mock_input_builder_cls, mock_safe_open_image,
                                         mock_check_video_path, mock_process_shm):
        """
        Test tokenize method with mixed text/image/video input
        Given: A Qwen3vlRouter instance with mocked dependencies
        When: Tokenize is called with mixed text, image and video inputs
        Then: 
        - Tokenization completes successfully (non-null result)
        - Image processing function is called once
        - Video check function is called once
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_image = MagicMock(spec=PIL.Image.Image)
        mock_safe_open_image.return_value = mock_image
        mock_check_video_path.return_value = '/path/to/video.mp4'
        
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [
            {'text': 'Describe this image and video'},
            {'image': '/path/to/image.jpg'},
            {'video': '/path/to/video.mp4'}
        ]
        result = self.router.tokenize(inputs)
        
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_called_once()
        mock_check_video_path.assert_called_once()
    
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.process_shared_memory')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.Qwen3vlInputBuilder')
    @patch.object(Qwen3vlRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_shm_name_save_path(self, mock_processor_prop, mock_input_builder_cls, mock_process_shm):
        """
        Test tokenize method with custom shm_name_save_path parameter
        Given: A Qwen3vlRouter instance with mocked dependencies
        When: Tokenize is called with text input and custom shm path
        Then: 
        - Tokenization completes successfully (non-null result)
        - process_shared_memory is called with the custom shm path parameter
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [{'text': 'Hello'}]
        shm_path = '/custom/shm/path.txt'
        result = self.router.tokenize(inputs, shm_name_save_path=shm_path)
        
        self.assertIsNotNone(result)
        mock_process_shm.assert_called_once()
        call_args = mock_process_shm.call_args
        self.assertEqual(call_args[0][1], shm_path)
    
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.Qwen3vlInputBuilder')
    @patch.object(Qwen3vlRouter, 'processor', new_callable=PropertyMock)
    @patch.object(Qwen3vlRouter, 'tokenizer', new_callable=PropertyMock)
    @patch.object(Qwen3vlRouter, 'config', new_callable=PropertyMock)
    def test_get_input_builder(self, mock_config_prop, mock_tokenizer_prop, mock_processor_prop, mock_input_builder_cls):
        """
        Test get_input_builder method invocation
        Given: A Qwen3vlRouter instance with mocked dependencies
        When: get_input_builder is called
        Then: 
        - Qwen3vlInputBuilder is initialized with correct parameters
        - Non-null input builder instance is returned
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_tokenizer_prop.return_value = self.mock_processor.tokenizer
        mock_config_prop.return_value = MockConfig()
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        
        result = self.router.get_input_builder()
        
        mock_input_builder_cls.assert_called_once_with(
            self.mock_processor.tokenizer,
            mock_config_prop.return_value,
            self.mock_processor
        )
        self.assertIsNotNone(result)

    @patch.object(Qwen3vlRouter, 'processor', new_callable=PropertyMock)
    def test_get_tokenizer(self, mock_processor_prop):
        """
        Test get_tokenizer method invocation
        Given: A Qwen3vlRouter instance with mocked processor
        When: get_tokenizer is called
        Then: Non-null tokenizer instance is returned
        """
        mock_processor_prop.return_value = self.mock_processor
        
        result = self.router.get_tokenizer()
        
        self.assertIsNotNone(result)
    
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.safe_from_pretrained')
    @patch('atb_llm.models.qwen3_vl.router_qwen3_vl.AutoProcessor')
    def test_get_processor(self, mock_auto_processor_cls, mock_safe_from_pretrained):
        """
        Test get_processor method invocation
        Given: A Qwen3vlRouter instance with mocked dependencies
        When: get_processor is called
        Then: 
        - safe_from_pretrained is called with correct parameters
        - Non-null processor instance is returned
        """
        mock_safe_from_pretrained.return_value = self.mock_processor
        
        result = self.router.get_processor()
        
        mock_safe_from_pretrained.assert_called_once_with(
            mock_auto_processor_cls,
            FAKE_MODEL_NAME_OR_PATH
        )
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
