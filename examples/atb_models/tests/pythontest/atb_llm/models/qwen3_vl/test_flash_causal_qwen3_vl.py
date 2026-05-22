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
from unittest.mock import MagicMock, patch, Mock

import torch
import numpy as np

from atb_llm.models.qwen3_vl.flash_causal_qwen3_vl import FlashQwen3vlForCausalLM, _SHM_TOKEN_LEN
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


@dataclass
class MockVisionConfig:
    """Mock vision configuration"""
    spatial_merge_size: int = 2
    deepstack_visual_indexes: list = None
    
    def __post_init__(self):
        if self.deepstack_visual_indexes is None:
            self.deepstack_visual_indexes = [0, 1, 2]


@dataclass
class MockTextConfig:
    """Mock text configuration"""
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    num_hidden_layers: int = 32
    hidden_size: int = 4096


@dataclass
class MockConfig:
    """Mock model configuration"""
    vision_config: MockVisionConfig = None
    text_config: MockTextConfig = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    quantize: str = None
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = MockVisionConfig()
        if self.text_config is None:
            self.text_config = MockTextConfig()


class TestFlashQwen3vlForCausalLM(unittest.TestCase):
    """Unit tests for FlashQwen3vlForCausalLM class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        
        self.config = MockConfig()
        self.weights = MagicMock()
        self.weights.device = torch.device("npu:0")
        self.weights.dtype = torch.float16
        
        # Patch MultiModalLLm base class
        with patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.MultiModalLLm.__init__', return_value=None):
            self.model = FlashQwen3vlForCausalLM(self.config, self.weights)
            self.model.config = self.config
            self.model.weights = self.weights
            self.model.npu_id = 0
            self.model.device = "npu:0"
            self.model.spatial_merge_size = self.config.vision_config.spatial_merge_size
            self.model.image_token_id = self.config.image_token_id
            self.model.video_token_id = self.config.video_token_id
            self.model.vision_start_token_id = self.config.vision_start_token_id
            self.model.inference_mode = None
            self.model.llm_config = None
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.Qwen3VLVisionModel')
    def test_init_vit(self, mock_vision_model_cls):
        """Test init_vit method"""
        mock_vision_model = MagicMock()
        mock_vision_model_cls.return_value = mock_vision_model
        mock_vision_model.to.return_value = mock_vision_model
        
        self.model.init_vit()
        
        self.assertEqual(self.config.vision_config.quantize, self.config.quantize)
        mock_vision_model_cls.assert_called_once_with(self.config.vision_config, self.weights)
        mock_vision_model.to.assert_called_once_with(self.weights.device)
        mock_vision_model.init_graph.assert_called_once()
        self.assertEqual(self.model.visual, mock_vision_model)
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    def test_init_llm(self, mock_llm_model_cls):
        """Test init_llm method"""
        mock_llm_model = MagicMock()
        mock_llm_model_cls.return_value = mock_llm_model
        
        self.model.init_llm()
        
        mock_llm_model_cls.assert_called_once_with(
            self.config.text_config,
            self.weights,
            llm_config=self.model.llm_config,
            inference_mode=self.model.inference_mode
        )
        self.assertEqual(self.model.language_model, mock_llm_model)
    
    def test_get_deepstack_embeds_for_llm_model_image_only(self):
        """Test _get_deepstack_embeds_for_llm_model with image only"""
        inputs_embeds = torch.randn(10, 4096, dtype=torch.float16, device="cpu")
        image_mask = torch.tensor([False, True, False, True, False, False, False, False, False, False], dtype=torch.bool)
        video_mask = None
        deepstack_image_embeds = [
            torch.randn(2, 4096, dtype=torch.float16, device="cpu"),  # 2 image tokens
            torch.randn(2, 4096, dtype=torch.float16, device="cpu"),
            torch.randn(2, 4096, dtype=torch.float16, device="cpu")
        ]
        deepstack_video_embeds = None
        
        result = self.model._get_deepstack_embeds_for_llm_model(
            inputs_embeds, image_mask, video_mask,
            deepstack_image_embeds, deepstack_video_embeds
        )
        
        self.assertEqual(len(result), len(deepstack_image_embeds))
        for embed in result:
            self.assertEqual(embed.shape, inputs_embeds.shape)
    
    def test_get_deepstack_embeds_for_llm_model_video_only(self):
        """Test _get_deepstack_embeds_for_llm_model with video only"""
        inputs_embeds = torch.randn(10, 4096, dtype=torch.float16, device="cpu")
        image_mask = None
        video_mask = torch.tensor([False, False, True, False, True, False, False, False, False, False], dtype=torch.bool)
        deepstack_image_embeds = None
        deepstack_video_embeds = [
            torch.randn(2, 4096, dtype=torch.float16, device="cpu"),
            torch.randn(2, 4096, dtype=torch.float16, device="cpu"),
            torch.randn(2, 4096, dtype=torch.float16, device="cpu")
        ]
        
        result = self.model._get_deepstack_embeds_for_llm_model(
            inputs_embeds, image_mask, video_mask,
            deepstack_image_embeds, deepstack_video_embeds
        )
        
        self.assertEqual(len(result), len(deepstack_video_embeds))
    
    def test_get_deepstack_embeds_for_llm_model_both(self):
        """Test _get_deepstack_embeds_for_llm_model with both image and video"""
        inputs_embeds = torch.randn(10, 4096, dtype=torch.float16, device="cpu")
        image_mask = torch.tensor([False, True, False, False, False, False, False, False, False, False], dtype=torch.bool)
        video_mask = torch.tensor([False, False, True, False, False, False, False, False, False, False], dtype=torch.bool)
        deepstack_image_embeds = [
            torch.randn(1, 4096, dtype=torch.float16, device="cpu"),
            torch.randn(1, 4096, dtype=torch.float16, device="cpu"),
            torch.randn(1, 4096, dtype=torch.float16, device="cpu")
        ]
        deepstack_video_embeds = [
            torch.randn(1, 4096, dtype=torch.float16, device="cpu"),
            torch.randn(1, 4096, dtype=torch.float16, device="cpu"),
            torch.randn(1, 4096, dtype=torch.float16, device="cpu")
        ]
        
        result = self.model._get_deepstack_embeds_for_llm_model(
            inputs_embeds, image_mask, video_mask,
            deepstack_image_embeds, deepstack_video_embeds
        )
        
        self.assertEqual(len(result), len(deepstack_image_embeds))
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._get_visual_features_from_shm')
    def test_get_llm_model_inputs_without_vision_info(self, mock_get_visual_features, mock_llm_model_cls):
        """Test _get_llm_model_inputs_without_vision_info method"""
        mock_llm_model = MagicMock()
        mock_llm_model.embed_tokens.return_value = torch.randn(5, 4096, dtype=torch.float16)
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        seq_len = input_ids.shape[0]
        position_ids = torch.arange(seq_len)
        mock_inputs_embeds = torch.randn(14, 4096, dtype=torch.float16)
        mock_image_mask = torch.tensor([False] * 3 + [True] + [False] * 10, dtype=torch.bool)
        mock_video_mask = None
        mock_deepstack_image_embeds = [torch.randn(1, 4096) for _ in range(3)]
        mock_deepstack_video_embeds = None
        mock_position_ids_thw = torch.randn(3, 14)
        mock_get_visual_features.return_value = (
            mock_inputs_embeds, (mock_image_mask, mock_video_mask),
            mock_deepstack_image_embeds, mock_deepstack_video_embeds, mock_position_ids_thw
        )

        inputs_embeds, position_ids_thw, deepstack_visual_embeds = \
        self.model._get_llm_model_inputs_without_vision_info(input_ids, position_ids)
        
        self.assertIsNotNone(inputs_embeds)
        self.assertEqual(position_ids_thw.shape, (3, len(input_ids)))
        self.assertEqual(len(deepstack_visual_embeds), len(self.config.vision_config.deepstack_visual_indexes))
        mock_llm_model.embed_tokens.assert_called_once_with(input_ids)
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._get_visual_features_from_shm')
    def test_get_llm_model_inputs_with_vision_info(self, mock_get_visual_features, mock_llm_model_cls):
        """Test _process_multimodal_input method"""
        mock_llm_model = MagicMock()
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        vision_start = self.model.vision_start_token_id
        image_token = self.model.image_token_id
        input_ids = torch.tensor([1, 2, vision_start, image_token, 0, 0, 0, 0, 0, 0, 0, 0, 100, 101])
        seq_len = input_ids.shape[0]
        position_ids = torch.arange(seq_len)
        
        mock_inputs_embeds = torch.randn(14, 4096, dtype=torch.float32)
        mock_image_mask = torch.tensor([False] * 3 + [True] + [False] * 10, dtype=torch.bool)
        mock_video_mask = None
        mock_deepstack_image_embeds = [torch.randn(1, 4096) for _ in range(3)]
        mock_deepstack_video_embeds = None
        mock_position_ids_thw = torch.randn(3, 14)
        
        mock_get_visual_features.return_value = (
            mock_inputs_embeds, (mock_image_mask, mock_video_mask),
            mock_deepstack_image_embeds, mock_deepstack_video_embeds, mock_position_ids_thw
        )
        
        inputs_embeds, position_ids_thw, deepstack_visual_embeds = \
        self.model._get_llm_model_inputs_with_vision_info(input_ids, position_ids)
        
        self.assertIsNotNone(inputs_embeds)
        self.assertIsNotNone(position_ids_thw)
        self.assertEqual(len(deepstack_visual_embeds), len(self.config.vision_config.deepstack_visual_indexes))
        mock_get_visual_features.assert_called_once()
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    def test_prepare_prefill_token_service_text_only(self, mock_llm_model_cls):
        """Test prepare_prefill_token_service with text-only input"""
        mock_llm_model = MagicMock()
        mock_llm_model.embed_tokens.return_value = torch.randn(5, 4096, dtype=torch.float16)
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        total_input_ids = torch.tensor([1, 2, 3, 4, 5])
        seq_len = total_input_ids.shape[0]
        total_position_ids = torch.arange(seq_len)
        input_lengths = torch.tensor([5])
        
        inputs_embeds, position_ids_thw, deepstack_visual_embeds = self.model.prepare_prefill_token_service(
            total_input_ids, total_position_ids, input_lengths
        )
        
        self.assertIsNotNone(inputs_embeds)
        self.assertEqual(position_ids_thw.shape, (3, len(total_input_ids)))
        self.assertEqual(len(deepstack_visual_embeds), len(self.config.vision_config.deepstack_visual_indexes))
        mock_llm_model.embed_tokens.assert_called_once_with(total_input_ids)
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._get_llm_model_inputs_with_vision_info')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._get_llm_model_inputs_without_vision_info')
    def test_prepare_prefill_token_service_with_vision(self, mock_process_llm, mock_process_multimodal, mock_llm_model_cls):
        """Test prepare_prefill_token_service with vision tokens"""
        mock_llm_model = MagicMock()
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        vision_start = self.model.vision_start_token_id
        total_input_ids = torch.tensor([1, 2, vision_start, 3, 4, 5, 6, 7])
        seq_len = total_input_ids.shape[0]
        total_position_ids = torch.arange(seq_len)
        input_lengths = torch.tensor([8])
        
        mock_inputs_embeds = torch.randn(8, 4096, dtype=torch.float16)
        mock_position_ids_thw = torch.randn(3, 8)
        mock_deepstack_visual_embeds = [torch.randn(8, 4096) for _ in range(3)]
        
        mock_process_multimodal.return_value = (mock_inputs_embeds, mock_position_ids_thw, mock_deepstack_visual_embeds)
        
        inputs_embeds, position_ids_thw, deepstack_visual_embeds = self.model.prepare_prefill_token_service(
            total_input_ids, total_position_ids, input_lengths
        )
        
        self.assertIsNotNone(inputs_embeds)
        self.assertIsNotNone(position_ids_thw)
        self.assertEqual(len(deepstack_visual_embeds), len(self.config.vision_config.deepstack_visual_indexes))
        mock_process_multimodal.assert_called_once()
        mock_process_llm.assert_not_called()
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM.prepare_prefill_token_service')
    def test_forward_prefill(self, mock_prepare_prefill, mock_llm_model_cls):
        """Test forward method with prefill"""
        mock_llm_model = MagicMock()
        mock_llm_model.forward.return_value = torch.randn(5, 152064, dtype=torch.float16)
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        mock_inputs_embeds = torch.randn(5, 4096, dtype=torch.float16)
        mock_position_ids_thw = torch.randn(3, 5)
        mock_deepstack_visual_embeds = [torch.randn(5, 4096) for _ in range(3)]
        mock_prepare_prefill.return_value = (mock_inputs_embeds, mock_position_ids_thw, mock_deepstack_visual_embeds)
        
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        seq_len = input_ids.shape[0]
        position_ids = torch.arange(seq_len)
        kv_cache = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))]
        block_tables = torch.tensor([[0]])
        slots = torch.tensor([0, 1, 2, 3, 4])
        input_lengths = torch.tensor([5])
        
        result = self.model.forward(
            input_ids, position_ids, True, kv_cache,
            block_tables, slots, input_lengths, 10, None
        )
        
        self.assertIsNotNone(result)
        mock_prepare_prefill.assert_called_once_with(input_ids, position_ids, input_lengths)
        mock_llm_model.forward.assert_called_once()
        call_kwargs = mock_llm_model.forward.call_args[1]
        self.assertIn('position_ids_thw', call_kwargs)
        self.assertIn('deepstack_visual_embeds', call_kwargs)
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    def test_forward_decode(self, mock_llm_model_cls):
        """Test forward method with decode"""
        mock_llm_model = MagicMock()
        mock_llm_model.embed_tokens.return_value = torch.randn(1, 4096, dtype=torch.float16)
        mock_llm_model.forward.return_value = torch.randn(1, 152064, dtype=torch.float16)
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        input_ids = torch.tensor([1])
        position_ids = torch.tensor([0])
        kv_cache = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))]
        block_tables = torch.tensor([[0]])
        slots = torch.tensor([0])
        input_lengths = torch.tensor([1])
        
        result = self.model.forward(
            input_ids, position_ids, False, kv_cache,
            block_tables, slots, input_lengths, 10, None
        )
        
        self.assertIsNotNone(result)
        mock_llm_model.embed_tokens.assert_called_once_with(input_ids)
        mock_llm_model.forward.assert_called_once()
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.get_data_from_shm')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._get_image_features')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._generate_position_ids')
    def test_get_visual_features_from_shm_image_only(self, mock_gen_pos_ids, mock_get_image_features, mock_get_data, mock_llm_model_cls):
        """Test _get_visual_features_from_shm with image only"""
        mock_llm_model = MagicMock()
        mock_llm_model.embed_tokens.return_value = torch.randn(20, 4096, dtype=torch.float32)
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        image_token = self.model.image_token_id
        input_ids_list = [
            1,
            2,
            self.config.vision_start_token_id
        ] + [image_token] * 16 + [
            self.config.vision_end_token_id,
            5,
            6
        ]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64)
        shm_info_idx = [
            100, 200,  # pixel_values_shm_name, pixel_values_shape_value
            300, 400,  # image_grid_thw_shm_name, image_grid_thw_shape_value
            0, 0,  # video shm (invalid)
            0, 0
        ]
        
        mock_pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        mock_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        mock_get_data.side_effect = [mock_pixel_values, mock_grid_thw]
        
        mock_image_embeds = [torch.randn(16, 4096)]  # 10*10//4 = 25
        mock_deepstack_image_embeds = [torch.randn(25, 4096) for _ in range(3)]
        mock_get_image_features.return_value = (mock_image_embeds, mock_deepstack_image_embeds)
        
        mock_position_ids_thw = torch.randn(3, 16)
        mock_gen_pos_ids.return_value = mock_position_ids_thw
        
        inputs_embeds, vision_mask, deepstack_image_embeds, deepstack_video_embeds, position_ids_thw = \
            self.model._get_visual_features_from_shm(input_ids, shm_info_idx)
        
        self.assertIsNotNone(inputs_embeds)
        self.assertIsNotNone(vision_mask)
        image_mask, video_mask = vision_mask
        self.assertIsNotNone(image_mask)
        self.assertIsNone(video_mask)
        self.assertIsNotNone(deepstack_image_embeds)
        self.assertIsNone(deepstack_video_embeds)
        self.assertIsNotNone(position_ids_thw)
        self.assertEqual(mock_get_data.call_count, 2)
        mock_get_image_features.assert_called_once()
        mock_gen_pos_ids.assert_called_once()
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.get_data_from_shm')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._get_image_features')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._generate_position_ids')
    def test_get_visual_features_from_shm_video_only(self, mock_gen_pos_ids, mock_get_image_features, mock_get_data, mock_llm_model_cls):
        """Test _get_visual_features_from_shm with video only"""
        mock_llm_model = MagicMock()
        mock_llm_model.embed_tokens.return_value = torch.randn(20, 4096, dtype=torch.float32)
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        video_token = self.model.video_token_id
        input_ids_list = [
            1,
            2,
            self.config.vision_start_token_id
        ] + [video_token] * 16 + [
            self.config.vision_end_token_id,
            5,
            6
        ]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64)
        shm_info_idx = [
            0, 0,  # image shm (invalid)
            0, 0,
            500, 600,  # pixel_values_videos_shm_name, pixel_values_videos_shape_value
            700, 800   # video_grid_thw_shm_name, video_grid_thw_shape_value
        ]
        
        mock_pixel_values = torch.randn(2, 3, 224, 224, dtype=torch.float32)
        mock_grid_thw = torch.tensor([[2, 4, 4]], dtype=torch.int32)
        mock_get_data.side_effect = [mock_pixel_values, mock_grid_thw]
        
        mock_video_embeds = [torch.randn(16, 4096)]  # 20*20//4 = 100
        mock_deepstack_video_embeds = [torch.randn(16, 4096) for _ in range(3)]
        mock_get_image_features.return_value = (mock_video_embeds, mock_deepstack_video_embeds)
        
        mock_position_ids_thw = torch.randn(3, 16)
        mock_gen_pos_ids.return_value = mock_position_ids_thw
        
        inputs_embeds, vision_mask, deepstack_image_embeds, deepstack_video_embeds, position_ids_thw = \
            self.model._get_visual_features_from_shm(input_ids, shm_info_idx)
        
        self.assertIsNotNone(inputs_embeds)
        self.assertIsNotNone(vision_mask)
        image_mask, video_mask = vision_mask
        self.assertIsNone(image_mask)
        self.assertIsNotNone(video_mask)
        self.assertIsNone(deepstack_image_embeds)
        self.assertIsNotNone(deepstack_video_embeds)
        self.assertIsNotNone(position_ids_thw)
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3VLTextModelForCausalLM')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.get_data_from_shm')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._get_image_features')
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM._generate_position_ids')
    def test_get_visual_features_from_shm_both(self, mock_gen_pos_ids, mock_get_image_features, mock_get_data, mock_llm_model_cls):
        """Test _get_visual_features_from_shm with both image and video"""
        mock_llm_model = MagicMock()
        mock_llm_model.embed_tokens.return_value = torch.randn(36, 4096, dtype=torch.float32)
        mock_llm_model_cls.return_value = mock_llm_model
        self.model.language_model = mock_llm_model
        
        image_token = self.model.image_token_id
        video_token = self.model.video_token_id
        input_ids = torch.tensor([1, image_token, 3, video_token, 5, 6, 7, 8, 9, 10])
        input_ids_list = [
            1,
            2,
            self.config.vision_start_token_id
        ] + [video_token] * 16 + [image_token] * 16 + [
            self.config.vision_end_token_id,
            5,
            6
        ]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64)
        shm_info_idx = [
            100, 200, 300, 400,  # image shm
            500, 600, 700, 800   # video shm
        ]
        
        mock_image_pixels = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        mock_image_grid = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        mock_video_pixels = torch.randn(2, 3, 224, 224, dtype=torch.float32)
        mock_video_grid = torch.tensor([[2, 4, 4]], dtype=torch.int32)
        mock_get_data.side_effect = [mock_image_pixels, mock_image_grid, mock_video_pixels, mock_video_grid]
        
        mock_image_embeds = [torch.randn(16, 4096)]
        mock_deepstack_image_embeds = [torch.randn(16, 4096) for _ in range(3)]
        mock_video_embeds = [torch.randn(16, 4096)]
        mock_deepstack_video_embeds = [torch.randn(16, 4096) for _ in range(3)]
        mock_get_image_features.side_effect = [
            (mock_image_embeds, mock_deepstack_image_embeds),
            (mock_video_embeds, mock_deepstack_video_embeds)
        ]
        
        mock_position_ids_thw = torch.randn(3, 10)
        mock_gen_pos_ids.return_value = mock_position_ids_thw
        
        inputs_embeds, vision_mask, deepstack_image_embeds, deepstack_video_embeds, position_ids_thw = \
            self.model._get_visual_features_from_shm(input_ids, shm_info_idx)
        
        self.assertIsNotNone(inputs_embeds)
        self.assertIsNotNone(vision_mask)
        image_mask, video_mask = vision_mask
        self.assertIsNotNone(image_mask)
        self.assertIsNotNone(video_mask)
        self.assertIsNotNone(deepstack_image_embeds)
        self.assertIsNotNone(deepstack_video_embeds)
        self.assertEqual(mock_get_data.call_count, 4)
        self.assertEqual(mock_get_image_features.call_count, 2)
    
    @patch('atb_llm.models.qwen3_vl.flash_causal_qwen3_vl.FlashQwen3vlForCausalLM.init_vit')
    def test_get_image_features(self, mock_init_vit):
        """Test _get_image_features method"""
        mock_visual = MagicMock()
        mock_visual.dtype = torch.float16
        mock_visual.spatial_merge_size = 2
        mock_visual.return_value = (
            torch.randn(41, 4096, dtype=torch.float16),
            [torch.randn(41, 4096, dtype=torch.float16) for _ in range(3)]
        )
        self.model.visual = mock_visual
        
        pixel_values = torch.randn(2, 3, 224, 224, dtype=torch.float32)
        image_grid_thw = torch.tensor([[1, 10, 10], [1, 8, 8]], dtype=torch.int64)
        
        image_embeds, deepstack_image_embeds = self.model._get_image_features(pixel_values, image_grid_thw)
        
        self.assertIsInstance(image_embeds, tuple)
        self.assertEqual(len(deepstack_image_embeds), 3)
        mock_visual.assert_called_once()
    
    def test_generate_position_ids_text_only(self):
        """Test _generate_position_ids with text-only input"""
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        image_grid_thw = None
        video_grid_thw = None
        
        position_ids = self.model._generate_position_ids(input_ids, image_grid_thw, video_grid_thw)
        
        self.assertEqual(position_ids.shape, (3, len(input_ids)))
        # All three dimensions should be sequential for text-only
        for i in range(3):
            expected = torch.arange(len(input_ids), dtype=input_ids.dtype).npu()
            torch.testing.assert_close(position_ids[i], expected)
    
    def test_generate_position_ids_with_image(self):
        """Test _generate_position_ids with image input"""
        vision_start = self.config.vision_start_token_id
        image_token = self.model.image_token_id
        input_ids = torch.tensor([1, 2, vision_start, image_token, 4, 5, 6, 7, 8, 9, 10])
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
        video_grid_thw = None
        
        position_ids = self.model._generate_position_ids(input_ids, image_grid_thw, video_grid_thw)
        
        self.assertEqual(position_ids.shape, (3, len(input_ids)))
        self.assertEqual(position_ids.dtype, input_ids.dtype)
    
    def test_generate_position_ids_with_video(self):
        """Test _generate_position_ids with video input"""
        vision_start = self.config.vision_start_token_id
        vision_end = self.config.vision_end_token_id
        video_token = self.model.video_token_id
        input_ids_list = [
            1,
            2,
            vision_start
        ] + [video_token] * 16 + [
            vision_end,
            5,
            6,
            vision_start
        ] + [video_token] * 16 + [
            vision_end
        ]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64)
        image_grid_thw = None
        video_grid_thw = torch.tensor([[2, 8, 8]], dtype=torch.int64)
        
        position_ids = self.model._generate_position_ids(input_ids, image_grid_thw, video_grid_thw)
        
        self.assertEqual(position_ids.shape, (3, len(input_ids)))
        self.assertEqual(position_ids.dtype, input_ids.dtype)
    
    def test_generate_position_ids_with_both(self):
        """Test _generate_position_ids with both image and video"""
        vision_start = self.model.vision_start_token_id
        image_token = self.model.image_token_id
        video_token = self.model.video_token_id
        vision_start = self.config.vision_start_token_id
        vision_end = self.config.vision_end_token_id
        video_token = self.model.video_token_id
        input_ids_list = [
            1,
            2,
            vision_start,
        ] + [image_token] * 4 + [
            vision_end,
            3,
            vision_start
        ] + [video_token] * 16 + [
            vision_end,
            5,
            6,
            vision_start
        ] + [video_token] * 16 + [
            vision_end
        ]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
        video_grid_thw = torch.tensor([[2, 8, 8]], dtype=torch.int64)
        
        position_ids = self.model._generate_position_ids(input_ids, image_grid_thw, video_grid_thw)
        
        self.assertEqual(position_ids.shape, (3, len(input_ids)))
        self.assertEqual(position_ids.dtype, input_ids.dtype)


if __name__ == '__main__':
    unittest.main()