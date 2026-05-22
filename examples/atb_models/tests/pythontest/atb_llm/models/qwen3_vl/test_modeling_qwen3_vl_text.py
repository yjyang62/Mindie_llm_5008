# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, Mock

import torch

from atb_llm.models.qwen3_vl.modeling_qwen3_vl_text import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextDecoderLayer,
    FlashQwen3VLTextModelForCausalLM
)
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


@dataclass
class MockRopeScaling:
    """Mock rope scaling configuration"""
    mrope_section: list = None
    
    def __post_init__(self):
        if self.mrope_section is None:
            self.mrope_section = [1, 2, 3]


@dataclass
class MockTextConfig:
    """Mock text configuration for Qwen3VLTextModel"""
    head_dim: int = 128
    rope_theta: float = 5000000.0
    rope_scaling: MockRopeScaling = None
    num_hidden_layers: int = 2
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_size: int = 2048
    intermediate_size: int = 4096
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    quantize: str = None
    quantization_config: dict = None
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = MockRopeScaling()
        if self.quantization_config is None:
            self.quantization_config = {"group_size": 128}


@dataclass
class Qwen3VLTextModelMocks:
    """Encapsulate all mock objects for the initialization test of the FlashQwen3VLTextModelForCausalLM."""
    qlen_modifier: MagicMock
    graph_manager: MagicMock
    soc_info: MagicMock
    rotary_emb: MagicMock
    load_column_multi: MagicMock
    tensor_head: MagicMock
    rms_norm: MagicMock
    decoder_layer: MagicMock
    tensor_embedding: MagicMock


def pack_qwen3vl_mocks(*mock_args) -> Qwen3VLTextModelMocks:
    """Pack mock arguments (in patch reverse order) into Qwen3VLTextModelMocks instance"""
    return Qwen3VLTextModelMocks(
        qlen_modifier=mock_args[0],
        graph_manager=mock_args[1],
        soc_info=mock_args[2],
        rotary_emb=mock_args[3],
        load_column_multi=mock_args[4],
        tensor_head=mock_args[5],
        rms_norm=mock_args[6],
        decoder_layer=mock_args[7],
        tensor_embedding=mock_args[8]
    )


class TestQwen3VLTextRotaryEmbedding(unittest.TestCase):
    """Unit tests for Qwen3VLTextRotaryEmbedding class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MockTextConfig()
        self.device = "cpu"
    
    def test_init(self):
        """Test Qwen3VLTextRotaryEmbedding initialization
        
        Given: Config with head_dim and rope_theta
        When: Initialize Qwen3VLTextRotaryEmbedding
        Then: Instance should have inv_freq buffer registered
        """
        # Given (already set via setUp)
        
        # When
        rotary_emb = Qwen3VLTextRotaryEmbedding(self.config, device=self.device)
        
        # Then
        self.assertIsNotNone(rotary_emb.inv_freq)
        self.assertEqual(len(rotary_emb.inv_freq), self.config.head_dim // 2)
        self.assertEqual(rotary_emb.config, self.config)
    
    def test_forward(self):
        """Test Qwen3VLTextRotaryEmbedding forward method
        
        Given: Initialized rotary embedding, max sequence length
        When: Call forward with max_seq_len
        Then: Should return cos and sin tensors with correct shapes
        """
        # Given
        rotary_emb = Qwen3VLTextRotaryEmbedding(self.config, device=self.device)
        max_seq_len = 1024
        
        # When
        cos, sin = rotary_emb(max_seq_len)
        
        # Then
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)
        self.assertEqual(cos.shape, (max_seq_len, self.config.head_dim // 2))
        self.assertEqual(sin.shape, (max_seq_len, self.config.head_dim // 2))
        self.assertEqual(cos.dtype, torch.float)
        self.assertEqual(sin.dtype, torch.float)


class TestQwen3VLTextAttention(unittest.TestCase):
    """Unit tests for Qwen3VLTextAttention class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MockTextConfig()
        self.prefix = "layers.0.self_attn"
        self.weights = MagicMock()
    
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorParallelRowLinear')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.calc_linear_pack_type')
    def test_init(self, mock_calc_pack_type, mock_tensor_parallel, mock_rms_norm, mock_load_column):
        """Test Qwen3VLTextAttention initialization
        
        Given: Config, prefix, and weights
        When: Initialize Qwen3VLTextAttention
        Then: Instance should have correct attributes set
        """
        # Given
        mock_calc_pack_type.return_value = "pack_type"
        mock_load_column.return_value = MagicMock()
        mock_rms_norm.return_value = MagicMock()
        mock_tensor_parallel.load.return_value = MagicMock()
        
        # When
        attention = Qwen3VLTextAttention(self.prefix, self.config, self.weights)
        
        # Then
        self.assertEqual(attention.config, self.config)
        self.assertEqual(attention.head_size, self.config.head_dim)
        self.assertIsNotNone(attention.query_key_value)
        self.assertIsNotNone(attention.q_norm)
        self.assertIsNotNone(attention.k_norm)
        self.assertIsNotNone(attention.o_proj)
    
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorParallelRowLinear')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.calc_linear_pack_type')
    def test_init_with_quantize(self, mock_calc_pack_type, mock_tensor_parallel, mock_rms_norm, mock_load_column):
        """Test Qwen3VLTextAttention initialization with quantization
        
        Given: Config with quantize="w8a8sc"
        When: Initialize Qwen3VLTextAttention
        Then: Should use different qkv_names
        """
        # Given
        self.config.quantize = "w8a8sc"
        mock_calc_pack_type.return_value = "pack_type"
        mock_load_column.return_value = MagicMock()
        mock_rms_norm.return_value = MagicMock()
        mock_tensor_parallel.load.return_value = MagicMock()
        
        # When
        attention = Qwen3VLTextAttention(self.prefix, self.config, self.weights)
        
        # Then
        self.assertEqual(attention.qkv_names, [f"{self.prefix}.query_key_value"])


class TestQwen3VLTextMLP(unittest.TestCase):
    """Unit tests for Qwen3VLTextMLP class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MockTextConfig()
        self.prefix = "layers.0.mlp"
        self.weights = MagicMock()
    
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.MLP.__init__')
    def test_init(self, mock_mlp_init):
        """Test Qwen3VLTextMLP initialization
        
        Given: Config, prefix, and weights
        When: Initialize Qwen3VLTextMLP
        Then: Should set correct gate_up_names and pack_name
        """
        # Given
        mock_mlp_init.return_value = None

        def mock_load_weights():
            pass
        Qwen3VLTextMLP.load_weights = Mock(side_effect=mock_load_weights)
        
        # When
        mlp = Qwen3VLTextMLP(self.prefix, self.config, self.weights)
        
        # Then
        self.assertEqual(mlp.gate_up_names, [f'{self.prefix}.gate_proj', f'{self.prefix}.up_proj'])
        self.assertEqual(mlp.pack_name, f'{self.prefix}.gate_up_proj')
        self.assertEqual(mlp.down_name, f'{self.prefix}.down_proj')
        mock_mlp_init.assert_called_once_with(self.prefix, self.config, self.weights)


class TestQwen3VLTextDecoderLayer(unittest.TestCase):
    """Unit tests for Qwen3VLTextDecoderLayer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MockTextConfig()
        self.layer_id = 0
        self.prefix = ""
        self.weights = MagicMock()
    
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextAttention')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextMLP')
    def test_init(self, mock_mlp, mock_attention):
        """Test Qwen3VLTextDecoderLayer initialization
        
        Given: Layer ID, config, weights, and prefix
        When: Initialize Qwen3VLTextDecoderLayer
        Then: Should create self_attn and mlp components
        """
        # Given
        mock_attention.return_value = MagicMock()
        mock_mlp.return_value = MagicMock()

        def mock_load_weights():
            pass
        Qwen3VLTextDecoderLayer.load_weights = Mock(side_effect=mock_load_weights)
        
        # When
        Qwen3VLTextDecoderLayer(self.layer_id, self.config, self.weights, self.prefix)
        
        # Then
        mock_attention.assert_called_once_with(
            prefix=f"{self.prefix}.layers.0.self_attn",
            config=self.config,
            weights=self.weights
        )
        mock_mlp.assert_called_once_with(
            prefix=f"{self.prefix}.layers.0.mlp",
            config=self.config,
            weights=self.weights
        )


class TestFlashQwen3VLTextModelForCausalLM(unittest.TestCase):
    """Unit tests for FlashQwen3VLTextModelForCausalLM class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        
        self.config = MockTextConfig()
        self.config.rope_scaling = Mock()
        self.config.model_type = "qwen3"
        self.config.quantization_config = Mock()
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.process_group = MagicMock()
        self.weights.process_group.rank.return_value = 0
        self.weights.process_group.size.return_value = 1
        
        # Patch common dependencies
        self.patcher_load_atb = patch('atb_llm.models.base.flash_causal_lm.load_atb_speed')
        self.mock_load_atb = self.patcher_load_atb.start()
        self.addCleanup(self.patcher_load_atb.stop)
    
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextDecoderLayer')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorHead')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextRotaryEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.NPUSocInfo')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.ATBGraphManager')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.QLenModifier')
    def test_init(self, *mock_args):
        """Test FlashQwen3VLTextModelForCausalLM initialization
        
        Given: Config and weights
        When: Initialize FlashQwen3VLTextModelForCausalLM
        Then: Should initialize all components correctly
        """
        # 
        mocks = pack_qwen3vl_mocks(*mock_args)

        # Given: Configure mock return values
        mocks.tensor_embedding.return_value = MagicMock()
        mocks.decoder_layer.return_value = None
        mocks.rms_norm.return_value = MagicMock()
        mocks.tensor_head.load_weight.return_value = MagicMock()
        mocks.load_column_multi.return_value = MagicMock()
        
        # Configure rotary embedding mock instance
        mock_rotary_emb_instance = MagicMock()
        mock_rotary_emb_instance.return_value = (torch.zeros(100, 64), torch.zeros(100, 64))
        mocks.rotary_emb.return_value = mock_rotary_emb_instance
        
        # Configure NPU soc info mock instance
        mock_soc_info_instance = MagicMock()
        mock_soc_info_instance.need_nz = False
        mock_soc_info_instance.is_300i.return_value = False
        mock_soc_info_instance.communication_backend = "hccl"
        mock_soc_info_instance.soc_version = 310
        mocks.soc_info.return_value = mock_soc_info_instance
        
        mocks.graph_manager.return_value = MagicMock()
        mocks.qlen_modifier.return_value = MagicMock()
        
        # When: Initialize the model
        model = FlashQwen3VLTextModelForCausalLM(self.config, self.weights)
        
        # Then: Verify model components are initialized correctly
        self.assertEqual(model.config, self.config)
        self.assertIsNotNone(model.embed_tokens)
        self.assertEqual(len(model.layers), self.config.num_hidden_layers)
        self.assertIsNotNone(model.norm)
        self.assertIsNotNone(model.lm_head)
    
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.FlashQwen3VLTextModelForCausalLM.__init__')
    def test_init_ascend_operations(self, mock_init):
        """Test init_ascend_operations method
        
        Given: Initialized model
        When: Call init_ascend_operations
        Then: Should pass without error (method is empty)
        """
        # Given
        mock_init.return_value = None
        model = FlashQwen3VLTextModelForCausalLM.__new__(FlashQwen3VLTextModelForCausalLM)
        model.config = self.config
        
        # When & Then
        model.init_ascend_operations(self.config)
        # Should not raise any exception
    
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.FlashQwen3VLTextModelForCausalLM.__init__')
    def test_apply_interleaved_mrope(self, mock_init):
        """Test apply_interleaved_mrope method
        
        Given: Embedding tensor and mrope_section
        When: Call apply_interleaved_mrope
        Then: Should return modified embedding tensor
        """
        # Given
        mock_init.return_value = None
        model = FlashQwen3VLTextModelForCausalLM.__new__(FlashQwen3VLTextModelForCausalLM)
        
        # Create mock embedding with shape (3, seq_len, head_dim)
        seq_len = 10
        head_dim = 64
        emb = torch.randn(3, seq_len, head_dim)
        mrope_section = [1, 2, 3]
        
        # When
        result = model.apply_interleaved_mrope(emb, mrope_section)
        
        # Then
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (seq_len, head_dim))
    
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.FlashQwen3VLTextModelForCausalLM.__init__')
    def test_update_thw_cos_sin(self, mock_init):
        """Test update_thw_cos_sin method
        
        Given: Position IDs THW and initialized model with cos/sin embeddings
        When: Call update_thw_cos_sin
        Then: Should return modified cos and sin tensors
        """
        # Given
        mock_init.return_value = None
        model = FlashQwen3VLTextModelForCausalLM.__new__(FlashQwen3VLTextModelForCausalLM)
        model.config = self.config
        model.mrope_section = [1, 2, 3]
        
        # Mock cos_embed and sin_embed
        seq_len = 10
        head_dim = 64
        model.cos_embed = torch.randn(seq_len, head_dim)
        model.sin_embed = torch.randn(seq_len, head_dim)
        
        # Mock position_ids_thw with shape (3, seq_len)
        position_ids_thw = torch.randint(0, seq_len, (3, seq_len))
        
        # Mock apply_interleaved_mrope
        model.apply_interleaved_mrope = MagicMock(side_effect=lambda x, y: x[0])
        
        # When
        cos, sin = model.update_thw_cos_sin(position_ids_thw)
        
        # Then
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)
        self.assertEqual(cos.shape[0], seq_len)
        self.assertEqual(sin.shape[0], seq_len)
        model.apply_interleaved_mrope.assert_called()
    
    def get_model_inputs(self, device, is_prefill=True):
        """Helper method to create model inputs"""
        max_seq_len = 6
        input_ids = torch.randint(self.config.vocab_size, (max_seq_len,))
        position_ids = torch.tensor(range(max_seq_len)).to(device)
        kv_cache = [(torch.zeros([9, 128, 8, 128]), torch.zeros([9, 128, 8, 128]))]
        block_tables = torch.tensor([[0]])
        slots = torch.tensor(range(max_seq_len))
        input_lengths = torch.tensor([max_seq_len])
        lm_head_indices = None if is_prefill else torch.tensor([0])
        model_inputs = (input_ids, position_ids, is_prefill, kv_cache, block_tables,
                slots, input_lengths, max_seq_len, lm_head_indices)
        return model_inputs

    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextDecoderLayer')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorHead')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextRotaryEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.NPUSocInfo')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.ATBGraphManager')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.QLenModifier')
    def test_prepare_inputs_for_ascend_prefill(self, *mock_args):
        """Test prepare_inputs_for_ascend with prefill mode
        
        Given: Model inputs for prefill
        When: Call prepare_inputs_for_ascend
        Then: Should prepare ACL operation inputs correctly
        """
        # Pack mock args into dataclass instance
        mocks = pack_qwen3vl_mocks(*mock_args)

        # Given: Configure mock return values
        mocks.tensor_embedding.return_value = MagicMock()
        mocks.decoder_layer.return_value = None
        mocks.rms_norm.return_value = MagicMock()
        mocks.tensor_head.load_weight.return_value = MagicMock()
        mocks.load_column_multi.return_value = MagicMock()
        
        # Configure rotary embedding mock
        mock_rotary_emb_instance = MagicMock()
        mock_rotary_emb_instance.return_value = (torch.zeros(100, 64), torch.zeros(100, 64))
        mocks.rotary_emb.return_value = mock_rotary_emb_instance
        
        # Configure NPU soc info mock
        mock_soc_info_instance = MagicMock()
        mock_soc_info_instance.need_nz = False
        mock_soc_info_instance.is_300i.return_value = False
        mock_soc_info_instance.communication_backend = "hccl"
        mock_soc_info_instance.soc_version = 310
        mocks.soc_info.return_value = mock_soc_info_instance
        
        mocks.graph_manager.return_value = MagicMock()
        mock_qlen_modifier_instance = MagicMock()
        mocks.qlen_modifier.return_value = mock_qlen_modifier_instance
        
        # Initialize model and inputs
        model = FlashQwen3VLTextModelForCausalLM(self.config, self.weights)
        model.device = torch.device("cpu")
        model.dtype = torch.float16
        model.max_base_len = 100
        model.cos_embed = torch.zeros(100, 64)
        model.sin_embed = torch.zeros(100, 64)
        model.placeholder = torch.zeros(1)
        model_inputs = self.get_model_inputs(model.device, is_prefill=True)
        
        # When: Call target method
        acl_inputs, acl_param = model.prepare_inputs_for_ascend(*model_inputs)
        
        # Then: Verify results
        self.assertIsNotNone(acl_inputs)
        self.assertIsNotNone(acl_param)
        self.assertIsInstance(acl_param, str)
        mock_qlen_modifier_instance.modify_inputs.assert_called_once()

    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextDecoderLayer')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorHead')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextRotaryEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.NPUSocInfo')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.ATBGraphManager')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.QLenModifier')
    def test_prepare_inputs_for_ascend_decode(self, *mock_args):
        """Test prepare_inputs_for_ascend with decode mode
        
        Given: Model inputs for decode
        When: Call prepare_inputs_for_ascend
        Then: Should prepare ACL operation inputs correctly
        """
        # Pack mock args into dataclass instance
        mocks = pack_qwen3vl_mocks(*mock_args)

        # Given: Configure mock return values
        mocks.tensor_embedding.return_value = MagicMock()
        mocks.decoder_layer.return_value = None
        mocks.rms_norm.return_value = MagicMock()
        mocks.tensor_head.load_weight.return_value = MagicMock()
        mocks.load_column_multi.return_value = MagicMock()
        
        # Configure rotary embedding mock
        mock_rotary_emb_instance = MagicMock()
        mock_rotary_emb_instance.return_value = (torch.zeros(100, 64), torch.zeros(100, 64))
        mocks.rotary_emb.return_value = mock_rotary_emb_instance
        
        # Configure NPU soc info mock
        mock_soc_info_instance = MagicMock()
        mock_soc_info_instance.need_nz = False
        mock_soc_info_instance.is_300i.return_value = False
        mock_soc_info_instance.communication_backend = "hccl"
        mock_soc_info_instance.soc_version = 310
        mocks.soc_info.return_value = mock_soc_info_instance
        
        mocks.graph_manager.return_value = MagicMock()
        mock_qlen_modifier_instance = MagicMock()
        mocks.qlen_modifier.return_value = mock_qlen_modifier_instance
        
        # Initialize model and inputs
        model = FlashQwen3VLTextModelForCausalLM(self.config, self.weights)
        model.device = torch.device("cpu")
        model.dtype = torch.float16
        model.max_base_len = 100
        model.cos_embed = torch.zeros(100, 64)
        model.sin_embed = torch.zeros(100, 64)
        model.placeholder = torch.zeros(1)
        model.lm_head_indices_fake = torch.tensor([0])
        model_inputs = self.get_model_inputs(model.device, is_prefill=False)
        
        # When: Call target method
        acl_inputs, acl_param = model.prepare_inputs_for_ascend(*model_inputs)
        
        # Then: Verify results
        self.assertIsNotNone(acl_inputs)
        self.assertIsNotNone(acl_param)
        mock_qlen_modifier_instance.modify_inputs.assert_called_once()

    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextDecoderLayer')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorHead')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextRotaryEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.NPUSocInfo')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.ATBGraphManager')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.QLenModifier')
    def test_init_kvcache(self, *mock_args):
        """Test init_kvcache method
        
        Given: Model with KV cache
        When: Call init_kvcache
        Then: Should initialize KV cache in graph manager
        """
        # Pack mock args into dataclass instance
        mocks = pack_qwen3vl_mocks(*mock_args)

        # Given: Configure mock return values
        mocks.tensor_embedding.return_value = MagicMock()
        mocks.decoder_layer.return_value = None
        mocks.rms_norm.return_value = MagicMock()
        mocks.tensor_head.load_weight.return_value = MagicMock()
        mocks.load_column_multi.return_value = MagicMock()
        
        # Configure rotary embedding mock
        mock_rotary_emb_instance = MagicMock()
        mock_rotary_emb_instance.return_value = (torch.zeros(100, 64), torch.zeros(100, 64))
        mocks.rotary_emb.return_value = mock_rotary_emb_instance
        
        # Configure NPU soc info mock
        mock_soc_info_instance = MagicMock()
        mock_soc_info_instance.need_nz = False
        mock_soc_info_instance.is_300i.return_value = False
        mock_soc_info_instance.communication_backend = "hccl"
        mock_soc_info_instance.soc_version = 310
        mocks.soc_info.return_value = mock_soc_info_instance
        
        mock_graph_manager_instance = MagicMock()
        mocks.graph_manager.return_value = mock_graph_manager_instance
        mocks.qlen_modifier.return_value = MagicMock()
        
        # Initialize model and KV cache
        model = FlashQwen3VLTextModelForCausalLM(self.config, self.weights)
        model.ascend_kcache_id = None
        model.ascend_vcache_id = None
        kv_cache = [(torch.zeros([9, 128, 8, 128]), torch.zeros([9, 128, 8, 128]))]
        
        # When: Call target method
        model.init_kvcache(kv_cache)
        
        # Then: Verify results
        mock_graph_manager_instance.set_kv_cache.assert_called_once()
        self.assertIsNotNone(model.ascend_kcache_id)
        self.assertIsNotNone(model.ascend_vcache_id)

    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextDecoderLayer')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorHead')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextRotaryEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.NPUSocInfo')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.ATBGraphManager')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.QLenModifier')
    def test_execute_ascend_operator(self, *mock_args):
        """Test execute_ascend_operator method
        
        Given: ACL inputs and parameters
        When: Call execute_ascend_operator
        Then: Should execute graph and return hidden state
        """
        # Pack mock args into dataclass instance
        mocks = pack_qwen3vl_mocks(*mock_args)

        # Given: Configure mock return values
        mocks.tensor_embedding.return_value = MagicMock()
        mocks.decoder_layer.return_value = None
        mocks.rms_norm.return_value = MagicMock()
        mocks.tensor_head.load_weight.return_value = MagicMock()
        mocks.load_column_multi.return_value = MagicMock()
        
        # Configure rotary embedding mock
        mock_rotary_emb_instance = MagicMock()
        mock_rotary_emb_instance.return_value = (torch.zeros(100, 64), torch.zeros(100, 64))
        mocks.rotary_emb.return_value = mock_rotary_emb_instance
        
        # Configure NPU soc info mock
        mock_soc_info_instance = MagicMock()
        mock_soc_info_instance.need_nz = False
        mock_soc_info_instance.is_300i.return_value = False
        mock_soc_info_instance.communication_backend = "hccl"
        mock_soc_info_instance.soc_version = 310
        mocks.soc_info.return_value = mock_soc_info_instance
        
        # Configure graph manager mock
        mock_graph_manager_instance = MagicMock()
        mock_graph_manager_instance.select_and_execute.return_value = [torch.zeros(1, 10, 2048)]
        mocks.graph_manager.return_value = mock_graph_manager_instance
        mocks.qlen_modifier.return_value = MagicMock()
        
        # Initialize model and inputs
        model = FlashQwen3VLTextModelForCausalLM(self.config, self.weights)
        acl_inputs = [torch.zeros(1, 10)] * 12
        acl_param = json.dumps({"seqLen": [10]})
        is_prefill = True
        
        # When: Call target method
        result = model.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        
        # Then: Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, torch.Tensor)
        mock_graph_manager_instance.select_and_execute.assert_called_once()

    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextDecoderLayer')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorHead')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextRotaryEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.NPUSocInfo')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.ATBGraphManager')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.QLenModifier')
    def test_forward_prefill(self, *mock_args):
        """Test forward method with prefill mode
        
        Given: Model inputs for prefill with position_ids_thw
        When: Call forward
        Then: Should return logits
        """
        # Pack mock args into dataclass instance
        mocks = pack_qwen3vl_mocks(*mock_args)

        # Given: Configure mock return values
        mocks.tensor_embedding.return_value = MagicMock()
        mocks.decoder_layer.return_value = None
        mocks.rms_norm.return_value = MagicMock()
        mocks.tensor_head.load_weight.return_value = MagicMock()
        mocks.load_column_multi.return_value = MagicMock()
        
        # Configure rotary embedding mock
        mock_rotary_emb_instance = MagicMock()
        mock_rotary_emb_instance.return_value = (torch.zeros(100, 64), torch.zeros(100, 64))
        mocks.rotary_emb.return_value = mock_rotary_emb_instance
        
        # Configure NPU soc info mock
        mock_soc_info_instance = MagicMock()
        mock_soc_info_instance.need_nz = False
        mock_soc_info_instance.is_300i.return_value = False
        mock_soc_info_instance.communication_backend = "hccl"
        mock_soc_info_instance.soc_version = 310
        mocks.soc_info.return_value = mock_soc_info_instance
        
        # Configure graph manager mock
        mock_graph_manager_instance = MagicMock()
        mock_graph_manager_instance.select_and_execute.return_value = [torch.zeros(1, 10, 2048)]
        mocks.graph_manager.return_value = mock_graph_manager_instance
        
        mock_qlen_modifier_instance = MagicMock()
        mocks.qlen_modifier.return_value = mock_qlen_modifier_instance
        
        # Initialize model and mock dependencies
        model = FlashQwen3VLTextModelForCausalLM(self.config, self.weights)
        model.device = torch.device("cpu")
        model.dtype = torch.float16
        model.max_base_len = 100
        model.cos_embed = torch.zeros(100, 64)
        model.sin_embed = torch.zeros(100, 64)
        model.cos_embed_decode = torch.zeros(1, 1, 128)
        model.sin_embed_decode = torch.zeros(1, 1, 128)
        model.placeholder = torch.zeros(1)
        model.ascend_weight = []  # Trigger init_ascend_weight
        model.mrope_section = [1, 2, 3]
        
        # Mock internal methods
        model.init_ascend_weight = MagicMock()
        model.init_kvcache = MagicMock()
        model.prepare_inputs_for_ascend = MagicMock(return_value=([torch.zeros(1, 10)] * 12, json.dumps({"seqLen": [10]})))
        model.update_thw_cos_sin = MagicMock(return_value=(torch.zeros(10, 64), torch.zeros(10, 64)))
        model.execute_ascend_operator = MagicMock(return_value=torch.zeros(1, 10, 2048))
        
        # Prepare inputs
        model_inputs = self.get_model_inputs(model.device, is_prefill=True)
        position_ids_thw = torch.randint(0, 100, (3, 10))
        deepstack_visual_embeds = []
        
        # When: Call forward method
        result = model.forward(*model_inputs, position_ids_thw=position_ids_thw, deepstack_visual_embeds=deepstack_visual_embeds)
        
        # Then: Verify method calls and results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, torch.Tensor)
        model.init_ascend_weight.assert_called_once()
        model.init_kvcache.assert_called_once()
        model.prepare_inputs_for_ascend.assert_called_once()
        model.update_thw_cos_sin.assert_called_once()
        model.execute_ascend_operator.assert_called_once()

    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextDecoderLayer')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.RMSNorm')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.TensorHead')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.load_column_multi')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.Qwen3VLTextRotaryEmbedding')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.NPUSocInfo')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.ATBGraphManager')
    @patch('atb_llm.models.qwen3_vl.modeling_qwen3_vl_text.QLenModifier')
    def test_forward_decode(self, *mock_args):
        """Test forward method with decode mode
        
        Given: Model inputs for decode
        When: Call forward
        Then: Should return logits using decode cos/sin embeddings
        """
        # Pack mock args into dataclass instance
        mocks = pack_qwen3vl_mocks(*mock_args)

        # Given: Configure mock return values
        mocks.tensor_embedding.return_value = MagicMock()
        mocks.decoder_layer.return_value = None
        mocks.rms_norm.return_value = MagicMock()
        mocks.tensor_head.load_weight.return_value = MagicMock()
        mocks.load_column_multi.return_value = MagicMock()
        
        # Configure rotary embedding mock
        mock_rotary_emb_instance = MagicMock()
        mock_rotary_emb_instance.return_value = (torch.zeros(100, 64), torch.zeros(100, 64))
        mocks.rotary_emb.return_value = mock_rotary_emb_instance
        
        # Configure NPU soc info mock
        mock_soc_info_instance = MagicMock()
        mock_soc_info_instance.need_nz = False
        mock_soc_info_instance.is_300i.return_value = False
        mock_soc_info_instance.communication_backend = "hccl"
        mock_soc_info_instance.soc_version = 310
        mocks.soc_info.return_value = mock_soc_info_instance
        
        # Configure graph manager mock
        mock_graph_manager_instance = MagicMock()
        mock_graph_manager_instance.select_and_execute.return_value = [torch.zeros(1, 1, 2048)]
        mocks.graph_manager.return_value = mock_graph_manager_instance
        
        mock_qlen_modifier_instance = MagicMock()
        mocks.qlen_modifier.return_value = mock_qlen_modifier_instance
        
        # Initialize model and mock dependencies
        model = FlashQwen3VLTextModelForCausalLM(self.config, self.weights)
        model.device = torch.device("cpu")
        model.dtype = torch.float16
        model.max_base_len = 100
        model.cos_embed = torch.zeros(100, 64)
        model.sin_embed = torch.zeros(100, 64)
        model.cos_embed_decode = torch.zeros(1, 1, 128)
        model.sin_embed_decode = torch.zeros(1, 1, 128)
        model.placeholder = torch.zeros(1)
        model.ascend_weight = []  # Trigger init_ascend_weight
        model.mrope_section = [1, 2, 3]
        
        # Mock internal methods
        model.init_ascend_weight = MagicMock()
        model.init_kvcache = MagicMock()
        model.prepare_inputs_for_ascend = MagicMock(return_value=([torch.zeros(1, 1)] * 12, json.dumps({"seqLen": [1]})))
        model.execute_ascend_operator = MagicMock(return_value=torch.zeros(1, 1, 2048))
        
        # Prepare inputs
        model_inputs = self.get_model_inputs(model.device, is_prefill=False)
        
        # When: Call forward method
        result = model.forward(*model_inputs)
        
        # Then: Verify method calls and results
        self.assertIsNotNone(result)
        self.assertIsInstance(result, torch.Tensor)
        model.init_ascend_weight.assert_called_once()
        model.init_kvcache.assert_called_once()
        model.prepare_inputs_for_ascend.assert_called_once()
        model.execute_ascend_operator.assert_called_once()
        
        # Verify decode embeddings shape
        self.assertEqual(model.cos_embed_decode.shape[0], 1)
        self.assertEqual(model.sin_embed_decode.shape[0], 1)

if __name__ == '__main__':
    unittest.main()