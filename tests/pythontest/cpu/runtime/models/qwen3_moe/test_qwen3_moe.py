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
from unittest.mock import patch, MagicMock

import torch

from mindie_llm.runtime.models.qwen3_moe.qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeLayer,
    Qwen3MoeModel,
    Qwen3MoeForCausalLM
)
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase


class TestQwen3MoeSparseMoeBlock(unittest.TestCase):

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.FusedMoE")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.ReplicatedLinear")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_initialization(self, mock_get_parallel_info, mock_replicated_linear, mock_fused_moe):
        """Verify Qwen3MoeSparseMoeBlock initializes correctly."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_gate_linear = MagicMock()
        mock_moe_experts = MagicMock()
        mock_replicated_linear.return_value = mock_gate_linear
        mock_fused_moe.return_value = mock_moe_experts

        # Mock config
        config = MagicMock(
            hidden_size=4096,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=14336,
        )
        prefix = "model.layers.0.mlp"
        quant_config = MagicMock(spec=QuantizationConfigBase)

        # Init instance
        moe_block = Qwen3MoeSparseMoeBlock(config, prefix, quant_config=quant_config)

        # Assert attrs
        self.assertEqual(moe_block.topk_num, 2)
        self.assertEqual(moe_block.expert_num, 8)
        self.assertIsInstance(moe_block.gate, MagicMock)
        self.assertIsInstance(moe_block.experts, MagicMock)
        # Assert ReplicatedLinear init param
        mock_replicated_linear.assert_called_once_with(
            4096, 8, bias=False, quant_config=quant_config, prefix=f"{prefix}.gate"
        )
        # Assert FusedMoE init param
        mock_fused_moe.assert_called_once_with(
            num_experts=8, topk_num=2, hidden_size=4096, intermediate_size=14336,
            quant_config=quant_config, prefix=f"{prefix}.experts",
            suffix=["gate_proj", "down_proj", "up_proj"]
        )

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.select_experts")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.FusedMoE")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.ReplicatedLinear")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_forward_pass(self, mock_get_parallel_info, mock_replicated_linear, mock_fused_moe, mock_select_experts):
        """Verify forward pass of Qwen3MoeSparseMoeBlock."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_gate_linear = MagicMock()
        mock_moe_experts = MagicMock()
        mock_replicated_linear.return_value = mock_gate_linear
        mock_fused_moe.return_value = mock_moe_experts

        # Mock config
        config = MagicMock(
            hidden_size=4096,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=14336,
        )
        prefix = "model.layers.0.mlp"

        # Init instance
        moe_block = Qwen3MoeSparseMoeBlock(config, prefix)

        # Setup test inputs/outputs
        hidden_states = torch.randn(3, 5, 4096)
        router_logits = torch.randn(3, 5, 8)
        topk_weights = torch.randn(3, 5, 2)
        topk_ids = torch.randint(0, 8, (3, 5, 2))
        moe_output = torch.randn(3, 5, 4096)

        # Mock forward logic
        mock_gate_linear.return_value = router_logits
        mock_select_experts.return_value = (topk_weights, topk_ids)
        mock_moe_experts.return_value = moe_output

        # Run forward pass
        output = moe_block(hidden_states)

        # Verify calls
        mock_gate_linear.assert_called_once_with(hidden_states)
        mock_select_experts.assert_called_once_with(
            hidden_states=hidden_states, router_logits=router_logits, top_k=2,
            use_grouped_topk=False, renormalize=True, topk_group=1, num_expert_group=1,
            scoring_func="softmax", routed_scaling_factor=1.0, e_score_correction_bias=None,
            global_num_experts=8
        )
        mock_moe_experts.assert_called_once_with(hidden_states, topk_weights, topk_ids)

        # Verify output shape
        self.assertEqual(output.shape, (3, 5, 4096))


class TestQwen3MoeLayer(unittest.TestCase):

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3Attention")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3MoeSparseMoeBlock")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.RMSNorm")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_initialization(self, mock_get_parallel_info, mock_norm, mock_moe_block, mock_attention):
        """Verify Qwen3MoeLayer initializes correctly."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_self_attn = MagicMock()
        mock_mlp_moe = MagicMock()
        mock_attention.return_value = mock_self_attn
        mock_moe_block.return_value = mock_mlp_moe
        mock_norm.side_effect = [MagicMock(), MagicMock()]

        # Mock config
        config = MagicMock(
            hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
            rms_norm_eps=1e-6, use_qk_norm=True, attention_bias=False,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=14336
        )
        prefix = "model"
        layer_idx = 0
        quant_config = MagicMock(spec=QuantizationConfigBase)

        # Init instance
        layer = Qwen3MoeLayer(config, prefix, layer_idx, quant_config=quant_config)

        # Assert attrs
        self.assertEqual(layer.prefix, "model.layers.0")
        self.assertEqual(layer.layer_idx, 0)
        self.assertIsInstance(layer.self_attn, MagicMock)
        self.assertIsInstance(layer.mlp, MagicMock)
        self.assertIsInstance(layer.input_layernorm, MagicMock)
        self.assertIsInstance(layer.post_attention_layernorm, MagicMock)

        # Verify sub-module init calls
        mock_attention.assert_called_once()
        mock_moe_block.assert_called_once()
        self.assertEqual(mock_norm.call_count, 2)

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3Attention")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3MoeSparseMoeBlock")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.RMSNorm")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_forward_pass(self, mock_get_parallel_info, mock_norm, mock_moe_block, mock_attention):
        """Verify forward pass of Qwen3MoeLayer."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_self_attn = MagicMock()
        mock_mlp_moe = MagicMock()
        mock_input_norm = MagicMock()
        mock_post_norm = MagicMock()

        mock_attention.return_value = mock_self_attn
        mock_moe_block.return_value = mock_mlp_moe
        mock_norm.side_effect = [mock_input_norm, mock_post_norm]

        # Mock config
        config = MagicMock(
            hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
            rms_norm_eps=1e-6, use_qk_norm=True, attention_bias=False,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=14336
        )
        prefix = "model.layers"
        layer_idx = 0

        # Init instance
        layer = Qwen3MoeLayer(config, prefix, layer_idx)

        # Setup test inputs/outputs
        positions = torch.tensor([0, 1, 2, 3, 4])
        hidden_states = torch.randn(3, 5, 4096)
        residual = None

        attn_norm_out = torch.randn(3, 5, 4096)
        attn_out = torch.randn(3, 5, 4096)
        mlp_norm_out = (torch.randn(3, 5, 4096), torch.randn(3, 5, 4096))
        mlp_out = torch.randn(3, 5, 4096)

        # Mock forward logic
        mock_input_norm.return_value = attn_norm_out
        mock_self_attn.return_value = attn_out
        mock_post_norm.return_value = mlp_norm_out
        mock_mlp_moe.return_value = mlp_out

        # Run forward pass
        output, res_out = layer(positions, hidden_states, residual)

        # Verify calls
        mock_input_norm.assert_called_once_with(hidden_states)
        mock_self_attn.assert_called_once_with(positions=positions, hidden_states=attn_norm_out)
        mock_post_norm.assert_called_once_with(attn_out, hidden_states)
        mock_mlp_moe.assert_called_once_with(mlp_norm_out[0])

        # Verify output shapes
        self.assertEqual(output.shape, (3, 5, 4096))
        self.assertEqual(res_out.shape, (3, 5, 4096))


class TestQwen3MoeModel(unittest.TestCase):

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3MoeLayer")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.VocabParallelEmbedding")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.RMSNorm")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_initialization(self, mock_get_parallel_info, mock_norm, mock_embedding, mock_layer):
        """Verify Qwen3MoeModel initializes correctly."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_embed_tokens = MagicMock()
        mock_layer_instance = MagicMock(spec=torch.nn.Module)
        mock_final_norm = MagicMock()

        mock_embedding.return_value = mock_embed_tokens
        mock_layer.return_value = mock_layer_instance
        mock_norm.return_value = mock_final_norm

        # Mock config
        config = MagicMock(
            vocab_size=151936, hidden_size=4096, num_hidden_layers=32,
            rms_norm_eps=1e-6, num_experts=8, num_experts_per_tok=2
        )
        quant_config = MagicMock(spec=QuantizationConfigBase)

        # Init instance
        model = Qwen3MoeModel(config, quant_config=quant_config)

        # Assert attrs
        self.assertIsInstance(model.embed_tokens, MagicMock)
        self.assertEqual(len(model.layers), 32)
        self.assertIsInstance(model.norm, MagicMock)

        # Verify sub-module init
        mock_embedding.assert_called_once_with(
            151936, 4096, quant_config=quant_config, prefix="model.embed_tokens"
        )
        self.assertEqual(mock_layer.call_count, 32)
        mock_norm.assert_called_once_with(4096, 1e-6, quant_config=quant_config, prefix="model.norm")

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3MoeLayer")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.VocabParallelEmbedding")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.RMSNorm")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_forward_pass(self, mock_get_parallel_info, mock_norm, mock_embedding, mock_layer):
        """Verify forward pass of Qwen3MoeModel."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_embed_tokens = MagicMock()
        mock_layer_instance = MagicMock(spec=torch.nn.Module)
        mock_final_norm = MagicMock()

        mock_embedding.return_value = mock_embed_tokens
        mock_layer.return_value = mock_layer_instance
        mock_norm.return_value = mock_final_norm

        # Mock config
        config = MagicMock(
            vocab_size=151936, hidden_size=4096, num_hidden_layers=32,
            rms_norm_eps=1e-6, num_experts=8, num_experts_per_tok=2
        )

        # Init instance
        model = Qwen3MoeModel(config)

        # Setup test inputs/outputs
        input_ids = torch.randint(0, 151936, (3, 5))
        positions = torch.arange(5).unsqueeze(0).repeat(3, 1)

        embed_output = torch.randn(3, 5, 4096)
        layer_output = torch.randn(3, 5, 4096)
        layer_residual = torch.randn(3, 5, 4096)
        final_output = torch.randn(3, 5, 4096)

        # Mock forward logic
        mock_embed_tokens.return_value = embed_output
        mock_layer_instance.side_effect = lambda pos, hs, res: (layer_output, layer_residual)
        mock_final_norm.side_effect = lambda hs, res: (final_output, None)

        # Run forward pass
        output = model(input_ids, positions)

        # Verify calls
        mock_embed_tokens.assert_called_once_with(input_ids)
        self.assertEqual(mock_layer_instance.call_count, 32)
        mock_final_norm.assert_called_once_with(layer_output, layer_residual)

        # Verify output shape
        self.assertEqual(output.shape, (3, 5, 4096))


class TestQwen3MoeForCausalLM(unittest.TestCase):

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3MoeModel")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.ParallelLMHead")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_initialization(self, mock_get_parallel_info, mock_lm_head, mock_model):
        """Verify Qwen3MoeForCausalLM initializes correctly."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_base_model = MagicMock()
        mock_lm_head_instance = MagicMock()
        mock_model.return_value = mock_base_model
        mock_lm_head.return_value = mock_lm_head_instance

        # Mock config
        hf_config = MagicMock(
            vocab_size=151936, hidden_size=4096, tie_word_embeddings=True,
            num_experts=8, num_experts_per_tok=2
        )
        mindie_llm_config = MagicMock(
            hf_config=hf_config,
            quant_config=MagicMock(spec=QuantizationConfigBase)
        )

        # Init instance
        model = Qwen3MoeForCausalLM(mindie_llm_config)

        # Assert attrs
        self.assertEqual(model.hf_config, hf_config)
        self.assertIsInstance(model.model, MagicMock)
        self.assertIsInstance(model.lm_head, MagicMock)

        # Verify sub-module init
        mock_model.assert_called_once_with(
            config=hf_config, prefix="model", quant_config=mindie_llm_config.quant_config
        )
        mock_lm_head.assert_called_once_with(
            151936, 4096, bias=False, quant_config=mindie_llm_config.quant_config, prefix="lm_head"
        )

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3MoeModel")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.ParallelLMHead")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_forward_pass(self, mock_get_parallel_info, mock_lm_head, mock_model):
        """Verify forward pass of Qwen3MoeForCausalLM (without LM head)."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_base_model = MagicMock()
        mock_lm_head_instance = MagicMock()
        mock_model.return_value = mock_base_model
        mock_lm_head.return_value = mock_lm_head_instance

        # Mock config
        hf_config = MagicMock(vocab_size=151936, hidden_size=4096)
        mindie_llm_config = MagicMock(hf_config=hf_config, quant_config=None)

        # Init instance
        model = Qwen3MoeForCausalLM(mindie_llm_config)

        # Setup test inputs/outputs
        input_ids = torch.randint(0, 151936, (3, 5))
        positions = torch.arange(5).unsqueeze(0).repeat(3, 1)
        model_output = torch.randn(3, 5, 4096)

        mock_base_model.return_value = model_output

        # Run forward pass
        output = model(input_ids, positions)

        # Verify calls
        mock_base_model.assert_called_once_with(input_ids, positions)

        # Verify output shape
        self.assertEqual(output.shape, (3, 5, 4096))

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_forward_context")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3MoeModel")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.ParallelLMHead")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_compute_logits(self, mock_get_parallel_info, mock_lm_head, mock_model, mock_get_forward_context):
        """Verify compute_logits pass through LM head for Qwen3MoeForCausalLM."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_base_model = MagicMock()
        mock_lm_head_instance = MagicMock()
        mock_model.return_value = mock_base_model
        mock_lm_head.return_value = mock_lm_head_instance

        # Mock forward context
        mock_forward_context = MagicMock()
        mock_forward_context.lm_head_indices = None
        mock_get_forward_context.return_value = mock_forward_context

        # Mock config
        hf_config = MagicMock(vocab_size=151936, hidden_size=4096)
        mindie_llm_config = MagicMock(hf_config=hf_config, quant_config=None)

        # Init instance
        model = Qwen3MoeForCausalLM(mindie_llm_config)

        # Setup test inputs/outputs
        hidden_states = torch.randn(3, 5, 4096)
        lm_head_output = torch.randn(3, 5, 151936)
        mock_lm_head_instance.forward.return_value = lm_head_output

        # Run compute logits
        output = model.compute_logits(hidden_states)

        # Verify calls
        mock_lm_head_instance.forward.assert_called_once_with(hidden_states, None)

        # Verify output shape
        self.assertEqual(output.shape, (3, 5, 151936))

    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_forward_context")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.Qwen3MoeModel")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.ParallelLMHead")
    @patch("mindie_llm.runtime.models.qwen3_moe.qwen3_moe.get_parallel_info_manager")
    def test_compute_logits_with_indices(self, mock_get_parallel_info, mock_lm_head, mock_model, mock_get_forward_context):
        """Verify compute_logits with lm_head_indices for Qwen3MoeForCausalLM."""
        # Setup mocks
        mock_parallel_info = MagicMock()
        mock_parallel_info.get.return_value = None
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_base_model = MagicMock()
        mock_lm_head_instance = MagicMock()
        mock_model.return_value = mock_base_model
        mock_lm_head.return_value = mock_lm_head_instance

        # Mock forward context with indices
        mock_forward_context = MagicMock()
        mock_forward_context.lm_head_indices = torch.tensor([0])
        mock_get_forward_context.return_value = mock_forward_context

        # Mock config
        hf_config = MagicMock(vocab_size=151936, hidden_size=4096)
        mindie_llm_config = MagicMock(hf_config=hf_config, quant_config=None)

        # Init instance
        model = Qwen3MoeForCausalLM(mindie_llm_config)

        # Setup test inputs/outputs
        hidden_states = torch.randn(3, 5, 4096)
        lm_head_output = torch.randn(3, 5, 151936)
        mock_lm_head_instance.forward.return_value = lm_head_output

        # Run compute logits
        output = model.compute_logits(hidden_states)

        # Verify calls with indices
        mock_lm_head_instance.forward.assert_called_once_with(hidden_states, torch.tensor([0]))

        # Verify output shape
        self.assertEqual(output.shape, (3, 5, 151936))


if __name__ == "__main__":
    unittest.main()