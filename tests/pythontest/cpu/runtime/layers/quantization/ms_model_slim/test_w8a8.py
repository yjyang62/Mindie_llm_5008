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
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch_npu

from mindie_llm.runtime.layers.linear.linear import ReplicatedLinear
from mindie_llm.runtime.layers.parameter import ModelWeightParameter
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.layers.quantization.ms_model_slim.w8a8 import (
    W8A8PerTensorLinearMethod,
    W8A8PerTokenLinearMethod,
    W8A8MixLinearMethod,
    W8A8PerTokenFusedMoEMethod,
    W8A8MXFP8PerGroupLinearMethod,
)
from mindie_llm.runtime.utils.distributed import set_parallel_info_manager


class TestW8A8PerTensorLinearMethod(unittest.TestCase):
    """Test cases for W8A8PerTensorLinearMethod."""

    def setUp(self):
        """Set up test fixtures."""
        # W8A8 for pertensor
        self.quant_config_w8a8 = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8,
            "layer.weight": QuantType.W8A8,
            "embed_tokens.weight": QuantType.W8A8,
            "lm_head.weight": QuantType.W8A8,
        })
        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_create_weights(self):
        """Test create_weights method."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Verify quantized parameters were created
        self.assertIsInstance(layer.quant_method, W8A8PerTensorLinearMethod)
        self.assertIsNotNone(layer.weight)
        self.assertEqual(layer.weight.data.dtype, torch.int8)
        self.assertIsNotNone(layer.input_scale)
        self.assertIsNotNone(layer.input_offset)
        self.assertIsNotNone(layer.deq_scale)
        self.assertEqual(layer.deq_scale.data.dtype, torch.int64)
        self.assertIsNotNone(layer.quant_bias)


    def test_create_weights_bfloat16(self):
        """Test create_weights with bfloat16."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8,
            weight_dtype=torch.bfloat16,
            prefix="layer"
        )

        # deq_scale should be float32 for bfloat16
        self.assertEqual(layer.deq_scale.data.dtype, torch.float32)


    def test_create_weights_unsupported_dtype(self):
        """Test create_weights with unsupported dtype."""
        with self.assertRaises(ValueError) as context:
            # Should raise error during apply, not during create_weights
            layer = ReplicatedLinear(
                input_size=512,
                output_size=1024,
                quant_config=self.quant_config_w8a8,
                weight_dtype=torch.float32,
                prefix="layer"
            )

            # Create weights should succeed, but apply will fail
            self.assertIsInstance(layer.quant_method, W8A8PerTensorLinearMethod)

            self.assertIn("is not supported in `W8A8PerTensorLinearMethod`", str(context.exception))

    @patch('torch_npu.npu_quant_matmul')
    @patch('torch_npu.npu_quantize')
    def test_apply(self, mock_npu_quantize, mock_npu_quant_matmul):
        """Test apply method."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Initialize parameters
        layer.input_scale.data = torch.tensor([1.0], dtype=torch.float16)
        layer.input_offset.data = torch.tensor([0], dtype=torch.int8)
        layer.deq_scale.data = torch.randn(1024).to(torch.int64)
        layer.quant_bias.data = torch.randn(1024).to(torch.int32)

        x = torch.randn(2, 3, 512, dtype=torch.float16)

        # Mock npu functions
        mock_quantized_tensor = torch.randn(2, 3, 512)
        mock_npu_quantize.return_value = mock_quantized_tensor
        mock_npu_quant_matmul.return_value = torch.randn(2, 3, 1024, dtype=torch.float16)

        output = layer.quant_method.apply(layer, x)

        # Verify npu_quantize was called
        mock_npu_quantize.assert_called_once()
        call_args = mock_npu_quantize.call_args
        self.assertIs(call_args.kwargs['input'], x)
        self.assertTrue(torch.allclose(call_args.kwargs['scales'], layer.input_scale.data))
        self.assertTrue(torch.allclose(call_args.kwargs['zero_points'], layer.input_offset.data))
        self.assertEqual(call_args.kwargs['dtype'], torch.qint8)

        # Verify npu_quant_matmul was called
        mock_npu_quant_matmul.assert_called_once()
        matmul_call_args = mock_npu_quant_matmul.call_args
        self.assertIs(matmul_call_args[0][0], mock_quantized_tensor)
        self.assertTrue(torch.equal(matmul_call_args[0][1], layer.weight.data))
        self.assertTrue(torch.allclose(matmul_call_args[0][2], layer.deq_scale.data))
        torch.testing.assert_close(matmul_call_args[1]['bias'], layer.quant_bias.data)
        self.assertEqual(matmul_call_args[1]['output_dtype'], torch.float16)

        self.assertEqual(output.shape, (2, 3, 1024))

    @patch('torch_npu.npu_format_cast')
    @patch('torch_npu.npu.get_soc_version')
    @patch('torch_npu.get_npu_format')
    def test_process_weights_after_loading(self, mock_get_npu_format, mock_get_soc_version, mock_format_cast):
        """Test process_weights_after_loading method."""
        # layer.weight.data shape will be (1024, 512)
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Initialize parameters
        layer.input_scale.data = torch.tensor([1.0], dtype=torch.float16)
        layer.input_offset.data = torch.tensor([0], dtype=torch.int8)

        # Mock soc version (A2/A3)
        mock_get_soc_version.return_value = 223
        mock_get_npu_format.return_value = 0
        mock_format_cast.return_value = layer.weight.data

        layer.quant_method.process_weights_after_loading(layer)

        # Verify input_scale was expanded
        self.assertEqual(layer.input_scale.data.shape, (512,))



class TestW8A8PerTokenLinearMethod(unittest.TestCase):
    """Test cases for W8A8PerTokenLinearMethod."""

    def setUp(self):
        """Set up test fixtures."""
        # W8A8_DYNAMIC for pertoken
        self.quant_config_w8a8_dynamic = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8_DYNAMIC,
            "layer.weight": QuantType.W8A8_DYNAMIC,
            "layer.bias": QuantType.W8A8_DYNAMIC,
            "embed_tokens.weight": QuantType.W8A8_DYNAMIC,
            "lm_head.weight": QuantType.W8A8_DYNAMIC,
        })

        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_create_weights(self):
        """Test create_weights method."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_dynamic,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Verify quantized parameters were created
        self.assertIsInstance(layer.quant_method, W8A8PerTokenLinearMethod)
        self.assertIsNotNone(layer.weight)
        self.assertEqual(layer.weight.data.dtype, torch.int8)
        self.assertIsNotNone(layer.weight_scale)
        self.assertIsNotNone(layer.weight_offset)

    def test_create_weights_with_bias(self):
        """Test create_weights with bias in quant_config."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_dynamic,
            weight_dtype=torch.float16,
            bias=True,
            prefix="layer"
        )

        # Should have bias parameter
        self.assertIsNotNone(layer.bias)

    @patch('torch_npu.npu_quant_matmul')
    @patch('torch_npu.npu_dynamic_quant')
    def test_apply(self, mock_npu_dynamic_quant, mock_npu_quant_matmul):
        """Test apply method."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_dynamic,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Initialize parameters
        layer.weight_scale.data = torch.randn(1024, 1, dtype=torch.float32)
        layer.weight_offset.data = torch.randn(1024, 1, dtype=torch.float16)

        x = torch.randn(2, 3, 512, dtype=torch.float16)

        # Mock npu functions
        mock_quantized_tensor = torch.randn(2, 3, 512)
        mock_pertoken_scale = torch.randn(2, 3, 1, dtype=torch.float16)
        mock_npu_dynamic_quant.return_value = (mock_quantized_tensor, mock_pertoken_scale)
        mock_npu_quant_matmul.return_value = torch.randn(2, 3, 1024, dtype=torch.float16)

        output = layer.quant_method.apply(layer, x)

        # Verify npu_dynamic_quant was called
        mock_npu_dynamic_quant.assert_called_once_with(x)

        # Verify npu_quant_matmul was called
        mock_npu_quant_matmul.assert_called_once()
        matmul_call_args = mock_npu_quant_matmul.call_args
        self.assertTrue(torch.allclose(matmul_call_args[0][0], mock_quantized_tensor))
        self.assertTrue(torch.equal(matmul_call_args[0][1], layer.weight.data))
        self.assertTrue(torch.allclose(matmul_call_args[0][2], layer.weight_scale.data))
        self.assertTrue(torch.allclose(matmul_call_args[1]['pertoken_scale'], mock_pertoken_scale))
        self.assertEqual(matmul_call_args[1]['bias'], None)
        self.assertEqual(matmul_call_args[1]['output_dtype'], torch.float16)

        self.assertEqual(output.shape, (2, 3, 1024))

    @patch('torch_npu.npu_quant_matmul')
    @patch('torch_npu.npu_dynamic_quant')
    def test_apply_with_bias(self, mock_npu_dynamic_quant, mock_npu_quant_matmul):
        """Test apply method with bias."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_dynamic,
            weight_dtype=torch.float16,
            bias=True,
            prefix="layer"
        )

        # Initialize parameters
        layer.weight_scale.data = torch.randn(1024, 1, dtype=torch.float32)
        layer.weight_offset.data = torch.randn(1024, 1, dtype=torch.float16)
        layer.bias.data = torch.randn(1024, dtype=torch.float16)

        x = torch.randn(2, 3, 512)

        # Mock npu functions
        mock_quantized_tensor = torch.randn(2, 3, 512)
        mock_pertoken_scale = torch.randn(2, 3, 1, dtype=torch.float16)
        mock_npu_dynamic_quant.return_value = (mock_quantized_tensor, mock_pertoken_scale)
        mock_npu_quant_matmul.return_value = torch.randn(2, 3, 1024, dtype=torch.float16)

        output = layer.quant_method.apply(layer, x)

        # Verify bias was added
        self.assertEqual(output.shape, (2, 3, 1024))

    @patch('torch_npu.npu_format_cast')
    @patch('torch_npu.npu.get_soc_version')
    @patch('torch_npu.get_npu_format')
    def test_process_weights_after_loading(self, mock_get_npu_format, mock_get_soc_version, mock_format_cast):
        """Test process_weights_after_loading method."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_dynamic,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Initialize parameters
        layer.weight.data = torch.randn(1024, 512).to(torch.int8)
        layer.weight_scale.data = torch.randn(1024, 1, dtype=torch.float32)

        # Mock soc version (A2/A3)
        mock_get_soc_version.return_value = 223
        mock_format_cast.return_value = layer.weight.data

        layer.quant_method.process_weights_after_loading(layer)

        # Verify weight_scale was flattened
        self.assertEqual(layer.weight_scale.data.shape, (1024,))


class TestW8A8MixLinearMethod(unittest.TestCase):
    """Test cases for W8A8MixLinearMethod."""

    def setUp(self):
        """Set up test fixtures."""
        # W8A8_DYNAMIC for pertoken
        self.quant_config_w8a8_mix = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8_MIX,
            "layer.weight": QuantType.W8A8_MIX,
            "layer.bias": QuantType.W8A8_MIX,
            "embed_tokens.weight": QuantType.W8A8_MIX,
            "lm_head.weight": QuantType.W8A8_MIX,
        })

        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_create_weights(self):
        """Test create_weights method creates weights for both modes."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_mix,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Verify quantized parameters were created (both per-tensor and per-token)
        self.assertIsInstance(layer.quant_method, W8A8MixLinearMethod)
        self.assertIsNotNone(layer.weight)
        # Should have both per-tensor and per-token parameters
        self.assertIsNotNone(layer.input_scale)  # Per-tensor
        self.assertIsNotNone(layer.input_offset)  # Per-tensor
        self.assertIsNotNone(layer.deq_scale)  # Per-tensor
        self.assertIsNotNone(layer.quant_bias)  # Per-tensor
        self.assertIsNotNone(layer.weight_scale)  # Per-token
        self.assertIsNotNone(layer.weight_offset)  # Per-token

    @patch('torch_npu.npu_format_cast')
    @patch('torch_npu.npu.get_soc_version')
    @patch('torch_npu.get_npu_format')
    def test_process_weights_after_loading(self, mock_get_npu_format, mock_get_soc_version, mock_format_cast):
        """Test process_weights_after_loading method."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_mix,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Initialize parameters
        layer.input_scale.data = torch.tensor([1.0], dtype=torch.float16)
        layer.input_offset.data = torch.tensor([0], dtype=torch.int8)
        layer.weight.data = torch.randn(1024, 512).to(torch.int8)
        layer.weight_scale.data = torch.randn(1024, 1).to(torch.float32)

        # Mock soc version (A2/A3)
        mock_get_soc_version.return_value = 223
        mock_format_cast.return_value = layer.weight.data

        layer.quant_method.process_weights_after_loading(layer)

        # Verify input_scale was expanded
        self.assertEqual(layer.input_scale.data.shape, (512,))
        # Verify weight_scale was flattened
        self.assertEqual(layer.weight_scale.data.shape, (1024,))


class TestW8A8MXFP8PerGroupLinearMethod(unittest.TestCase):
    """Test cases for W8A8MXFP8PerGroupLinearMethod."""

    def setUp(self):
        """Set up test fixtures."""
        # W8A8_MXFP8 for pertoken
        self.quant_config_w8a8_mxfp8 = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8_MXFP8,
            "layer.weight": QuantType.W8A8_MXFP8,
            "layer.bias": None,
            "embed_tokens.weight": QuantType.W8A8_MXFP8,
            "lm_head.weight": QuantType.W8A8_MXFP8,
        })

        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_create_weights(self):
        """Test create_weights method."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_mxfp8,
            weight_dtype=torch.float16,
            prefix="layer"
        )

        # Verify quantized parameters were created
        self.assertIsInstance(layer.quant_method, W8A8MXFP8PerGroupLinearMethod)
        self.assertIsNotNone(layer.weight)
        self.assertEqual(layer.weight.data.dtype, torch.float8_e4m3fn)
        self.assertIsNotNone(layer.weight_scale)

    @patch('torch.float8_e4m3fn', torch.float16)
    @patch('torch_npu.float8_e8m0fnu', torch.float16, create=True)
    @patch('torch_npu.npu_quant_matmul')
    @patch('torch_npu.npu_dynamic_mx_quant', create=True)
    def test_apply(self, mock_npu_dynamic_quant, mock_npu_quant_matmul):
        """Test apply method."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            quant_config=self.quant_config_w8a8_mxfp8,
            prefix="layer"
        )

        # Initialize parameters
        layer.weight.data = torch.randn(1024, 512).to(torch.float8_e4m3fn)
        layer.weight_scale.data = torch.randn(1024, 1).to(torch.uint8)

        x = torch.randn(2, 3, 512, dtype=torch.float16)

        # Mock npu functions
        mock_quantized_tensor = torch.randn(2, 3, 512)
        mock_pertoken_scale = torch.randn(2, 3, 1)
        mock_npu_dynamic_quant.return_value = (mock_quantized_tensor, mock_pertoken_scale)
        mock_npu_quant_matmul.return_value = torch.randn(2, 3, 1024, dtype=torch.float16)

        output = layer.quant_method.apply(layer, x)

        # Verify npu_dynamic_quant was called
        mock_npu_dynamic_quant.assert_called_once_with(x, dst_type=torch.float8_e4m3fn)

        # Verify npu_quant_matmul was called
        mock_npu_quant_matmul.assert_called_once()
        matmul_call_args = mock_npu_quant_matmul.call_args
        self.assertTrue(torch.allclose(matmul_call_args[0][0], mock_quantized_tensor))
        self.assertTrue(torch.equal(matmul_call_args[0][1], layer.weight))
        self.assertTrue(torch.allclose(matmul_call_args[0][2], layer.weight_scale))
        self.assertTrue(torch.allclose(matmul_call_args[1]['pertoken_scale'], mock_pertoken_scale))
        self.assertEqual(matmul_call_args[1]['bias'], None)
        self.assertEqual(matmul_call_args[1]['output_dtype'], torch.float16)

        self.assertEqual(output.shape, (2, 3, 1024))


class TestW8A8PerTokenFusedMoEMethod(unittest.TestCase):
    """Test cases for W8A8PerTokenFusedMoEMethod."""

    def setUp(self):
        self.layer = nn.Module()
        self.num_experts = 64
        self.hidden_size = 1024
        self.intermediate_size_per_partition = 128
        self.weight_dtype = torch.float16
        self.extra_attrs = {}
        self.method = W8A8PerTokenFusedMoEMethod()

    def test_create_weights(self):
        """Test create_weights method."""
        self.method.create_weights(
            layer=self.layer,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.weight_dtype,
            **self.extra_attrs
        )

        self.assertTrue(hasattr(self.layer, 'gate_up_weight'))
        self.assertEqual(self.layer.gate_up_weight.data.shape, (64, 256, 1024))
        self.assertEqual(self.layer.gate_up_weight.data.dtype, torch.int8)
        self.assertEqual(self.layer.gate_up_weight.input_dim, 1)
        self.assertEqual(self.layer.gate_up_weight.output_dim, 0)

        self.assertTrue(hasattr(self.layer, 'down_weight'))
        self.assertEqual(self.layer.down_weight.data.shape, (64, 1024, 128))
        self.assertEqual(self.layer.down_weight.data.dtype, torch.int8)
        self.assertEqual(self.layer.down_weight.input_dim, 1)
        self.assertEqual(self.layer.down_weight.output_dim, 0)

        self.assertTrue(hasattr(self.layer, 'gate_up_weight_scale'))
        self.assertEqual(self.layer.gate_up_weight_scale.data.shape, (64, 256, 1))
        self.assertEqual(self.layer.gate_up_weight_scale.data.dtype, torch.float32)
        self.assertEqual(self.layer.gate_up_weight_scale.output_dim, 0)

        self.assertTrue(hasattr(self.layer, 'down_weight_scale'))
        self.assertEqual(self.layer.down_weight_scale.data.shape, (64, 1024, 1))
        self.assertEqual(self.layer.down_weight_scale.data.dtype, torch.float32)
        self.assertEqual(self.layer.down_weight_scale.input_dim, 0)

    @patch("torch_npu.npu_dynamic_quant")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_dequant_swiglu_quant")
    def test_apply(self, mock_npu_dequant_swiglu_quant, mock_npu_grouped_matmul, mock_npu_dynamic_quant):
        """Test apply method."""

        self.layer.gate_up_weight = torch.randint(-100, 100, (64, 1024, 256)).to(torch.int8)
        self.layer.down_weight = torch.randint(-100, 100, (64, 128, 1024)).to(torch.int8)
        self.layer.gate_up_weight_scale = torch.randn(64, 256, dtype=torch.float32)
        self.layer.down_weight_scale = torch.randn(64, 1024, dtype=torch.float32)

        x = torch.randn(100, 1024, dtype=torch.float16)
        group_list = torch.randint(0, 100, (64,)).to(torch.int8)

        mock_quantized_tensor = torch.randint(-100, 100, (100, 1024)).to(torch.int8)
        mock_pertoken_scale = torch.randn(100, dtype=torch.float32)
        mock_npu_dynamic_quant.return_value = (mock_quantized_tensor, mock_pertoken_scale)

        mock_act_out = torch.randint(-100, 100, (100, 256)).to(torch.int8)
        mock_down_scale = torch.randn(100, dtype=torch.float32)
        mock_npu_dequant_swiglu_quant.return_value = (mock_act_out, mock_down_scale)

        mock_gate_up_out = torch.randint(-100, 100, (100, 256)).to(torch.int32)
        mock_down_out = torch.randn(100, 1024, dtype=torch.float16)
        mock_npu_grouped_matmul.side_effect = [
            [mock_gate_up_out],
            [mock_down_out]
        ]

        output = self.method.apply(self.layer, x, group_list)

        mock_npu_dynamic_quant.assert_called_once_with(x)
        self.assertEqual(mock_npu_grouped_matmul.call_count, 2)
        mock_npu_dequant_swiglu_quant.assert_called_once()

        first_call_args = mock_npu_grouped_matmul.call_args_list[0]
        self.assertTrue(torch.equal(first_call_args[1]["x"][0], mock_quantized_tensor))
        self.assertTrue(torch.equal(first_call_args[1]["weight"][0], self.layer.gate_up_weight))
        self.assertTrue(torch.equal(first_call_args[1]["group_list"], group_list))

        second_call_args = mock_npu_grouped_matmul.call_args_list[1]
        self.assertTrue(torch.equal(second_call_args[1]["x"][0], mock_act_out))
        self.assertTrue(torch.equal(second_call_args[1]["weight"][0], self.layer.down_weight))
        self.assertTrue(torch.equal(second_call_args[1]["group_list"], group_list))

        self.assertEqual(output.shape, (100, 1024))

    def test_process_weights_after_loading(self):
        """Test process_weights_after_loading method."""
        self.layer.gate_up_weight = ModelWeightParameter(torch.randn(64, 256, 1024))
        self.layer.down_weight = ModelWeightParameter(torch.randn(64, 1024, 128))
        self.layer.gate_up_weight_scale = ModelWeightParameter(torch.randn(64, 256, 1))
        self.layer.down_weight_scale = ModelWeightParameter(torch.randn(64, 1024, 1))
        self.method.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.gate_up_weight.data.shape, (64, 1024, 256))
        self.assertEqual(self.layer.down_weight.data.shape, (64, 128, 1024))
        self.assertEqual(self.layer.gate_up_weight_scale.data.shape, (64, 256))
        self.assertEqual(self.layer.gate_up_weight_scale.data.dtype, torch.float32)
        self.assertEqual(self.layer.down_weight_scale.data.shape, (64, 1024))


if __name__ == '__main__':
    unittest.main()
