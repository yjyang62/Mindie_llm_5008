#!/usr/bin/env python
# coding=utf-8
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

from mindie_llm.runtime.layers.normalization import RMSNorm, LayerNorm
from mindie_llm.runtime.layers.quantization.unquantized import (
    UnquantizedNormMethod,
    UnquantizedLayerNormBiasMethod,
)


class TestRMSNorm(unittest.TestCase):
    """Test cases for RMSNorm with UnquantizedNormMethod."""

    def test_init_without_quant_config(self):
        """Test initialization without quantization config."""
        hidden_size = 512

        layer = RMSNorm(hidden_size=hidden_size)

        self.assertEqual(layer.hidden_size, hidden_size)
        self.assertEqual(layer.variance_epsilon, 1e-6)
        self.assertIsInstance(layer.quant_method, UnquantizedNormMethod)
        self.assertIsNotNone(layer.weight)

    def test_init_with_custom_eps(self):
        """Test initialization with custom epsilon."""
        hidden_size = 512
        eps = 1e-5

        layer = RMSNorm(hidden_size=hidden_size, eps=eps)

        self.assertEqual(layer.variance_epsilon, eps)

    def test_init_with_custom_dtype(self):
        """Test initialization with custom dtype."""
        hidden_size = 512
        weight_dtype = torch.float16

        layer = RMSNorm(hidden_size=hidden_size, weight_dtype=weight_dtype)

        self.assertEqual(layer.weight_dtype, weight_dtype)
        self.assertEqual(layer.weight.data.dtype, weight_dtype)

    def test_init_with_prefix(self):
        """Test initialization with prefix."""
        layer = RMSNorm(hidden_size=512, prefix="layers.0.norm")

        self.assertEqual(layer.prefix, "layers.0.norm")

    def test_init_with_var_hidden_size_warning(self):
        """Test initialization with var_hidden_size triggers warning."""
        hidden_size = 512

        with patch('mindie_llm.runtime.layers.normalization.logger') as mock_logger:
            layer = RMSNorm(hidden_size=hidden_size, var_hidden_size=256)

            mock_logger.warning.assert_called_once()
            self.assertEqual(layer.hidden_size, hidden_size)

    def test_weight_shape(self):
        """Test that weight shape is correct."""
        hidden_size = 512

        layer = RMSNorm(hidden_size=hidden_size)

        # Weight should be 1D tensor with shape (hidden_size,)
        expected_shape = (hidden_size,)
        self.assertEqual(layer.weight.data.shape, expected_shape)

    def test_weight_loader(self):
        """Test weight loading."""
        layer = RMSNorm(hidden_size=512)

        loaded_weight = torch.randn(512)
        param = layer.weight

        # Mock the load_weight method
        with patch.object(param, 'load_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight)

    @patch('torch_npu.npu_rms_norm')
    def test_forward_without_residual(self, mock_npu_rms_norm):
        """Test forward pass without residual."""
        layer = RMSNorm(hidden_size=512)

        # Initialize weight
        layer.weight.data = torch.ones(512)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock npu_rms_norm to return expected output
        mock_npu_rms_norm.return_value = (torch.randn(2, 3, 512), None)

        output = layer.forward(x)

        # Verify npu_rms_norm was called
        mock_npu_rms_norm.assert_called_once()
        call_args = mock_npu_rms_norm.call_args
        self.assertEqual(len(call_args[0]), 3)
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        self.assertEqual(call_args[0][2], layer.variance_epsilon)
        # Verify output shape
        self.assertEqual(output.shape, (2, 3, 512))

    @patch('torch_npu.npu_add_rms_norm')
    def test_forward_with_residual(self, mock_npu_add_rms_norm):
        """Test forward pass with residual."""
        layer = RMSNorm(hidden_size=512)

        # Initialize weight
        layer.weight.data = torch.ones(512)

        # Create input and residual
        x = torch.randn(2, 3, 512)
        residual = torch.randn(2, 3, 512)

        # Mock npu_add_rms_norm to return expected output
        mock_npu_add_rms_norm.return_value = (
            torch.randn(2, 3, 512),
            None,
            torch.randn(2, 3, 512)
        )

        output = layer.forward(x, residual)

        # Verify npu_add_rms_norm was called
        mock_npu_add_rms_norm.assert_called_once()
        call_args = mock_npu_add_rms_norm.call_args
        self.assertEqual(len(call_args[0]), 4)
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], residual))
        self.assertTrue(torch.allclose(call_args[0][2], layer.weight.data))
        self.assertEqual(call_args[0][3], layer.variance_epsilon)
        # Verify output is a tuple
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, (2, 3, 512))
        self.assertEqual(output[1].shape, (2, 3, 512))

    def test_extra_repr(self):
        """Test extra_repr method."""
        layer = RMSNorm(
            hidden_size=512,
            eps=1e-5,
            weight_dtype=torch.float32,
        )

        repr_str = layer.extra_repr()
        self.assertIn("hidden_size=512", repr_str)
        self.assertIn("eps=1e-05", repr_str)
        self.assertIn("UnquantizedNormMethod", repr_str)
        self.assertIn("dtype=torch.float32", repr_str)

    def test_unquantized_norm_method_apply(self):
        """Test that UnquantizedNormMethod.apply is called correctly."""
        layer = RMSNorm(hidden_size=512)

        # Initialize weight
        layer.weight.data = torch.ones(512)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock the quant_method.apply
        with patch.object(layer.quant_method, 'apply') as mock_apply:
            mock_apply.return_value = torch.randn(2, 3, 512)
            output = layer.forward(x)

            mock_apply.assert_called_once_with(layer, x, None)
            self.assertEqual(output.shape, (2, 3, 512))

    def test_unquantized_norm_method_apply_with_residual(self):
        """Test that UnquantizedNormMethod.apply is called correctly with residual."""
        layer = RMSNorm(hidden_size=512)

        # Initialize weight
        layer.weight.data = torch.ones(512)

        # Create input and residual
        x = torch.randn(2, 3, 512)
        residual = torch.randn(2, 3, 512)

        # Mock the quant_method.apply
        with patch.object(layer.quant_method, 'apply') as mock_apply:
            mock_apply.return_value = (torch.randn(2, 3, 512), torch.randn(2, 3, 512))
            output = layer.forward(x, residual)

            mock_apply.assert_called_once_with(layer, x, residual)
            self.assertIsInstance(output, tuple)
            self.assertEqual(len(output), 2)


class TestLayerNorm(unittest.TestCase):
    """Test cases for LayerNorm with UnquantizedLayerNormBiasMethod."""

    def test_init_without_quant_config(self):
        """Test initialization without quantization config."""
        hidden_size = 512

        layer = LayerNorm(hidden_size=hidden_size)

        self.assertEqual(layer.hidden_size, hidden_size)
        self.assertEqual(layer.variance_epsilon, 1e-6)
        self.assertIsInstance(layer.quant_method, UnquantizedLayerNormBiasMethod)
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)

    def test_init_with_custom_eps(self):
        """Test initialization with custom epsilon."""
        hidden_size = 512
        eps = 1e-5

        layer = LayerNorm(hidden_size=hidden_size, eps=eps)

        self.assertEqual(layer.variance_epsilon, eps)

    def test_init_with_custom_dtype(self):
        """Test initialization with custom dtype."""
        hidden_size = 512
        weight_dtype = torch.float16

        layer = LayerNorm(hidden_size=hidden_size, weight_dtype=weight_dtype)

        self.assertEqual(layer.weight_dtype, weight_dtype)
        self.assertEqual(layer.weight.data.dtype, weight_dtype)
        self.assertEqual(layer.bias.data.dtype, weight_dtype)

    def test_init_with_prefix(self):
        """Test initialization with prefix."""
        layer = LayerNorm(hidden_size=512, prefix="layers.0.norm")

        self.assertEqual(layer.prefix, "layers.0.norm")

    def test_init_with_var_hidden_size_warning(self):
        """Test initialization with var_hidden_size triggers warning."""
        hidden_size = 512

        with patch('mindie_llm.runtime.layers.normalization.logger') as mock_logger:
            layer = LayerNorm(hidden_size=hidden_size, var_hidden_size=256)

            mock_logger.warning.assert_called_once()
            self.assertEqual(layer.hidden_size, hidden_size)

    def test_weight_and_bias_shape(self):
        """Test that weight and bias shapes are correct."""
        hidden_size = 512

        layer = LayerNorm(hidden_size=hidden_size)

        # Weight and bias should be 1D tensors with shape (hidden_size,)
        expected_shape = (hidden_size,)
        self.assertEqual(layer.weight.data.shape, expected_shape)
        self.assertEqual(layer.bias.data.shape, expected_shape)

    def test_weight_loader(self):
        """Test weight loading."""
        layer = LayerNorm(hidden_size=512)

        loaded_weight = torch.randn(512)
        param = layer.weight

        # Mock the load_weight method
        with patch.object(param, 'load_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight)

    @patch('torch.nn.functional.layer_norm')
    def test_forward(self, mock_layer_norm):
        """Test forward pass."""
        layer = LayerNorm(hidden_size=512)

        # Initialize weight and bias
        layer.weight.data = torch.ones(512)
        layer.bias.data = torch.zeros(512)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock layer_norm to return expected output
        mock_layer_norm.return_value = torch.randn(2, 3, 512)

        output = layer.forward(x)

        # Verify layer_norm was called
        mock_layer_norm.assert_called_once()
        call_args = mock_layer_norm.call_args
        self.assertEqual(len(call_args[0]), 5)
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertEqual(call_args[0][1], (512,))
        self.assertTrue(torch.allclose(call_args[0][2], layer.weight.data))
        self.assertTrue(torch.allclose(call_args[0][3], layer.bias.data))
        self.assertEqual(call_args[0][4], layer.variance_epsilon)
        # Verify output shape
        self.assertEqual(output.shape, (2, 3, 512))

    def test_unquantized_layer_norm_bias_method_apply(self):
        """Test that UnquantizedLayerNormBiasMethod.apply is called correctly."""
        layer = LayerNorm(hidden_size=512)

        # Initialize weight and bias
        layer.weight.data = torch.ones(512)
        layer.bias.data = torch.zeros(512)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock the quant_method.apply
        with patch.object(layer.quant_method, 'apply') as mock_apply:
            mock_apply.return_value = torch.randn(2, 3, 512)
            output = layer.forward(x)

            mock_apply.assert_called_once_with(layer, x, layer.hidden_size)
            self.assertEqual(output.shape, (2, 3, 512))


class TestUnquantizedNormMethod(unittest.TestCase):
    """Test cases for UnquantizedNormMethod directly."""

    def test_create_weights(self):
        """Test UnquantizedNormMethod.create_weights."""
        import torch.nn as nn
        
        method = UnquantizedNormMethod()
        # Use a real nn.Module to test register_parameter properly
        layer = nn.Module()
        hidden_size = 512
        params_dtype = torch.float32

        method.create_weights(layer, hidden_size, params_dtype)

        # Verify weight was registered and has correct properties
        self.assertTrue(hasattr(layer, 'weight'))
        self.assertEqual(layer.weight.data.shape, (hidden_size,))
        self.assertEqual(layer.weight.data.dtype, params_dtype)
        # Verify weight is initialized to ones
        self.assertTrue(torch.allclose(layer.weight.data, torch.ones(hidden_size, dtype=params_dtype)))

    @patch('torch_npu.npu_rms_norm')
    def test_apply_without_residual(self, mock_npu_rms_norm):
        """Test UnquantizedNormMethod.apply without residual."""
        method = UnquantizedNormMethod()
        layer = MagicMock()
        layer.weight.data = torch.ones(512)
        layer.variance_epsilon = 1e-6

        x = torch.randn(2, 3, 512)

        # Mock npu_rms_norm to return expected output
        mock_npu_rms_norm.return_value = (torch.randn(2, 3, 512), None)

        output = method.apply(layer, x, None)

        # Verify npu_rms_norm was called with correct arguments
        mock_npu_rms_norm.assert_called_once()
        call_args = mock_npu_rms_norm.call_args
        self.assertEqual(len(call_args[0]), 3)
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        self.assertEqual(call_args[0][2], layer.variance_epsilon)
        # Verify output shape
        self.assertEqual(output.shape, (2, 3, 512))

    @patch('torch_npu.npu_add_rms_norm')
    def test_apply_with_residual(self, mock_npu_add_rms_norm):
        """Test UnquantizedNormMethod.apply with residual."""
        method = UnquantizedNormMethod()
        layer = MagicMock()
        layer.weight.data = torch.ones(512)
        layer.variance_epsilon = 1e-6

        x = torch.randn(2, 3, 512)
        residual = torch.randn(2, 3, 512)

        # Mock npu_add_rms_norm to return expected output
        mock_npu_add_rms_norm.return_value = (
            torch.randn(2, 3, 512),
            None,
            torch.randn(2, 3, 512)
        )

        output = method.apply(layer, x, residual)

        # Verify npu_add_rms_norm was called with correct arguments
        mock_npu_add_rms_norm.assert_called_once()
        call_args = mock_npu_add_rms_norm.call_args
        self.assertEqual(len(call_args[0]), 4)
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], residual))
        self.assertTrue(torch.allclose(call_args[0][2], layer.weight.data))
        self.assertEqual(call_args[0][3], layer.variance_epsilon)
        # Verify output is a tuple
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, (2, 3, 512))
        self.assertEqual(output[1].shape, (2, 3, 512))


class TestUnquantizedLayerNormBiasMethod(unittest.TestCase):
    """Test cases for UnquantizedLayerNormBiasMethod directly."""

    def test_create_weights(self):
        """Test UnquantizedLayerNormBiasMethod.create_weights."""
        import torch.nn as nn
        
        method = UnquantizedLayerNormBiasMethod()
        # Use a real nn.Module to test register_parameter properly
        layer = nn.Module()
        hidden_size = 512
        params_dtype = torch.float32

        method.create_weights(layer, hidden_size, params_dtype)

        # Verify weight and bias were registered and have correct properties
        self.assertTrue(hasattr(layer, 'weight'))
        self.assertTrue(hasattr(layer, 'bias'))
        
        # Check weight
        self.assertEqual(layer.weight.data.shape, (hidden_size,))
        self.assertEqual(layer.weight.data.dtype, params_dtype)
        self.assertTrue(torch.allclose(layer.weight.data, torch.ones(hidden_size, dtype=params_dtype)))

        # Check bias
        self.assertEqual(layer.bias.data.shape, (hidden_size,))
        self.assertEqual(layer.bias.data.dtype, params_dtype)
        self.assertTrue(torch.allclose(layer.bias.data, torch.zeros(hidden_size, dtype=params_dtype)))

    @patch('torch.nn.functional.layer_norm')
    def test_apply(self, mock_layer_norm):
        """Test UnquantizedLayerNormBiasMethod.apply."""
        method = UnquantizedLayerNormBiasMethod()
        layer = MagicMock()
        layer.weight.data = torch.ones(512)
        layer.bias.data = torch.zeros(512)
        layer.variance_epsilon = 1e-6

        x = torch.randn(2, 3, 512)
        dim = 512

        # Mock layer_norm to return expected output
        mock_layer_norm.return_value = torch.randn(2, 3, 512)

        output = method.apply(layer, x, dim)

        # Verify layer_norm was called with correct arguments
        mock_layer_norm.assert_called_once()
        call_args = mock_layer_norm.call_args
        self.assertEqual(len(call_args[0]), 5)
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertEqual(call_args[0][1], (dim,))
        self.assertTrue(torch.allclose(call_args[0][2], layer.weight.data))
        self.assertTrue(torch.allclose(call_args[0][3], layer.bias.data))
        self.assertEqual(call_args[0][4], layer.variance_epsilon)
        # Verify output shape
        self.assertEqual(output.shape, (2, 3, 512))


if __name__ == '__main__':
    unittest.main()
