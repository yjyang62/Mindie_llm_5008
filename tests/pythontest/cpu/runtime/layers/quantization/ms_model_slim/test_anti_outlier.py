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

from mindie_llm.runtime.layers.quantization.ms_model_slim.anti_outlier import AntiOutlierNormMethod
from mindie_llm.runtime.layers.parameter import BaseParameter


class TestAntiOutlierNormMethod(unittest.TestCase):
    """Test cases for AntiOutlierNormMethod."""

    def setUp(self):
        """Set up test fixtures."""
        self.quant_method = AntiOutlierNormMethod()
        self.hidden_size = 512

    def test_create_weights(self):
        """Test create_weights method."""
        # Create a mock layer
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.register_parameter = MagicMock()

        params_dtype = torch.float32
        extra_attrs = {"output_dim": 0}

        self.quant_method.create_weights(
            layer=mock_layer,
            hidden_size=self.hidden_size,
            params_dtype=params_dtype,
            **extra_attrs,
        )

        # Verify register_parameter was called twice (weight and bias)
        self.assertEqual(mock_layer.register_parameter.call_count, 2)

        # Get the registered parameters
        weight_call = mock_layer.register_parameter.call_args_list[0]
        bias_call = mock_layer.register_parameter.call_args_list[1]

        # Verify weight parameter
        self.assertEqual(weight_call[0][0], "weight")
        weight_param = weight_call[0][1]
        self.assertIsInstance(weight_param, BaseParameter)
        self.assertEqual(weight_param.data.shape, (self.hidden_size,))
        self.assertEqual(weight_param.data.dtype, params_dtype)
        # Verify weight is initialized to ones
        self.assertTrue(torch.allclose(weight_param.data, torch.ones(self.hidden_size, dtype=params_dtype)))

        # Verify bias parameter
        self.assertEqual(bias_call[0][0], "bias")
        bias_param = bias_call[0][1]
        self.assertIsInstance(bias_param, BaseParameter)
        self.assertEqual(bias_param.data.shape, (self.hidden_size,))
        self.assertEqual(bias_param.data.dtype, params_dtype)
        # Verify bias is initialized to zeros
        self.assertTrue(torch.allclose(bias_param.data, torch.zeros(self.hidden_size, dtype=params_dtype)))

    def test_create_weights_with_custom_dtype(self):
        """Test create_weights with custom dtype."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.register_parameter = MagicMock()

        params_dtype = torch.float16

        self.quant_method.create_weights(
            layer=mock_layer,
            hidden_size=self.hidden_size,
            params_dtype=params_dtype,
        )

        # Verify weight dtype
        weight_call = mock_layer.register_parameter.call_args_list[0]
        weight_param = weight_call[0][1]
        self.assertEqual(weight_param.data.dtype, params_dtype)

        # Verify bias dtype
        bias_call = mock_layer.register_parameter.call_args_list[1]
        bias_param = bias_call[0][1]
        self.assertEqual(bias_param.data.dtype, params_dtype)

    def test_create_weights_with_extra_attrs(self):
        """Test create_weights with extra attributes."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.register_parameter = MagicMock()

        extra_attrs = {"output_dim": 0, "input_dim": 1, "custom_attr": "test"}

        self.quant_method.create_weights(
            layer=mock_layer,
            hidden_size=self.hidden_size,
            params_dtype=torch.float32,
            **extra_attrs,
        )

        # Verify extra attributes were added to weight
        weight_call = mock_layer.register_parameter.call_args_list[0]
        weight_param = weight_call[0][1]
        self.assertEqual(getattr(weight_param,"output_dim"), 0)
        self.assertEqual(getattr(weight_param,"input_dim"), 1)
        self.assertEqual(getattr(weight_param,"custom_attr"), "test")

        # Verify extra attributes were added to bias
        bias_call = mock_layer.register_parameter.call_args_list[1]
        bias_param = bias_call[0][1]
        self.assertEqual(getattr(bias_param,"output_dim"), 0)
        self.assertEqual(getattr(bias_param,"input_dim"), 1)
        self.assertEqual(getattr(bias_param,"custom_attr"), "test")

    def test_create_weights_different_hidden_sizes(self):
        """Test create_weights with different hidden sizes."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.register_parameter = MagicMock()

        for hidden_size in [128, 256, 512, 1024, 2048]:
            mock_layer.register_parameter.reset_mock()

            self.quant_method.create_weights(
                layer=mock_layer,
                hidden_size=hidden_size,
                params_dtype=torch.float32,
            )

            # Verify weight shape
            weight_call = mock_layer.register_parameter.call_args_list[0]
            weight_param = weight_call[0][1]
            self.assertEqual(weight_param.data.shape, (hidden_size,))

            # Verify bias shape
            bias_call = mock_layer.register_parameter.call_args_list[1]
            bias_param = bias_call[0][1]
            self.assertEqual(bias_param.data.shape, (hidden_size,))

    @patch('torch_npu.npu_rms_norm')
    def test_apply_without_residual(self, mock_npu_rms_norm):
        """Test apply method without residual."""
        # Create a mock layer
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        mock_layer.bias.data = torch.zeros(self.hidden_size, dtype=torch.float32)
        mock_layer.variance_epsilon = 1e-6

        # Create input tensor
        x = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)

        # Mock npu_rms_norm to return normalized tensor and variance
        normalized_tensor = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        variance = torch.randn(2, 3, dtype=torch.float32)
        mock_npu_rms_norm.return_value = (normalized_tensor, variance)

        # Call apply
        output = self.quant_method.apply(layer=mock_layer, x=x)

        # Verify npu_rms_norm was called with correct arguments
        mock_npu_rms_norm.assert_called_once()
        call_args = mock_npu_rms_norm.call_args
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], mock_layer.weight.data))
        self.assertEqual(call_args[0][2], mock_layer.variance_epsilon)

        # Verify output shape (should be normalized + bias)
        self.assertEqual(output.shape, (2, 3, self.hidden_size))
        # Output should be normalized_tensor + bias
        expected_output = normalized_tensor + mock_layer.bias.data
        self.assertTrue(torch.allclose(output, expected_output))

    @patch('torch_npu.npu_rms_norm')
    def test_apply_without_residual_different_shapes(self, mock_npu_rms_norm):
        """Test apply method without residual with different input shapes."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        mock_layer.bias.data = torch.zeros(self.hidden_size, dtype=torch.float32)
        mock_layer.variance_epsilon = 1e-6

        test_shapes = [
            (self.hidden_size,),  # 1D
            (10, self.hidden_size),  # 2D
            (2, 3, self.hidden_size),  # 3D
            (1, 2, 3, self.hidden_size),  # 4D
        ]

        for shape in test_shapes:
            mock_npu_rms_norm.reset_mock()

            x = torch.randn(*shape, dtype=torch.float32)
            normalized_tensor = torch.randn(*shape, dtype=torch.float32)
            variance = torch.randn(shape[:-1], dtype=torch.float32)
            mock_npu_rms_norm.return_value = (normalized_tensor, variance)

            output = self.quant_method.apply(layer=mock_layer, x=x)

            # Verify output shape matches input shape
            self.assertEqual(output.shape, shape)
            mock_npu_rms_norm.assert_called_once()

    @patch('torch_npu.npu_add_rms_norm')
    def test_apply_with_residual(self, mock_npu_add_rms_norm):
        """Test apply method with residual."""
        # Create a mock layer
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        mock_layer.bias.data = torch.zeros(self.hidden_size, dtype=torch.float32)
        mock_layer.variance_epsilon = 1e-6

        # Create input and residual tensors
        x = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        residual = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)

        # Mock npu_add_rms_norm to return normalized tensor, variance, and updated residual
        normalized_tensor = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        variance = torch.randn(2, 3, dtype=torch.float32)
        updated_residual = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        mock_npu_add_rms_norm.return_value = (normalized_tensor, variance, updated_residual)

        # Call apply
        output, output_residual = self.quant_method.apply(layer=mock_layer, x=x, residual=residual)

        # Verify npu_add_rms_norm was called with correct arguments
        mock_npu_add_rms_norm.assert_called_once()
        call_args = mock_npu_add_rms_norm.call_args
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], residual))
        self.assertTrue(torch.allclose(call_args[0][2], mock_layer.weight.data))
        self.assertEqual(call_args[0][3], mock_layer.variance_epsilon)

        # Verify output shape (should be normalized + bias)
        self.assertEqual(output.shape, (2, 3, self.hidden_size))
        # Output should be normalized_tensor + bias
        expected_output = normalized_tensor + mock_layer.bias.data
        self.assertTrue(torch.allclose(output, expected_output))

        # Verify residual is returned
        self.assertEqual(output_residual.shape, (2, 3, self.hidden_size))
        self.assertTrue(torch.allclose(output_residual, updated_residual))

    @patch('torch_npu.npu_add_rms_norm')
    def test_apply_with_residual_different_shapes(self, mock_npu_add_rms_norm):
        """Test apply method with residual with different input shapes."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        mock_layer.bias.data = torch.zeros(self.hidden_size, dtype=torch.float32)
        mock_layer.variance_epsilon = 1e-6

        test_shapes = [
            (self.hidden_size,),  # 1D
            (10, self.hidden_size),  # 2D
            (2, 3, self.hidden_size),  # 3D
        ]

        for shape in test_shapes:
            mock_npu_add_rms_norm.reset_mock()

            x = torch.randn(*shape, dtype=torch.float32)
            residual = torch.randn(*shape, dtype=torch.float32)
            normalized_tensor = torch.randn(*shape, dtype=torch.float32)
            variance = torch.randn(shape[:-1], dtype=torch.float32)
            updated_residual = torch.randn(*shape, dtype=torch.float32)
            mock_npu_add_rms_norm.return_value = (normalized_tensor, variance, updated_residual)

            output, output_residual = self.quant_method.apply(layer=mock_layer, x=x, residual=residual)

            # Verify output shape matches input shape
            self.assertEqual(output.shape, shape)
            self.assertEqual(output_residual.shape, shape)
            mock_npu_add_rms_norm.assert_called_once()

    @patch('torch_npu.npu_rms_norm')
    def test_apply_without_residual_bias_addition(self, mock_npu_rms_norm):
        """Test that bias is added correctly when no residual."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        # Set bias to non-zero values
        mock_layer.bias.data = torch.ones(self.hidden_size, dtype=torch.float32) * 0.5
        mock_layer.variance_epsilon = 1e-6

        x = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        normalized_tensor = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        variance = torch.randn(2, 3, dtype=torch.float32)
        mock_npu_rms_norm.return_value = (normalized_tensor, variance)

        output = self.quant_method.apply(layer=mock_layer, x=x)

        # Verify bias was added
        expected_output = normalized_tensor + mock_layer.bias.data
        self.assertTrue(torch.allclose(output, expected_output))

    @patch('torch_npu.npu_add_rms_norm')
    def test_apply_with_residual_bias_addition(self, mock_npu_add_rms_norm):
        """Test that bias is added correctly when residual is provided."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        # Set bias to non-zero values
        mock_layer.bias.data = torch.ones(self.hidden_size, dtype=torch.float32) * 0.5
        mock_layer.variance_epsilon = 1e-6

        x = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        residual = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        normalized_tensor = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        variance = torch.randn(2, 3, dtype=torch.float32)
        updated_residual = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        mock_npu_add_rms_norm.return_value = (normalized_tensor, variance, updated_residual)

        output, output_residual = self.quant_method.apply(layer=mock_layer, x=x, residual=residual)

        # Verify bias was added
        expected_output = normalized_tensor + mock_layer.bias.data
        self.assertTrue(torch.allclose(output, expected_output))

    @patch('torch_npu.npu_rms_norm')
    def test_apply_without_residual_variance_epsilon(self, mock_npu_rms_norm):
        """Test that variance_epsilon is passed correctly when no residual."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        mock_layer.bias.data = torch.zeros(self.hidden_size, dtype=torch.float32)
        mock_layer.variance_epsilon = 1e-5  # Custom epsilon

        x = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        normalized_tensor = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        variance = torch.randn(2, 3, dtype=torch.float32)
        mock_npu_rms_norm.return_value = (normalized_tensor, variance)

        self.quant_method.apply(layer=mock_layer, x=x)

        # Verify variance_epsilon was passed correctly
        call_args = mock_npu_rms_norm.call_args
        self.assertEqual(call_args[0][2], 1e-5)

    @patch('torch_npu.npu_add_rms_norm')
    def test_apply_with_residual_variance_epsilon(self, mock_npu_add_rms_norm):
        """Test that variance_epsilon is passed correctly when residual is provided."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        mock_layer.bias.data = torch.zeros(self.hidden_size, dtype=torch.float32)
        mock_layer.variance_epsilon = 1e-5  # Custom epsilon

        x = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        residual = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        normalized_tensor = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        variance = torch.randn(2, 3, dtype=torch.float32)
        updated_residual = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        mock_npu_add_rms_norm.return_value = (normalized_tensor, variance, updated_residual)

        self.quant_method.apply(layer=mock_layer, x=x, residual=residual)

        # Verify variance_epsilon was passed correctly
        call_args = mock_npu_add_rms_norm.call_args
        self.assertEqual(call_args[0][3], 1e-5)

    @patch('torch_npu.npu_rms_norm')
    def test_apply_without_residual_return_type(self, mock_npu_rms_norm):
        """Test that apply returns a single tensor when no residual."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        mock_layer.bias.data = torch.zeros(self.hidden_size, dtype=torch.float32)
        mock_layer.variance_epsilon = 1e-6

        x = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        normalized_tensor = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        variance = torch.randn(2, 3, dtype=torch.float32)
        mock_npu_rms_norm.return_value = (normalized_tensor, variance)

        output = self.quant_method.apply(layer=mock_layer, x=x)

        # Should return a single tensor, not a tuple
        self.assertIsInstance(output, torch.Tensor)
        self.assertNotIsInstance(output, tuple)

    @patch('torch_npu.npu_add_rms_norm')
    def test_apply_with_residual_return_type(self, mock_npu_add_rms_norm):
        """Test that apply returns a tuple when residual is provided."""
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.weight = MagicMock()
        mock_layer.weight.data = torch.ones(self.hidden_size, dtype=torch.float32)
        mock_layer.bias = MagicMock()
        mock_layer.bias.data = torch.zeros(self.hidden_size, dtype=torch.float32)
        mock_layer.variance_epsilon = 1e-6

        x = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        residual = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        normalized_tensor = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        variance = torch.randn(2, 3, dtype=torch.float32)
        updated_residual = torch.randn(2, 3, self.hidden_size, dtype=torch.float32)
        mock_npu_add_rms_norm.return_value = (normalized_tensor, variance, updated_residual)

        result = self.quant_method.apply(layer=mock_layer, x=x, residual=residual)

        # Should return a tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        output, output_residual = result
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(output_residual, torch.Tensor)


if __name__ == '__main__':
    unittest.main()

