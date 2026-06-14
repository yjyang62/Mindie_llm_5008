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
import torch

from mindie_llm.runtime.layers.parameter import (
    BaseParameter,
    RowParameter,
    ColumnParameter,
    ModelWeightParameter,
    BiasParameter,
    ScalerParameter,
    PerTensorScaleParameter,
)


class TestBaseParameter(unittest.TestCase):
    """Test cases for BaseParameter."""

    def test_init_requires_grad_false(self):
        """Test that BaseParameter has requires_grad=False by default."""
        data = torch.randn(10, 20)
        param = BaseParameter(data)

        self.assertFalse(param.requires_grad)
        self.assertTrue(torch.allclose(param.data, data))

    def test_weight_loader_property(self):
        """Test weight_loader property getter and setter."""
        param = BaseParameter(torch.randn(10))

        # Initially should be None
        self.assertIsNone(param.weight_loader)

        # Set a weight loader function
        def loader_func(weight):
            pass

        param.weight_loader = loader_func
        self.assertEqual(param.weight_loader, loader_func)

    def test_check_and_copy_success(self):
        """Test _check_and_copy with matching shapes."""
        param_data = torch.randn(10, 20)
        loaded_weight = torch.randn(10, 20)

        BaseParameter._check_and_copy(param_data, loaded_weight)

        self.assertTrue(torch.allclose(param_data, loaded_weight))

    def test_check_and_copy_shape_mismatch(self):
        """Test _check_and_copy raises ValueError on shape mismatch."""
        param_data = torch.randn(10, 20)
        loaded_weight = torch.randn(10, 30)

        with self.assertRaises(ValueError) as context:
            BaseParameter._check_and_copy(param_data, loaded_weight)

        self.assertIn("Tried to load weights", str(context.exception))

    def test_add_attrs(self):
        """Test add_attrs method."""
        param = BaseParameter(torch.randn(10))

        attrs = {"input_dim": 1, "output_dim": 0}
        param.add_attrs(attrs)

        self.assertEqual(param.input_dim, 1)
        self.assertEqual(param.output_dim, 0)

    def test_add_attrs_overwrite_error(self):
        """Test add_attrs raises KeyError when overwriting existing attribute."""
        param = BaseParameter(torch.randn(10))

        param.add_attrs({"input_dim": 1})

        with self.assertRaises(KeyError) as context:
            param.add_attrs({"input_dim": 2})

        self.assertIn("Overwriting existing attribute", str(context.exception))

    def test_load_weight(self):
        """Test load_weight method."""
        param = BaseParameter(torch.randn(10, 20))
        loaded_weight = torch.randn(10, 20)

        param.load_weight(loaded_weight)

        self.assertTrue(torch.allclose(param.data, loaded_weight))

    def test_load_weight_shape_mismatch(self):
        """Test load_weight raises ValueError on shape mismatch."""
        param = BaseParameter(torch.randn(10, 20))
        loaded_weight = torch.randn(10, 30)

        with self.assertRaises(ValueError):
            param.load_weight(loaded_weight)

    def test_check_required_attr_success(self):
        """Test check_required_attr with all attributes present."""
        param = BaseParameter(torch.randn(10))
        param.add_attrs({"input_dim": 1, "output_dim": 0})

        # Should not raise
        param.check_required_attr(["input_dim", "output_dim"])

    def test_check_required_attr_missing(self):
        """Test check_required_attr raises AttributeError when attribute is missing."""
        param = BaseParameter(torch.randn(10))
        param.add_attrs({"input_dim": 1})

        with self.assertRaises(AttributeError) as context:
            param.check_required_attr(["input_dim", "output_dim"])

        self.assertIn("not defined", str(context.exception))


class TestRowParameter(unittest.TestCase):
    """Test cases for RowParameter."""

    def test_load_row_parallel_weight(self):
        """Test load_row_parallel_weight method."""
        # Create parameter with partitioned shape
        param = RowParameter(torch.randn(256, 1000))  # Partitioned along input_dim
        param.add_attrs({"input_dim": 0})

        # Full weight tensor
        full_weight = torch.randn(512, 1000)  # Full size
        tp_rank = 0

        param.load_row_parallel_weight(full_weight, tp_rank)

        # Should load the first 256 rows
        expected = full_weight[:256, :]
        self.assertTrue(torch.allclose(param.data, expected))

    def test_load_row_parallel_weight_rank_1(self):
        """Test load_row_parallel_weight with rank 1."""
        param = RowParameter(torch.randn(256, 1000))
        param.add_attrs({"input_dim": 0})

        full_weight = torch.randn(512, 1000)
        tp_rank = 1

        param.load_row_parallel_weight(full_weight, tp_rank)

        # Should load rows 256:512
        expected = full_weight[256:512, :]
        self.assertTrue(torch.allclose(param.data, expected))

    def test_load_row_parallel_weight_with_custom_offset_and_size(self):
        """Test load_row_parallel_weight with explicit shard offset and size."""
        param = RowParameter(torch.randn(256, 1000))
        param.add_attrs({"input_dim": 0})

        full_weight = torch.randn(512, 1000)
        tp_rank = 0
        loaded_weight_shard_offset = 128  # Custom offset (e.g., uneven split)
        loaded_weight_shard_size = 192

        param.load_row_parallel_weight(
            full_weight, tp_rank,
            loaded_weight_shard_offset=loaded_weight_shard_offset,
            loaded_weight_shard_size=loaded_weight_shard_size,
        )

        # First 192 rows of param should match loaded slice
        expected_slice = full_weight[128:320, :]
        self.assertTrue(torch.allclose(param.data[:192, :], expected_slice))
        # Remaining 64 rows should be zero-padded
        self.assertTrue(torch.allclose(param.data[192:256, :], torch.zeros(64, 1000)))

    def test_load_row_parallel_weight_with_padding(self):
        """Test load_row_parallel_weight when param is larger than loaded shard."""
        param = RowParameter(torch.randn(256, 1000))
        param.add_attrs({"input_dim": 0})

        full_weight = torch.randn(512, 1000)
        tp_rank = 0
        loaded_weight_shard_offset = 0
        loaded_weight_shard_size = 192  # Smaller than param's 256

        param.data.fill_(1.0)
        param.load_row_parallel_weight(
            full_weight, tp_rank,
            loaded_weight_shard_offset=loaded_weight_shard_offset,
            loaded_weight_shard_size=loaded_weight_shard_size,
        )

        # First 192 rows match
        self.assertTrue(torch.allclose(param.data[:192, :], full_weight[:192, :]))
        # Last 64 rows zero-padded
        self.assertTrue(torch.allclose(param.data[192:256, :], torch.zeros(64, 1000)))

    def test_load_row_parallel_weight_missing_attr(self):
        """Test load_row_parallel_weight raises AttributeError without input_dim."""
        param = RowParameter(torch.randn(256, 1000))
        full_weight = torch.randn(512, 1000)

        with self.assertRaises(AttributeError):
            param.load_row_parallel_weight(full_weight, tp_rank=0)


class TestColumnParameter(unittest.TestCase):
    """Test cases for ColumnParameter."""

    def test_load_column_parallel_weight(self):
        """Test load_column_parallel_weight method."""
        # Create parameter with partitioned shape
        param = ColumnParameter(torch.randn(1000, 256))  # Partitioned along output_dim
        param.add_attrs({"output_dim": 1})

        # Full weight tensor
        full_weight = torch.randn(1000, 512)  # Full size
        tp_rank = 0

        param.load_column_parallel_weight(full_weight, tp_rank)

        # Should load the first 256 columns
        expected = full_weight[:, :256]
        self.assertTrue(torch.allclose(param.data, expected))

    def test_load_column_parallel_weight_rank_1(self):
        """Test load_column_parallel_weight with rank 1."""
        param = ColumnParameter(torch.randn(1000, 256))
        param.add_attrs({"output_dim": 1})

        full_weight = torch.randn(1000, 512)
        tp_rank = 1

        param.load_column_parallel_weight(full_weight, tp_rank)

        # Should load columns 256:512
        expected = full_weight[:, 256:512]
        self.assertTrue(torch.allclose(param.data, expected))

    def test_load_column_parallel_weight_missing_attr(self):
        """Test load_column_parallel_weight raises AttributeError without output_dim."""
        param = ColumnParameter(torch.randn(1000, 256))
        full_weight = torch.randn(1000, 512)

        with self.assertRaises(AttributeError):
            param.load_column_parallel_weight(full_weight, tp_rank=0)

    def test_load_merged_column_weight(self):
        """Test load_merged_column_weight method."""
        param = ColumnParameter(torch.randn(1000, 512))
        param.add_attrs({"output_dim": 1})

        full_weight = torch.randn(1000, 1024)
        tp_rank = 0
        shard_offset = 0
        shard_size = 256

        param.load_merged_column_weight(full_weight, tp_rank, shard_offset, shard_size)

        # Should load columns 0:256 from the first 256 columns of full_weight
        expected = full_weight[:, :256]
        self.assertTrue(torch.allclose(param.data[:, shard_offset:shard_offset+shard_size], expected))

    def test_load_qkv_weight_q_shard(self):
        """Test load_qkv_weight with Q shard."""
        param = ColumnParameter(torch.randn(1000, 256))
        param.add_attrs({"output_dim": 1})

        full_weight = torch.randn(1000, 768)  # QKV combined
        shard_offset = 0
        shard_size = 256
        loaded_weight_shard_offset = 256  # Rank 1's slice
        loaded_weight_shard_size = 256

        param.load_qkv_weight(
            full_weight,
            shard_offset=shard_offset,
            shard_size=shard_size,
            loaded_weight_shard_offset=loaded_weight_shard_offset,
            loaded_weight_shard_size=loaded_weight_shard_size,
        )

        expected = full_weight[:, loaded_weight_shard_offset:loaded_weight_shard_offset + loaded_weight_shard_size]
        self.assertTrue(torch.allclose(param.data[:, shard_offset:shard_offset + shard_size], expected))

    def test_load_qkv_weight_with_padding(self):
        """Test load_qkv_weight when param shard is larger than loaded shard (zero-padding)."""
        param = ColumnParameter(torch.randn(1000, 256))
        param.add_attrs({"output_dim": 1})

        full_weight = torch.randn(1000, 192)  # Smaller loaded shard (e.g., uneven head split)
        shard_offset = 0
        shard_size = 256
        loaded_weight_shard_offset = 0
        loaded_weight_shard_size = 192

        param.data.fill_(1.0)  # Fill with non-zero to verify padding is zeros
        param.load_qkv_weight(
            full_weight,
            shard_offset=shard_offset,
            shard_size=shard_size,
            loaded_weight_shard_offset=loaded_weight_shard_offset,
            loaded_weight_shard_size=loaded_weight_shard_size,
        )

        # First 192 columns should match loaded weight
        self.assertTrue(torch.allclose(param.data[:, :192], full_weight))
        # Last 64 columns should be zero (padding)
        self.assertTrue(torch.allclose(param.data[:, 192:256], torch.zeros(1000, 64)))

    def test_load_qkv_weight_kv_shard(self):
        """Test load_qkv_weight with K/V shard."""
        param = ColumnParameter(torch.randn(1000, 256))
        param.add_attrs({"output_dim": 1})

        full_weight = torch.randn(1000, 768)  # QKV combined
        shard_offset = 0
        shard_size = 256
        loaded_weight_shard_offset = 256  # Effective rank 1's slice
        loaded_weight_shard_size = 256

        param.load_qkv_weight(
            full_weight,
            shard_offset=shard_offset,
            shard_size=shard_size,
            loaded_weight_shard_offset=loaded_weight_shard_offset,
            loaded_weight_shard_size=loaded_weight_shard_size,
        )

        expected = full_weight[:, loaded_weight_shard_offset:loaded_weight_shard_offset + loaded_weight_shard_size]
        self.assertTrue(torch.allclose(param.data[:, shard_offset:shard_offset + shard_size], expected))


class TestModelWeightParameter(unittest.TestCase):
    """Test cases for ModelWeightParameter."""

    def test_inherits_both_methods(self):
        """Test that ModelWeightParameter inherits both row and column methods."""
        param = ModelWeightParameter(torch.randn(256, 1000))
        param.add_attrs({"input_dim": 0, "output_dim": 1})

        # Should have both methods
        self.assertTrue(hasattr(param, 'load_row_parallel_weight'))
        self.assertTrue(hasattr(param, 'load_column_parallel_weight'))

    def test_load_row_parallel_weight(self):
        """Test ModelWeightParameter can use load_row_parallel_weight."""
        param = ModelWeightParameter(torch.randn(256, 1000))
        param.add_attrs({"input_dim": 0})

        full_weight = torch.randn(512, 1000)
        param.load_row_parallel_weight(full_weight, tp_rank=0)

        expected = full_weight[:256, :]
        self.assertTrue(torch.allclose(param.data, expected))

    def test_load_column_parallel_weight(self):
        """Test ModelWeightParameter can use load_column_parallel_weight."""
        param = ModelWeightParameter(torch.randn(1000, 256))
        param.add_attrs({"output_dim": 1})

        full_weight = torch.randn(1000, 512)
        param.load_column_parallel_weight(full_weight, tp_rank=0)

        expected = full_weight[:, :256]
        self.assertTrue(torch.allclose(param.data, expected))


class TestBiasParameter(unittest.TestCase):
    """Test cases for BiasParameter."""

    def test_load_row_parallel_weight_rank_0(self):
        """Test BiasParameter.load_row_parallel_weight with rank 0 loads weight."""
        param = BiasParameter(torch.randn(256))
        param.add_attrs({"input_dim": 0})

        full_weight = torch.randn(512)
        tp_rank = 0

        param.load_row_parallel_weight(full_weight, tp_rank)

        # Rank 0 should load the weight
        expected = full_weight[:256]
        self.assertTrue(torch.allclose(param.data, expected))

    def test_load_row_parallel_weight_rank_nonzero(self):
        """Test BiasParameter.load_row_parallel_weight with non-zero rank zeros out."""
        param = BiasParameter(torch.randn(256))
        param.add_attrs({"input_dim": 0})

        # Initialize with non-zero values
        param.data.fill_(1.0)

        full_weight = torch.randn(512)
        tp_rank = 1

        param.load_row_parallel_weight(full_weight, tp_rank)

        # Non-zero rank should zero out the bias
        self.assertTrue(torch.allclose(param.data, torch.zeros_like(param.data)))

    def test_load_row_parallel_weight_rank_0_with_custom_offset_size(self):
        """Test BiasParameter passes custom offset/size to parent when rank 0."""
        param = BiasParameter(torch.randn(256))
        param.add_attrs({"input_dim": 0})

        full_weight = torch.randn(512)
        tp_rank = 0
        loaded_weight_shard_offset = 64
        loaded_weight_shard_size = 192

        param.data.fill_(1.0)
        param.load_row_parallel_weight(
            full_weight, tp_rank,
            loaded_weight_shard_offset=loaded_weight_shard_offset,
            loaded_weight_shard_size=loaded_weight_shard_size,
        )

        # First 192 elements from loaded slice
        expected = full_weight[64:256]
        self.assertTrue(torch.allclose(param.data[:192], expected))
        # Remainder zero-padded
        self.assertTrue(torch.allclose(param.data[192:256], torch.zeros(64)))

    def test_inherits_column_methods(self):
        """Test that BiasParameter also inherits column methods."""
        param = BiasParameter(torch.randn(256))
        param.add_attrs({"output_dim": 0})

        full_weight = torch.randn(512)
        param.load_column_parallel_weight(full_weight, tp_rank=0)

        expected = full_weight[:256]
        self.assertTrue(torch.allclose(param.data, expected))


class TestScalerParameter(unittest.TestCase):
    """Test cases for ScalerParameter."""

    def test_inherits_base_methods(self):
        """Test that ScalerParameter inherits BaseParameter methods."""
        param = ScalerParameter(torch.randn(10))

        # Should have BaseParameter methods
        self.assertTrue(hasattr(param, 'load_weight'))
        self.assertTrue(hasattr(param, 'add_attrs'))
        self.assertFalse(param.requires_grad)


class TestPerTensorScaleParameter(unittest.TestCase):
    """Test cases for PerTensorScaleParameter."""

    def test_inherits_column_methods(self):
        """Test that PerTensorScaleParameter inherits ColumnParameter methods."""
        param = PerTensorScaleParameter(torch.randn(256))
        param.add_attrs({"output_dim": 0})

        # Should have ColumnParameter methods
        self.assertTrue(hasattr(param, 'load_column_parallel_weight'))
        self.assertTrue(hasattr(param, 'load_merged_column_weight'))

    def test_load_column_parallel_weight(self):
        """Test PerTensorScaleParameter can use load_column_parallel_weight."""
        param = PerTensorScaleParameter(torch.randn(256))
        param.add_attrs({"output_dim": 0})

        full_weight = torch.randn(512)
        param.load_column_parallel_weight(full_weight, tp_rank=0)

        expected = full_weight[:256]
        self.assertTrue(torch.allclose(param.data, expected))


if __name__ == '__main__':
    unittest.main()
