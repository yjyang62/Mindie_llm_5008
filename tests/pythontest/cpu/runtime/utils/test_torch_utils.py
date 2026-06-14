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

from mindie_llm.runtime.utils.torch_utils import set_default_torch_dtype


class TestSetDefaultTorchDtype(unittest.TestCase):
    """Test cases for set_default_torch_dtype context manager."""

    def setUp(self):
        """Set up test fixtures."""
        # Save original default dtype
        self.original_dtype = torch.get_default_dtype()

    def tearDown(self):
        """Clean up after tests."""
        # Restore original default dtype
        torch.set_default_dtype(self.original_dtype)

    def test_set_default_torch_dtype_float32(self):
        """Test setting default dtype to float32."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(torch.float32):
            # Verify dtype was set inside context
            self.assertEqual(torch.get_default_dtype(), torch.float32)
            
            # Verify new tensors use the new dtype
            tensor = torch.tensor([1.0, 2.0, 3.0])
            self.assertEqual(tensor.dtype, torch.float32)
        
        # Verify dtype was restored after context
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_float16(self):
        """Test setting default dtype to float16."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(torch.float16):
            # Verify dtype was set inside context
            self.assertEqual(torch.get_default_dtype(), torch.float16)
            
            # Verify new tensors use the new dtype
            tensor = torch.tensor([1.0, 2.0, 3.0])
            self.assertEqual(tensor.dtype, torch.float16)
        
        # Verify dtype was restored after context
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_bfloat16(self):
        """Test setting default dtype to bfloat16."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(torch.bfloat16):
            # Verify dtype was set inside context
            self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
            
            # Verify new tensors use the new dtype
            tensor = torch.tensor([1.0, 2.0, 3.0])
            self.assertEqual(tensor.dtype, torch.bfloat16)
        
        # Verify dtype was restored after context
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_float64(self):
        """Test setting default dtype to float64."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(torch.float64):
            # Verify dtype was set inside context
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            
            # Verify new tensors use the new dtype
            tensor = torch.tensor([1.0, 2.0, 3.0])
            self.assertEqual(tensor.dtype, torch.float64)
        
        # Verify dtype was restored after context
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_restores_after_exception(self):
        """Test that dtype is restored even when exception occurs."""
        original_dtype = torch.get_default_dtype()
        
        try:
            with set_default_torch_dtype(torch.float16):
                # Verify dtype was set inside context
                self.assertEqual(torch.get_default_dtype(), torch.float16)
                
                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify dtype was restored even after exception
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_nested_contexts(self):
        """Test nested context managers."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(torch.float16):
            self.assertEqual(torch.get_default_dtype(), torch.float16)
            
            with set_default_torch_dtype(torch.float32):
                self.assertEqual(torch.get_default_dtype(), torch.float32)
                
                tensor = torch.tensor([1.0, 2.0])
                self.assertEqual(tensor.dtype, torch.float32)
            
            # After inner context, should be back to float16
            self.assertEqual(torch.get_default_dtype(), torch.float16)
            
            tensor = torch.tensor([1.0, 2.0])
            self.assertEqual(tensor.dtype, torch.float16)
        
        # After outer context, should be back to original
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_multiple_tensors(self):
        """Test that multiple tensors created inside context use new dtype."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(torch.float16):
            tensor1 = torch.tensor([1.0, 2.0])
            tensor2 = torch.tensor([3.0, 4.0])
            tensor3 = torch.randn(5, 5)
            
            self.assertEqual(tensor1.dtype, torch.float16)
            self.assertEqual(tensor2.dtype, torch.float16)
            self.assertEqual(tensor3.dtype, torch.float16)
        
        # Verify dtype was restored
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_context_manager_returns_none(self):
        """Test that context manager yields None."""
        with set_default_torch_dtype(torch.float16) as ctx:
            # Context manager should yield None
            self.assertIsNone(ctx)

    def test_set_default_torch_dtype_same_dtype(self):
        """Test setting dtype to the same value."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(original_dtype):
            # Should still work correctly
            self.assertEqual(torch.get_default_dtype(), original_dtype)
            
            tensor = torch.tensor([1.0, 2.0])
            self.assertEqual(tensor.dtype, original_dtype)
        
        # Should still be the same after context
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_empty_context(self):
        """Test context manager with empty body."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(torch.float16):
            pass
        
        # Verify dtype was restored
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_sequential_contexts(self):
        """Test sequential context managers."""
        original_dtype = torch.get_default_dtype()
        
        # First context
        with set_default_torch_dtype(torch.float16):
            self.assertEqual(torch.get_default_dtype(), torch.float16)
        
        # Verify restored
        self.assertEqual(torch.get_default_dtype(), original_dtype)
        
        # Second context
        with set_default_torch_dtype(torch.float32):
            self.assertEqual(torch.get_default_dtype(), torch.float32)
        
        # Verify restored again
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_with_explicit_dtype_override(self):
        """Test that explicit dtype in tensor creation overrides default."""
        original_dtype = torch.get_default_dtype()
        
        with set_default_torch_dtype(torch.float16):
            # Explicit dtype should override default
            tensor1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
            self.assertEqual(tensor1.dtype, torch.float32)
            
            # No explicit dtype should use default
            tensor2 = torch.tensor([1.0, 2.0])
            self.assertEqual(tensor2.dtype, torch.float16)
        
        # Verify dtype was restored
        self.assertEqual(torch.get_default_dtype(), original_dtype)

    def test_set_default_torch_dtype_exception_in_context(self):
        """Test exception handling within context."""
        original_dtype = torch.get_default_dtype()
        
        try:
            with set_default_torch_dtype(torch.float16):
                self.assertEqual(torch.get_default_dtype(), torch.float16)
                
                # Create some tensors
                tensor1 = torch.tensor([1.0, 2.0])
                self.assertEqual(tensor1.dtype, torch.float16)
                
                # Raise exception
                raise RuntimeError("Test error")
        except RuntimeError:
            pass
        
        # Verify dtype was restored even after exception
        self.assertEqual(torch.get_default_dtype(), original_dtype)
        
        # Verify new tensors use original dtype
        tensor2 = torch.tensor([1.0, 2.0])
        self.assertEqual(tensor2.dtype, original_dtype)


if __name__ == '__main__':
    unittest.main()
