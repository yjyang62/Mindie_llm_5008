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
from unittest.mock import MagicMock, patch, Mock
import torch
import torch_npu
import torch.nn as nn

from mindie_llm.runtime.layers.quantization.ms_model_slim.w4a8 import W4A8PerTokenFusedMoEMethod





class TestW4A8PerTokenFusedMoEMethod(unittest.TestCase):
    """Test cases for W4A8PerTokenFusedMoEMethod."""

    def setUp(self):
        self.layer = nn.Module()
        self.layer.__setattr__("moe_tp_size",1)
        self.num_experts = 4
        self.hidden_size = 64
        self.intermediate_size_per_partition = 128
        self.extra_attrs = {}
        self.method = W4A8PerTokenFusedMoEMethod()

    def test_create_weights(self):
        """Test create_weights method."""
        self.method.create_weights(
            layer=self.layer,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            **self.extra_attrs
        )

        self.assertTrue(hasattr(self.layer, "gate_up_weight"))
        self.assertTrue(hasattr(self.layer, "down_weight"))
        self.assertTrue(hasattr(self.layer, "gate_up_weight_scale"))
        self.assertTrue(hasattr(self.layer, "down_weight_scale"))
        self.assertTrue(hasattr(self.layer, "gate_up_scale_bias"))
        self.assertTrue(hasattr(self.layer, "down_scale_bias"))

        self.assertEqual(self.layer.gate_up_weight.shape, (4, 128, 64))
        self.assertEqual(self.layer.down_weight.shape, (4, 32, 128))
        self.assertEqual(self.layer.gate_up_weight_scale.shape, (4, 256, 1))
        self.assertEqual(self.layer.down_weight_scale.shape, (4, 64, 1))
        self.assertEqual(self.layer.gate_up_scale_bias.shape, (4, 256, 1))
        self.assertEqual(self.layer.down_scale_bias.shape, (4, 64, 16))

    @patch("torch_npu.npu_dynamic_quant")
    @patch("torch_npu.npu_grouped_matmul")
    @patch("torch_npu.npu_swiglu")
    def test_apply(self, mock_npu_swiglu, mock_npu_grouped_matmul, mock_npu_dynamic_quant):
        """Test apply method."""
        x = torch.randn(2, 64, dtype=torch.float32)
        group_list = torch.tensor([0, 1, 1, 0], dtype=torch.int32)
        group_list_type = 1
        
        self.layer.gate_up_weight = torch.randint(-128, 127,(4, 128, 64), dtype=torch.int8)
        self.layer.down_weight = torch.randint(-128, 127,(4, 32, 128), dtype=torch.int8)
        self.layer.gate_up_weight_scale = torch.randn(4, 256, 1, dtype=torch.float32)
        self.layer.down_weight_scale = torch.randn(4, 64, 1, dtype=torch.float32)
        self.layer.gate_up_scale_bias = torch.randn(4, 256, 1, dtype=torch.float32)
        self.layer.down_scale_bias = torch.randn(4, 64, 8, dtype=torch.float32)
        
        mock_npu_dynamic_quant.return_value = (x, torch.randn(64, dtype=torch.float32))
        
        mock_npu_grouped_matmul.return_value = (torch.randn(2, 256, dtype=torch.float32),)
        
        mock_npu_swiglu.return_value = torch.randn(2, 128, dtype=torch.float32)

        output = self.method.apply(self.layer, x, group_list, group_list_type)
        # 检查是否调用了 mock 函数
        mock_npu_dynamic_quant.assert_called()
        mock_npu_grouped_matmul.assert_called() 
        mock_npu_swiglu.assert_called_once()
        
        self.assertEqual(output.shape, (2, 256))


    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading(self, mock_npu_format_cast):
        # Test the process_weights_after_loading method
        self.layer.gate_up_weight = torch.randint(-128, 127,(4, 128, 64), dtype=torch.int8)
        self.layer.down_weight = torch.randint(-128, 127, (4, 128, 64), dtype=torch.int8)
        self.layer.gate_up_weight_scale = torch.randn(4, 256, 1, dtype=torch.float32)
        self.layer.down_weight_scale = torch.randn(4, 64, 1, dtype=torch.float32)
        self.layer.gate_up_scale_bias = torch.randn(4, 256, 1, dtype=torch.float32)
        self.layer.down_scale_bias = torch.randn(4, 64, 8, dtype=torch.float32)
        mock_npu_format_cast.return_value = self.layer.gate_up_weight.data
        
        self.method.process_weights_after_loading(self.layer)
        self.assertEqual(self.layer.gate_up_weight.shape, (4, 128, 16))
        self.assertEqual(self.layer.down_weight.shape, (4, 128, 16))
        self.assertEqual(self.layer.gate_up_weight_scale.shape, (4, 1, 256))
        self.assertEqual(self.layer.down_weight_scale.shape, (4, 1, 64, 1))
        self.assertEqual(self.layer.gate_up_scale_bias.shape, (4, 256))
        self.assertEqual(self.layer.down_scale_bias.shape, (4, 64))


if __name__ == '__main__':
    unittest.main()
