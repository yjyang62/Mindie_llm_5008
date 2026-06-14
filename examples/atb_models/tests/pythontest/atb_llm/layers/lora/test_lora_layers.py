#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import Mock, MagicMock, patch

import math
import random
import torch

from atb_llm.nn.tensor import Tensor
from atb_llm.nn.parameter import Parameter
from atb_llm.layers.linear.linear import BaseLinear, ColumnParallelLinear, RowParallelLinear
from atb_llm.layers.lora.lora_layers import ColumnParallelLinearWithLoRA, RowParallelLinearWithLoRA
from atb_llm.models.base.config import BaseConfig, LoraModelConfig
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.quantize.quant_type import LinearTypeV2


class MockModelParam(MagicMock):
    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.hf_config = MagicMock()
        self.hf_config.torch_dtype = torch.float16
        self.soc_info = MagicMock()
        self.soc_info.need_nz = False
        self.mapping = MagicMock()
        self.mapping.rank = 1
        self.mapping.world_size = 4
        self.lora_config = LoraModelConfig(max_loras=5, max_lora_rank=128)


class FakeParallelLinear(BaseLinear):
    def __init__(self, prefixes):
        config = BaseConfig(torch_dtype=torch.float16)
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        file_loader = weight_tool_cls()
        super().__init__(config, file_loader, prefixes, bias=False)


class FakeColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, weight_tensor, prefixes):
        config = BaseConfig(torch_dtype=torch.float16)
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        file_loader = weight_tool_cls()
        file_loader.mapping = MagicMock(spec=Mapping)
        file_loader.mapping.rank = 1
        file_loader.mapping.world_size = 4
        file_loader.get_sharded = MagicMock(return_value=weight_tensor)
        file_loader.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        super().__init__(config, file_loader, prefixes, bias=False)


class FakeRowParallelLinear(RowParallelLinear):
    def __init__(self, weight_tensor, prefixes):
        config = BaseConfig(torch_dtype=torch.float16)
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        file_loader = weight_tool_cls()
        file_loader.mapping = MagicMock(spec=Mapping)
        file_loader.mapping.rank = 1
        file_loader.mapping.world_size = 4
        file_loader.get_sharded = MagicMock(return_value=weight_tensor)
        file_loader.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        super().__init__(config, file_loader, prefixes, bias=False)


class TestLoraLayers(unittest.TestCase):
    def setUp(self):
        world_size = 2 * random.randint(1, 4)
        self.k = world_size * random.randint(1, 1024)
        self.n = world_size * random.randint(1, 1024)
        self.r = 2 ** random.randint(1, 6)
        self.mindie_llm_config = MockModelParam()
        self.max_loras = self.mindie_llm_config.lora_config.max_loras
        self.max_lora_rank = self.mindie_llm_config.lora_config.max_lora_rank
        self.need_nz = self.mindie_llm_config.soc_info.need_nz
        self.dtype = self.mindie_llm_config.hf_config.torch_dtype
        self.device = torch.device("cpu")

    @patch("atb_llm.layers.base_layer.BaseLayer.weight_loader")
    def test_get_base_weight(self, mock_weight_loader_func):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        mock_weight_loader_func.return_value = linear_tensor
        linear_layer = FakeParallelLinear(["linear"])
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        self.assertTrue(torch.allclose(lora_layer.weight, linear_tensor))
        self.assertIsNone(lora_layer.bias)

    @patch("atb_llm.layers.base_layer.BaseLayer.weight_loader")
    def test_parallel_linear_with_lora_create_weights_fp32(self, mock_weight_loader_func):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        mock_weight_loader_func.return_value = linear_tensor
        linear_layer = FakeParallelLinear(["linear"])
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        mindie_llm_config = MockModelParam()
        mindie_llm_config.hf_config.torch_dtype = torch.float
        with self.assertRaises(RuntimeError):
            lora_layer.create_lora_weights(mindie_llm_config, self.device)

    @patch("atb_llm.layers.base_layer.BaseLayer.weight_loader")
    def test_parallel_linear_with_lora_create_weights(self, mock_weight_loader_func):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        mock_weight_loader_func.return_value = linear_tensor
        linear_layer = FakeParallelLinear(["linear"])
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.create_lora_weights(self.mindie_llm_config, self.device)
        dim_r = math.ceil(self.max_lora_rank / 16) * 16 if self.need_nz else math.ceil(self.max_lora_rank / 64) * 64
        lora_a = torch.zeros(self.max_loras + 1, dim_r, self.k, dtype=self.dtype)
        lora_b = torch.zeros(self.max_loras + 1, dim_r, self.n, dtype=self.dtype)
        self.assertIsInstance(lora_layer.lora_a_stacked, Parameter)
        self.assertIsInstance(lora_layer.lora_b_stacked, Parameter)
        self.assertTrue(torch.allclose(lora_layer.lora_a_stacked.cpu(), lora_a))
        self.assertTrue(torch.allclose(lora_layer.lora_b_stacked.cpu(), lora_b))

    @patch("atb_llm.layers.base_layer.BaseLayer.weight_loader")
    def test_parallel_linear_with_lora_set_lora(self, mock_weight_loader_func):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        mock_weight_loader_func.return_value = linear_tensor
        linear_layer = FakeParallelLinear(["linear"])
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_layer.device = self.device
        dim_r = math.ceil(self.max_lora_rank / 16) * 16 if self.need_nz else math.ceil(self.max_lora_rank / 64) * 64
        lora_layer.lora_a_stacked.data = torch.zeros(self.max_loras + 1, dim_r, self.k, dtype=self.dtype, device=self.device)
        lora_layer.lora_b_stacked.data = torch.zeros(self.max_loras + 1, dim_r, self.n, dtype=self.dtype, device=self.device)
        lora_a = torch.rand((self.r, self.k), device=self.device, dtype=torch.float16)
        lora_b = torch.rand((self.r, self.n), device=self.device, dtype=torch.float16)
        index = random.randint(0, self.max_loras - 1)
        lora_layer.set_lora(index, lora_a, lora_b)
        self.assertTrue(torch.allclose(lora_layer.lora_a_stacked[index, :self.r].cpu(), lora_a.cpu()))
        self.assertTrue(torch.allclose(lora_layer.lora_b_stacked[index, :self.r].cpu(), lora_b.cpu()))

    @patch("atb_llm.layers.base_layer.BaseLayer.weight_loader")
    def test_parallel_linear_with_lora_reset_lora(self, mock_weight_loader_func):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        mock_weight_loader_func.return_value = linear_tensor
        linear_layer = FakeParallelLinear(["linear"])
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_layer.device = self.device
        dim_r = math.ceil(self.max_lora_rank / 16) * 16 if self.need_nz else math.ceil(self.max_lora_rank / 64) * 64
        lora_layer.lora_a_stacked.data = torch.rand(self.max_loras + 1, dim_r, self.k, dtype=self.dtype, device=self.device)
        lora_layer.lora_b_stacked.data = torch.rand(self.max_loras + 1, dim_r, self.n, dtype=self.dtype, device=self.device)
        index = random.randint(0, self.max_loras - 1)
        lora_layer.reset_lora(index)
        lora_a = torch.zeros(dim_r, self.k, dtype=self.dtype)
        lora_b = torch.zeros(dim_r, self.n, dtype=self.dtype)
        self.assertTrue(torch.allclose(lora_layer.lora_a_stacked[index].cpu(), lora_a))
        self.assertTrue(torch.allclose(lora_layer.lora_b_stacked[index].cpu(), lora_b))

    @patch("atb_llm.layers.linear.linear.nn.functional.grouped_matmul")
    @patch("atb_llm.layers.linear.linear.nn.functional.linear")
    @patch("atb_llm.layers.base_layer.BaseLayer.weight_loader")
    def test_parallel_linear_with_lora_forward_disable_lora(self, mock_weight_loader_func, mock_matmul, mock_gmm):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        mock_weight_loader_func.return_value = linear_tensor
        linear_layer = FakeParallelLinear(["linear"])
        linear_layer.forward = Mock(return_value=Tensor("out"))
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        out = lora_layer.forward(Tensor("input"))
        self.assertEqual(out, Tensor("out"))
        mock_matmul.assert_not_called()
        mock_gmm.assert_not_called()

    @patch("atb_llm.layers.linear.linear.nn.functional.grouped_matmul")
    @patch("atb_llm.layers.linear.linear.nn.functional.linear")
    @patch("atb_llm.layers.base_layer.BaseLayer.weight_loader")
    def test_parallel_linear_with_lora_forward_matmul(self, mock_weight_loader_func, mock_matmul, mock_gmm):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        mock_weight_loader_func.return_value = linear_tensor
        linear_layer = FakeParallelLinear(["linear"])
        linear_layer.forward = Mock(return_value=Tensor("fake_out"))
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        mock_matmul.return_value = Mock(side_effect=[Tensor("fake_out1"), Tensor("fake_out2")])
        mock_gmm.return_value = Mock(side_effect=[Tensor("fake_out1"), Tensor("fake_out2")])
        out = lora_layer.forward(Tensor("input"), enable_lora=True)
        self.assertIsNotNone(out)
        mock_gmm.assert_not_called()

    @patch("atb_llm.layers.linear.linear.nn.functional.grouped_matmul")
    @patch("atb_llm.layers.linear.linear.nn.functional.linear")
    @patch("atb_llm.layers.base_layer.BaseLayer.weight_loader")
    def test_parallel_linear_with_lora_forward_gmm(self, mock_weight_loader_func, mock_matmul, mock_gmm):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        mock_weight_loader_func.return_value = linear_tensor
        linear_layer = FakeParallelLinear(["linear"])
        linear_layer.forward = Mock(return_value=Tensor("fake_out"))
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        mock_matmul.return_value = Mock(side_effect=[Tensor("fake_out1"), Tensor("fake_out2")])
        mock_gmm.return_value = Mock(side_effect=[Tensor("fake_out1"), Tensor("fake_out2")])
        out = lora_layer.forward(Tensor("input"), enable_lora=True, group_list=Tensor("group_list"))
        self.assertIsNotNone(out)
        mock_matmul.assert_not_called()

    def test_column_parallel_linear_with_lora(self):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        linear_layer = FakeColumnParallelLinear(linear_tensor, ["linear"])
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        tensor_1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float16)
        tensor_2 = torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=torch.float16)
        mock_weight_tool_obj.get_sharded = MagicMock(return_value=tensor_1)
        mock_weight_tool_obj.get_tensor = MagicMock(return_value=tensor_2)
        mock_weight_tool_obj.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_a = lora_layer.load_lora_a(mock_weight_tool_obj, ["lora_A"])
        lora_b = lora_layer.load_lora_b(mock_weight_tool_obj, ["lora_B"], [1])
        self.assertTrue(torch.allclose(lora_a, tensor_2))
        self.assertTrue(torch.allclose(lora_b, tensor_1.T.contiguous()))
        mock_weight_tool_obj.get_sharded.assert_called_once()
        mock_weight_tool_obj.get_tensor.assert_called_once()

    def test_merged_column_parallel_linear_with_lora(self):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        linear_layer = FakeColumnParallelLinear(linear_tensor, ["linear1", "linear2"])
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        tensor_1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float16)
        tensor_2 = torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=torch.float16)
        mock_weight_tool_obj.get_sharded = MagicMock(return_value=tensor_1)
        mock_weight_tool_obj.get_tensor = MagicMock(return_value=tensor_2)
        mock_weight_tool_obj.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_a = lora_layer.load_lora_a(mock_weight_tool_obj, ["lora1_A", "lora2_A"])
        lora_b = lora_layer.load_lora_b(mock_weight_tool_obj, ["lora1_B", "lora2_B"], [1, 1])
        self.assertTrue(torch.allclose(lora_a, torch.cat([tensor_2] * 2, dim=0)))
        self.assertTrue(torch.allclose(lora_b, torch.block_diag(*([tensor_1] * 2)).T.contiguous()))

    def test_row_parallel_linear_with_lora(self):
        linear_tensor = torch.rand((self.n, self.k), device=self.device, dtype=torch.float16)
        linear_layer = FakeRowParallelLinear(linear_tensor, ["linear"])
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        tensor_1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float16)
        tensor_2 = torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=torch.float16)
        mock_weight_tool_obj.get_sharded = MagicMock(return_value=tensor_1)
        mock_weight_tool_obj.get_tensor = MagicMock(return_value=tensor_2)
        mock_weight_tool_obj.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        lora_layer = RowParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_a = lora_layer.load_lora_a(mock_weight_tool_obj, ["lora_A"])
        lora_b = lora_layer.load_lora_b(mock_weight_tool_obj, ["lora_B"], [1])
        self.assertTrue(torch.allclose(lora_a, tensor_1))
        self.assertTrue(torch.allclose(lora_b, tensor_2.T.contiguous()))
        mock_weight_tool_obj.get_sharded.assert_called_once()
        mock_weight_tool_obj.get_tensor.assert_called_once()


if __name__ == '__main__':
    unittest.main()