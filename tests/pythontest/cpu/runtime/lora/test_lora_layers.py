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

import math
import random
import torch
from torch.nn import Parameter

from mindie_llm.runtime.config.mindie_llm_config import LoraModelConfig
from mindie_llm.runtime.layers.linear.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear
)
from mindie_llm.runtime.lora.lora_layers import (
    ParallelLinearWithLoRA,
    ColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA
)
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelInfo


class FakeColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, prefix, parallel_info):
        input_size = random.randint(1, 1024)
        output_size = parallel_info.group_size * random.randint(1, 1024)
        weight_dtype = torch.float16
        self.parallel_info = parallel_info
        super().__init__(input_size=input_size, output_size=output_size,
            weight_dtype=weight_dtype, prefix=prefix, parallel_info=parallel_info)


class FakeMergedColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(self, prefix, parallel_info):
        input_size = random.randint(1, 1024)
        output_sizes = [parallel_info.group_size * random.randint(1, 1024), \
            parallel_info.group_size * random.randint(1, 1024)]
        weight_dtype = torch.float16
        super().__init__(input_size=input_size, output_sizes=output_sizes,
            weight_dtype=weight_dtype, prefix=prefix, parallel_info=parallel_info)


class FakeQKVParallelLinear(QKVParallelLinear):
    def __init__(self, prefix, parallel_info):
        head_size = random.randint(1, 64)
        total_num_heads = parallel_info.group_size * 16
        total_num_kv_heads = parallel_info.group_size * 4
        hidden_size = head_size * total_num_heads
        weight_dtype = torch.float16
        super().__init__(hidden_size=hidden_size, head_size=head_size, total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads, weight_dtype=weight_dtype, \
            prefix=prefix, parallel_info=parallel_info)


class FakeRowParallelLinear(RowParallelLinear):
    def __init__(self, prefix, parallel_info):
        input_size = parallel_info.group_size * random.randint(1, 1024)
        output_size = random.randint(1, 1024)
        weight_dtype = torch.float16
        super().__init__(input_size=input_size, output_size=output_size,
            weight_dtype=weight_dtype, prefix=prefix, parallel_info=parallel_info)


class TestLoraLayers(unittest.TestCase):
    def setUp(self):
        self.world_size = 2 ** random.randint(0, 2)
        self.tp_rank = random.randint(0, self.world_size - 1)
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = self.tp_rank
        self.mock_parallel_info_manager.world_size = self.world_size
        self.parallel_info = ParallelInfo()
        self.parallel_info.group_size = self.world_size
        self.parallel_info.rank = self.tp_rank
        self.r = 2 ** random.randint(1, 6)
        self.soc_info = MagicMock()
        self.soc_info.need_nz = False
        self.device = torch.device("cpu")
        self.dtype = torch.float16
        self.max_loras = random.randint(1, 10)
        self.max_lora_rank = 128
        self.lora_model_config = LoraModelConfig(max_loras=self.max_loras, max_lora_rank=self.max_lora_rank)

    @patch.object(ParallelLinearWithLoRA, "weight_format_cast")
    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_parallel_linear_with_lora_create_weights_fp32(self, mock_get_parallel_info_manager, \
        mock_weight_format_cast):
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        mock_weight_format_cast.side_effect = lambda x: x
        linear_layer = FakeColumnParallelLinear(["linear"], self.parallel_info)
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        dtype = torch.float32
        with self.assertRaises(RuntimeError):
            lora_layer.create_lora_weights(self.lora_model_config, dtype, self.device)

    @patch.object(ParallelLinearWithLoRA, "weight_format_cast")
    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_parallel_linear_with_lora_create_weights(self, mock_get_parallel_info_manager, mock_weight_format_cast):
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        mock_weight_format_cast.side_effect = lambda x: x
        linear_layer = FakeColumnParallelLinear(["linear"], self.parallel_info)
        n, k = sum(linear_layer.output_partition_sizes), linear_layer.input_size_per_partition
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.create_lora_weights(self.lora_model_config, self.dtype, self.device)
        dim_r = math.ceil(self.max_lora_rank / 16) * 16 if self.soc_info.need_nz \
            else math.ceil(self.max_lora_rank / 64) * 64
        lora_a = torch.zeros(self.max_loras + 1, dim_r, k, dtype=self.dtype)
        lora_b = torch.zeros(self.max_loras + 1, dim_r, n, dtype=self.dtype)
        self.assertIsInstance(lora_layer.lora_a_stacked, Parameter)
        self.assertIsInstance(lora_layer.lora_b_stacked, Parameter)
        self.assertTrue(torch.allclose(lora_layer.lora_a_stacked.cpu(), lora_a))
        self.assertTrue(torch.allclose(lora_layer.lora_b_stacked.cpu(), lora_b))

    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_parallel_linear_with_lora_set_lora(self, mock_get_parallel_info_manager):
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeColumnParallelLinear(["linear"], self.parallel_info)
        n, k = sum(linear_layer.output_partition_sizes), linear_layer.input_size_per_partition
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_layer.device = self.device
        dim_r = math.ceil(self.max_lora_rank / 16) * 16 if self.soc_info.need_nz \
            else math.ceil(self.max_lora_rank / 64) * 64
        lora_layer.lora_a_stacked.data = torch.zeros(self.max_loras + 1, dim_r, k, dtype=self.dtype, device=self.device)
        lora_layer.lora_b_stacked.data = torch.zeros(self.max_loras + 1, dim_r, n, dtype=self.dtype, device=self.device)
        lora_a = torch.rand((self.r, k), device=self.device, dtype=torch.float16)
        lora_b = torch.rand((self.r, n), device=self.device, dtype=torch.float16)
        index = random.randint(0, self.max_loras - 1)
        lora_layer.set_lora(index, lora_a, lora_b)
        self.assertTrue(torch.allclose(lora_layer.lora_a_stacked[index, :self.r].cpu(), lora_a.cpu()))
        self.assertTrue(torch.allclose(lora_layer.lora_b_stacked[index, :self.r].cpu(), lora_b.cpu()))

    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_column_parallel_linear_with_lora_load_lora(self, mock_get_parallel_info_manager):
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeColumnParallelLinear(["linear"], self.parallel_info)
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_tensors_dic = {"linear.lora_A.weight": \
                            torch.rand(self.r, linear_layer.input_size, device=self.device, dtype=self.dtype),
                            "linear.lora_B.weight": \
                            torch.rand(linear_layer.output_size, self.r, device=self.device, dtype=self.dtype)}
        lora_a = lora_layer.slice_lora_a(lora_tensors_dic, ["linear.lora_A"])
        self.assertTrue(torch.allclose(lora_a, lora_tensors_dic["linear.lora_A.weight"]))
        lora_b = lora_layer.slice_lora_b(lora_tensors_dic, ["linear.lora_B"], [1])
        start_idx = self.tp_rank * linear_layer.output_partition_sizes[0]
        end_idx = (self.tp_rank + 1) * linear_layer.output_partition_sizes[0]
        self.assertTrue(torch.allclose(lora_b, \
            lora_tensors_dic["linear.lora_B.weight"][start_idx:end_idx, :].T.contiguous()))

    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_merged_column_parallel_linear_with_lora_load_lora(self, mock_get_parallel_info_manager):
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeMergedColumnParallelLinear(["gate", "up"], self.parallel_info)
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_tensors_dic = {"gate.lora_A.weight": \
                            torch.rand(self.r, linear_layer.input_size, device=self.device, dtype=self.dtype),
                            "gate.lora_B.weight": \
                            torch.rand(linear_layer.output_sizes[0], self.r, device=self.device, dtype=self.dtype),
                            "up.lora_A.weight": \
                            torch.rand(self.r, \
                            linear_layer.input_size, device=self.device, dtype=self.dtype),
                            "up.lora_B.weight": \
                            torch.rand(linear_layer.output_sizes[1], self.r, device=self.device, dtype=self.dtype)}
        lora_a = lora_layer.slice_lora_a(lora_tensors_dic, ["gate.lora_A", "up.lora_A"])
        self.assertTrue(torch.allclose(lora_a, torch.cat([lora_tensors_dic["gate.lora_A.weight"], \
            lora_tensors_dic["up.lora_A.weight"]])))
        lora_b = lora_layer.slice_lora_b(lora_tensors_dic, ["gate.lora_B", "up.lora_B"], [1, 1])
        gate_start_idx = self.tp_rank * linear_layer.output_partition_sizes[0]
        gate_end_idx = (self.tp_rank + 1) * linear_layer.output_partition_sizes[0]
        up_start_idx = self.tp_rank * linear_layer.output_partition_sizes[1]
        up_end_idx = (self.tp_rank + 1) * linear_layer.output_partition_sizes[1]
        self.assertTrue(torch.allclose(lora_b, torch.block_diag(
            lora_tensors_dic["gate.lora_B.weight"][gate_start_idx:gate_end_idx, :],
            lora_tensors_dic["up.lora_B.weight"][up_start_idx:up_end_idx, :]).T.contiguous()))

    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_qkv_column_parallel_linear_with_lora_load_lora(self, mock_get_parallel_info_manager):
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeQKVParallelLinear(["q", "k", "v"], self.parallel_info)
        lora_layer = ColumnParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_tensors_dic = {"q.lora_A.weight": \
                            torch.rand(self.r, linear_layer.input_size, device=self.device, dtype=self.dtype),
                            "q.lora_B.weight": \
                            torch.rand(linear_layer.num_heads * self.world_size * linear_layer.head_size, \
                            self.r, device=self.device, dtype=self.dtype),
                            "k.lora_A.weight": \
                            torch.rand(self.r, linear_layer.input_size, device=self.device, dtype=self.dtype),
                            "k.lora_B.weight": \
                            torch.rand(linear_layer.num_kv_heads * self.world_size * linear_layer.head_size, \
                            self.r, device=self.device, dtype=self.dtype),
                            "v.lora_A.weight": 
                            torch.rand(self.r, linear_layer.input_size, device=self.device, dtype=self.dtype),
                            "v.lora_B.weight": \
                            torch.rand(linear_layer.num_kv_heads * self.world_size * linear_layer.head_size, \
                            self.r, device=self.device, dtype=self.dtype)}
        lora_a = lora_layer.slice_lora_a(lora_tensors_dic, ["q.lora_A", "k.lora_A", "v.lora_A"])
        self.assertTrue(torch.allclose(lora_a, torch.cat([lora_tensors_dic["q.lora_A.weight"], \
            lora_tensors_dic["k.lora_A.weight"], lora_tensors_dic["v.lora_A.weight"]])))
        lora_b = lora_layer.slice_lora_b(lora_tensors_dic, ["q.lora_B", "k.lora_B", "v.lora_B"], [1, 1, 1])
        q_start_idx = self.tp_rank * linear_layer.output_partition_sizes[0]
        q_end_idx = (self.tp_rank + 1) * linear_layer.output_partition_sizes[0]
        k_start_idx = self.tp_rank * linear_layer.output_partition_sizes[1]
        k_end_idx = (self.tp_rank + 1) * linear_layer.output_partition_sizes[1]
        v_start_idx = self.tp_rank * linear_layer.output_partition_sizes[2]
        v_end_idx = (self.tp_rank + 1) * linear_layer.output_partition_sizes[2]
        self.assertTrue(torch.allclose(lora_b, torch.block_diag(
            lora_tensors_dic["q.lora_B.weight"][q_start_idx:q_end_idx, :],
            lora_tensors_dic["k.lora_B.weight"][k_start_idx:k_end_idx, :],
            lora_tensors_dic["v.lora_B.weight"][v_start_idx:v_end_idx, :],).T.contiguous()))

    @patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager")
    def test_row_parallel_linear_with_lora_load_lora(self, mock_get_parallel_info_manager):
        mock_get_parallel_info_manager.return_value = self.mock_parallel_info_manager
        linear_layer = FakeRowParallelLinear(["linear"], self.parallel_info)
        lora_layer = RowParallelLinearWithLoRA(linear_layer)
        lora_layer.dtype = self.dtype
        lora_tensors_dic = {"linear.lora_A.weight": \
                            torch.rand(self.r, linear_layer.input_size, device=self.device, dtype=self.dtype),
                            "linear.lora_B.weight": \
                            torch.rand(linear_layer.output_size, self.r, device=self.device, dtype=self.dtype)}
        lora_a = lora_layer.slice_lora_a(lora_tensors_dic, ["linear.lora_A"])
        start_idx = self.tp_rank * linear_layer.input_size_per_partition
        end_idx = (self.tp_rank + 1) * linear_layer.input_size_per_partition
        self.assertTrue(torch.allclose(lora_a, lora_tensors_dic["linear.lora_A.weight"][:, start_idx:end_idx]))
        lora_b = lora_layer.slice_lora_b(lora_tensors_dic, ["linear.lora_B"], [1])
        self.assertTrue(torch.allclose(lora_b, lora_tensors_dic["linear.lora_B.weight"].T.contiguous()))


if __name__ == '__main__':
    unittest.main()