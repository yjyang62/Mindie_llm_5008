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
from unittest.mock import call, MagicMock

import torch

from atb_llm.layers.moe.fused_experts.fused_experts import FusedExperts
from atb_llm.models.base.config import BaseConfig
from atb_llm.nn.parameter import Parameter
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.utils.mapping import Mapping, ParallelInfo
from atb_llm.utils.quantize.quant_type import LinearTypeV2


class TestFusedExperts(unittest.TestCase):

    def setUp(self):
        self.config = BaseConfig(torch_dtype=torch.float16)
        self.config.n_routed_experts = 2

    def test_fused_experts(self):
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        mock_weight_tool_obj.mapping = MagicMock(spec=Mapping)
        mock_weight_tool_obj.mapping.rank = 0
        mock_weight_tool_obj.mapping.world_size = 2
        mock_weight_tool_obj.mapping.mlp_tp = MagicMock(spec=ParallelInfo)
        mock_weight_tool_obj.mapping.mlp_tp.rank = 0
        mock_weight_tool_obj.mapping.mlp_tp.world_size = 2
        expert_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        mock_weight_tool_obj.get_sharded = MagicMock(return_value=expert_tensor)
        mock_weight_tool_obj.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        module = FusedExperts(self.config, mock_weight_tool_obj, ["expert"], bias=True)
        self.assertIsInstance(module, FusedExperts)
        self.assertIsInstance(module.module.gate_up_weight, Parameter)
        mock_weight_tool_obj.get_sharded.assert_has_calls(
            [
                call("expert.0.gate_proj.weight", dim=0, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.0.up_proj.weight", dim=0, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.1.gate_proj.weight", dim=0, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.1.up_proj.weight", dim=0, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.0.down_proj.weight", dim=1, chunk_id=mock_weight_tool_obj.mapping.rank,
                    num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.1.down_proj.weight", dim=1, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.0.gate_proj.bias", dim=0, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.0.up_proj.bias", dim=0, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.1.gate_proj.bias", dim=0, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.1.up_proj.bias", dim=0, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.0.down_proj.bias", dim=1, chunk_id=mock_weight_tool_obj.mapping.rank,
                    num_chunk=mock_weight_tool_obj.mapping.world_size),
                call("expert.1.down_proj.bias", dim=1, chunk_id=mock_weight_tool_obj.mapping.rank,
                     num_chunk=mock_weight_tool_obj.mapping.world_size),
            ]
        )
        out = module.module(Tensor("sorted_hidden_states"), Tensor("group_list"))
        self.assertIsInstance(out, Tensor)

    def test_fused_experts_without_bias(self):
        weight_tool_cls = MagicMock(spec=SafetensorFileLoader)
        mock_weight_tool_obj = weight_tool_cls()
        mock_weight_tool_obj.mapping = MagicMock(spec=Mapping)
        mock_weight_tool_obj.mapping.rank = 0
        mock_weight_tool_obj.mapping.world_size = 2
        mock_weight_tool_obj.mapping.mlp_tp = MagicMock(spec=ParallelInfo)
        mock_weight_tool_obj.mapping.mlp_tp.rank = 0
        mock_weight_tool_obj.mapping.mlp_tp.world_size = 2
        expert_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        mock_weight_tool_obj.get_sharded = MagicMock(return_value=expert_tensor)
        mock_weight_tool_obj.get_linear_quant_type = MagicMock(return_value=LinearTypeV2.FLOAT16)
        module = FusedExperts(self.config, mock_weight_tool_obj, ["expert"], bias=False)
        self.assertIsInstance(module, FusedExperts)
        self.assertIsInstance(module.module.gate_up_weight, Parameter)
        out = module.module(Tensor("sorted_hidden_states"), Tensor("group_list"))
        self.assertIsInstance(out, Tensor)


if __name__ == '__main__':
    unittest.main()
