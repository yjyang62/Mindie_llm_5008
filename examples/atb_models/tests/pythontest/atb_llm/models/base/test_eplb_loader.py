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
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch_npu

from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader import EplbRebalanceLoader
from atb_llm.utils.quantize.pack_type import PackType


class TestEplbPlannerUtils(unittest.TestCase):
    def setUp(self):
        config_dict = {
            "q_lora_rank": None,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "v_head_dim": 128,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_hidden_layers": 28,
            "rope_scaling": {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 0.707,
                "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
                "parallel_embedding": True
            }
        }
        self.config = DeepseekV2Config.from_dict(config_dict)
        self.config.routed_scaling_factor = 2
        self.config.n_group = 1
        self.config.topk_group = 1
        self.config.topk_method = None
        self.config.parallel_embedding = True

        mock = MagicMock()
        mock.configure_mock(warmup_is_end=False)
        mock.configure_mock(config=self.config)
        self.loader = EplbRebalanceLoader(mock)
        self.loader.layer_group_size = 1
        self.loader.num_moe_layers = 5
        self.loader.priority = np.arange(55)

    def tearDown(self):
        self.loader = None

    def test_h2d_update_times(self):
        mock = MagicMock()
        mock.num_layers = 58
        mock.first_k_dense_replace = 3
        update_times = self.loader.h2d_update_times(mock)
        assert update_times == 55

    @patch("atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader.EplbRebalanceLoader.update",
           return_value=MagicMock())
    def test_do_load_prepare_h2d_no_need_update(self, mock_update):
        copy_stream = torch_npu.npu.Stream(0)
        new_expert_map = torch.Tensor([[22, 22, 33, 33, 10],
                                       [1, 2, 3, 4, 5]] * 29)
        i = 0
        self.loader.layer_group_size = 1
        self.loader.num_moe_layers = 5
        mock_update.return_value = False
        ret = not self.loader.do_load_prepare_h2d(copy_stream, new_expert_map, i)
        assert ret

    @patch("atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader.EplbRebalanceLoader.update",
           return_value=MagicMock())
    def test_do_load_prepare_h2d_need_update(self, mock_update):
        copy_stream = torch_npu.npu.Stream(0)
        new_expert_map = torch.Tensor([[22, 22, 33, 33, 10],
                                       [1, 2, 3, 4, 5]] * 29)
        i = 0
        self.loader.layer_group_size = 1
        self.loader.num_moe_layers = 5
        mock_update.return_value = True
        self.loader.flash_causal_model.warmup_is_end = 1
        ret = self.loader.do_load_prepare_h2d(copy_stream, new_expert_map, i)
        assert ret

    @patch("atb_llm.utils.layers.TensorParallelRowLinear.load_moe", return_value=MagicMock())
    @patch("atb_llm.utils.layers.TensorParallelColumnLinear.load_moe", return_value=MagicMock())
    def test_load_weight_ssd2host(self, mock_column, mock_row):
        update_expert_list = [[22, 22, 33, 33],
                              [1, 2, 3, 4]]
        layer_idx = 1
        self.loader.load_weight_ssd2host(update_expert_list, layer_idx)

    @patch("atb_llm.utils.layers.TensorParallelRowLinear.load_moe", return_value=MagicMock())
    @patch("atb_llm.utils.layers.TensorParallelColumnLinear.load_moe", return_value=MagicMock())
    def test_load_weight_ssd2host1(self, mock_column, mock_row):
        update_expert_list = [[22, 22, 33, 33],
                              [1, 2, 3, 4]]
        layer_idx = 1
        self.loader.pack_type = PackType.ALL_W8A8_DYNAMIC
        self.loader.load_weight_ssd2host(update_expert_list, layer_idx)
        self.loader.enable_eplb_multi_process = True
        setattr(self.loader, "eplb_loader_process", MagicMock())
        self.loader.load_weight_ssd2host(update_expert_list, layer_idx)

    @patch("atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader.EplbRebalanceLoader.load_weight_ssd2host",
           return_value=MagicMock())
    def test_update_no_need_update(self, mock_ssd2host):
        self.loader.buffer_expert_layer_num = 5

        copy_stream = torch_npu.npu.Stream(0)
        new_expert_map = torch.Tensor([[22, 22, 33, 33, 10],
                                       [1, 2, 3, 4, 5]] * 29)
        ret = not self.loader.update(copy_stream, new_expert_map)
        assert ret

    @patch("atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader.EplbRebalanceLoader.load_weight_ssd2host",
           return_value=MagicMock())
    def test_update_need_update(self, mock_ssd2host):
        mock_ssd2host.return_value = [[torch.Tensor(1)]]
        self.loader.buffer_expert_layer_num = 5

        copy_stream = torch_npu.npu.Stream(0)
        new_expert_map = torch.Tensor([[22, 22, 33, 33, 10],
                                       [1, 2, 3, 4, 5]] * 29)
        ret = self.loader.update(copy_stream, new_expert_map)
        assert ret

    def test_weight_memory_copy(self):
        self.loader.weight_memory_copy(start_layer=0, end_layer=1)


if __name__ == '__main__':
    unittest.main()
