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

import queue

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader import EplbRebalanceLoader
from atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_forwarder import EplbForwarder, ExpertWeightUpdator


class TestEplbForwarder(unittest.TestCase):
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

        model = MagicMock()
        model.warmup_is_end = False
        model.execute_expert_all_gather.return_value = torch.ones(1).npu()
        model.num_expert_update_ready_countdown = 16
        model.buffer_expert_layer_num = 1
        model.mapping.moe_ep.rank = 0
        self.mock_allgather = patch(("atb_llm.utils.eplb_expert_data_collect."
        "EplbExpertDataCollect.all_gather_token_num_per_expert")).start()
        self.mock_allgather.return_value = torch.zeros(1).npu()
        self.forwarder = EplbForwarder(model, self.loader)
        self.forwarder.priority = np.array([0])
        self.forwarder.new_map = np.arange(16).reshape((2, 2, -1)) % 8
        self.forwarder.mask = np.ones_like(self.forwarder.new_map)
        self.updater = ExpertWeightUpdator(self.forwarder, self.loader)

        self.forwarder.aggregate_threshold = 3

    def tearDown(self):
        self.config = None
        self.info = None
        self.forwarder = None
        self.updater = None

    def test_fetch_and_sum_load_info(self):
        self.forwarder.load_info_queue = queue.Queue()
        self.forwarder.fetch_and_sum_load_info()

    def test_put_load_info(self):
        self.forwarder.load_info_queue = queue.Queue()
        self.forwarder.put_load_info({'layer_id': 1})

    def test_check_aggregate(self):
        self.forwarder.aggregate_threshold = 3
        ret = not self.forwarder.check_aggregate()
        self.assertTrue(ret)

    def test_set_update_flag(self):
        self.forwarder.set_update_flag(True)

    def test_get_update_flag(self):
        ret = not self.forwarder.get_update_flag()
        self.assertTrue(ret)

    def test_do_aggregate(self):
        self.config.eplb_level = 2
        self.forwarder.npu_synced = False
        self.forwarder.do_aggregate()
        self.config.eplb_level = 0

    # ExpertWeightUpdator
    def test_expert_get_prepare_update_flag(self):
        ret = self.updater.expert_get_prepare_update_flag()
        assert ret == 0

    def test_expert_update_ready_count_down_without_reblance_prepare(self):
        self.updater.flash_deepseekv2_model.warmup_is_end = True
        self.updater.eplb_forwarder.start_reblance_prepare = False
        self.updater.expert_update_ready_count_down()

    def test_expert_update_ready_count_down_without_warmup_is_end(self):
        self.updater.flash_deepseekv2_model.warmup_is_end = False
        self.updater.flash_deepseekv2_model.execute_expert_all_gather.return_value = torch.tensor([[[1, 1]]])
        self.updater.expert_update_ready_count_down()

    def test_expert_update_ready_count_down(self):
        self.updater.flash_deepseekv2_model.warmup_is_end = True
        self.updater.eplb_forwarder.start_reblance_prepare = True
        self.updater.expert_update_ready_count_down()

    def test_build_experts_map_acl_input(self):
        self.updater.flash_deepseekv2_model.num_layers = 3
        self.updater.flash_deepseekv2_model.first_k_dense_replace = 1
        self.updater.flash_deepseekv2_model.mapping.moe_ep.rank = 0
        self.updater.eplb_forwarder.priority = np.arange(3)

        new_expert_table = np.array([[[0, 1],
                                      [2, 3]],

                                     [[0, 1],
                                      [2, 3]]])
        ret1, ret2 = self.updater.build_experts_map_acl_input(new_expert_table, start_layer=0, end_layer=2)
        ret_list1 = [t.tolist() for t in ret1]
        ret_list2 = [t.tolist() for t in ret2]
        assert ret_list1,ret_list2 == [[[0, 1, 1, 1, 1]], [[[-1, 0, 1, 2, 3]]]]

    def test_reset_forward_count(self):
        self.forwarder.reset_forward_count()
        assert self.forwarder.aggregate_count == 0


if __name__ == '__main__':
    unittest.main()
