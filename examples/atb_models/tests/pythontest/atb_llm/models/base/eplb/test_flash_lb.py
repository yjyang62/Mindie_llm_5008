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
import numpy as np
from atb_llm.utils.moe_utils import random_generation
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.eplb_policy import DynamicConfig
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.flash_lb import FlashLB, local_swap


NUM_MOE_LAYERS = 58
NUM_ROUTED_EXPERTS = 256
NUM_DEVICES = 64
NUM_REAL_EXPERTS = 320


class TestPARunner(unittest.TestCase):
    def setUp(self):
        dynamicconfig = DynamicConfig()
        dynamicconfig.max_stage_window = 16
        self.policy = FlashLB(dynamicconfig)
        self.current_deployment = np.array(random_generation())
        for _ in range(4):
            self.expert_workload = np.random.randint(0, 10000, size=(NUM_MOE_LAYERS,NUM_DEVICES,5))
            self.policy.register_hotness(self.current_deployment, self.expert_workload, NUM_MOE_LAYERS, NUM_ROUTED_EXPERTS)
        self.hotness = np.array(self.policy.hotness_window[0])
        
    def test_register_hotness(self):
        hotness_layer0 = self.expert_workload[0][0]
        hotness_layer0[0] += hotness_layer0[-1]
        self.assertTrue(np.array_equal(hotness_layer0[:-1], self.policy.hotness_window[0][-1][:4]))
        self.assertTrue(len(self.policy.hotness_window[0]) <= 16)
    
    def test_group_based_adaptive_bloating(self):
        stage_weights = self.policy.compute_stage_weight(self.hotness)
        new_deployment, pieces = self.policy.group_based_adaptive_bloating(self.hotness, 320, NUM_DEVICES, stage_weights)
        self.assertTrue(min(pieces) >= 1)
        self.assertEqual(sum(pieces), 320)
        self.assertTrue(set(new_deployment.ravel()).issuperset(range(NUM_ROUTED_EXPERTS)))

    def test_rebalance_layer(self):
        new_deployment, new_par, current_par = self.policy.rebalance_layer(self.current_deployment[0], self.hotness, 0)
        self.assertTrue(new_par <= current_par)
        self.assertTrue(new_par >= 1)

    def test_rebalance_experts(self):
        results = self.policy.rebalance_experts(self.current_deployment, self.expert_workload)
        if results.change:
            self.assertGreater(len(results.priority), 0)
        self.assertTrue(set(results.deployment_table[-1].ravel()).issuperset(range(NUM_ROUTED_EXPERTS)))

    def test_local_swap(self):
        new_mask, repointer_par = local_swap(self.current_deployment[0], self.hotness[-1])
        self.assertEqual(sum(new_mask[0]), len(new_mask[0]) - 1)


if __name__ == '__main__':
    unittest.main()