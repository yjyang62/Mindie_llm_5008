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

from atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_planner_process import EplbPlannerProcess
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.policy_factory import DynamicConfig
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.mock_load_balance import MockLoadBalance


class TestEplbPlannerProcess(unittest.TestCase):

    def setUp(self):
        policy = MockLoadBalance(DynamicConfig())
        self.process = EplbPlannerProcess(policy)

    def test_rebalance_experts(self):
        new_expert_table = np.array([[[1, 2],
                                      [3, 4]],

                                     [[1, 2],
                                      [3, 4]]])
        results = self.process.rebalance_experts(new_expert_table, [])
        self.process.shutdown()
        assert results.change == 1
        assert not self.process.is_alive
        
if __name__ == '__main__':
    unittest.main()
