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
import os
import json
import unittest
from unittest.mock import MagicMock, patch
import subprocess
from atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_planner import EplbPlanner
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.dynamic_ep import DynamicEP
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.eplb_policy import DynamicConfig


class TestEplbPlanner(unittest.TestCase):
    def setUp(self):
        self.ep_file_path = '/home/test/ep_file.json'
        self.expected_experts_table = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]
        self.config = DynamicConfig()
        self.config.num_layer = 2

    def tearDown(self):
        pass

    def test_parse_ep_file(self):
        # 创建目录 /home/test 及其父目录（如果它们不存在的话）
        subprocess.run(['mkdir', '-p', '/home/test'])

        # 在 /home/test 目录下创建一个名为 ep_file.json 的空文件
        subprocess.run(['touch', '/home/test/ep_file.json'])
        # 创建一个模拟的 JSON 文件内容
        with open(self.ep_file_path, 'w') as f:
            json.dump({
                "moe_layer_count": 2,
                "layer_list": [
                    {
                        "device_count": 2,
                        "device_list": [
                            {"device_expert": [1, 2]},
                            {"device_expert": [3, 4]}
                        ]
                    },
                    {
                        "device_count": 2,
                        "device_list": [
                            {"device_expert": [5, 6]},
                            {"device_expert": [7, 8]}
                        ]
                    }
                ]
            }, f)

        result = EplbPlanner.parse_ep_file(self.ep_file_path)
        self.assertEqual(result, self.expected_experts_table)
        os.remove(self.ep_file_path)

    @patch('atb_llm.models.deepseekv2.eplb.eplb_planner.policy.policy_factory.PolicyFactory.generate_policy', return_value=MagicMock())
    def test_eplb_planner_init(self, mock_generate_policy):
        policy_type = 1
        # 初始化 EplbPlanner 实例
        planner = EplbPlanner(self.config, policy_type, False)
        assert planner.policy == mock_generate_policy.return_value
        planner.shutdown()

    def test_rebalance_experts(self):
        policy_type = 1
        planner = EplbPlanner(self.config, policy_type, False)

        current_expert_table = [[[1, 2],
                                 [3, 4]],

                                [[1, 2],
                                 [3, 4]]]
        expert_workload = [[[1, 2],
                            [3, 4]],

                           [[1, 2],
                            [3, 4]]]
        results = planner.rebalance_experts(current_expert_table, expert_workload)
        changed, sort_layers, new_map = results.change, results.priority, results.deployment_table
        self.assertIsNotNone(changed)
        self.assertIsNotNone(sort_layers)
        self.assertIsNotNone(new_map)
        planner.shutdown()

    def test_calculate_rebalance_experts(self):
        init_expert_table = [[[1, 2],
                              [3, 4]],

                             [[1, 2],
                              [3, 4]]]
        policy_type = 1
        planner = EplbPlanner(self.config, policy_type, True)

        expert_workload = [[[1, 2],
                            [3, 4]],

                           [[1, 2],
                            [3, 4]]]
        results = planner.calculate_rebalance_experts(expert_workload, init_expert_table)
        changed, sort_layers, new_map = results.change, results.priority, results.deployment_table
        self.assertIsNotNone(changed)
        self.assertIsNotNone(sort_layers)
        self.assertIsNotNone(new_map)
        planner.shutdown()

    @patch("atb_llm.models.deepseekv2.eplb.eplb_planner.policy.eplb_policy.EplbPolicy", return_value=MagicMock())
    def test_do_rebalance_experts2(self, mock_policy):
        policy = mock_policy.return_value
        dynamic_ep_policy = DynamicEP(policy)
        current_expert_table = [[[1, 2],
                                 [3, 4]],

                                [[1, 2],
                                 [3, 4]]]
        expert_workload = [[[1, 2],
                            [3, 4]],

                           [[1, 2],
                            [3, 4]]]
        q = queue.Queue()
        EplbPlanner._do_rebalance_experts2(dynamic_ep_policy, current_expert_table, expert_workload, q)


if __name__ == '__main__':
    unittest.main()
