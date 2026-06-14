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
from unittest.mock import MagicMock

import torch
import numpy as np

from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.dynamic_ep import DynamicEP, DynamicTable
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.eplb_policy import EplbPolicy, DynamicConfig
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.mock_load_balance import MockLoadBalance
from atb_llm.models.deepseekv2.eplb.eplb_planner.policy.policy_factory import PolicyFactory


class TestEplbPlannerUtils(unittest.TestCase):
    def setUp(self):
        self.config = DynamicConfig()
        self.config.num_layer = 2
        info = DynamicTable()
        info.workload_table = np.array([[[1, 2],
                                         [3, 4]],

                                        [[1, 2],
                                         [3, 4]]])
        info.placement_table = np.array([[[1, 2],
                                          [3, 4]],

                                         [[1, 2],
                                          [3, 4]]])
        self.info = info
        self.dynamic_ep = DynamicEP(self.config)

    def tearDown(self):
        self.config = None
        self.info = None

    def test_calculate_max_heat_per_layer(self):
        info = self.info
        layer_num = 2
        max_heat_per_layer = DynamicEP.calculate_max_heat_per_layer(info.workload_table, layer_num)
        assert max_heat_per_layer == [7, 7]

    def test_add_redundant(self):
        info = self.info
        layer_workloads = DynamicEP.add_redundant(info.placement_table, info.workload_table, 4)
        assert layer_workloads.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]

    def test_get_redundant_num(self):
        info = self.info
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        expert_ids, counts = np.unique(info.placement_table[0], return_counts=True)
        num_redundancy_expert = DynamicEP.get_redundant_num(num_npus, counts)
        assert num_redundancy_expert == 0

    def test_original_compute_balanced_pack_redundancy(self):
        info = self.info
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        layer_workloads = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        expert_num = layer_workloads.shape[1]
        num_redundancy_expert = 1
        for layer in range(layer_num):
            # 获取当前层专家ID和对应负载，负载需要进行正则化处理, 每个卡加一个冗余专家
            weights = np.zeros((expert_num,), dtype='object')
            for expert_id, workload_weight in enumerate(layer_workloads[layer]):
                weights[expert_id] = (expert_id, workload_weight)

            # 获取每一层全局计算均衡的放置策略
            result, layer_deployment = self.dynamic_ep.original_compute_balanced_pack_redundancy(
                weights, num_npus, num_redundancy_expert
            )
            assert result == [{'box_index': 1, 'items': [3, 3, 0], 'weight': [1.5, 1.5, 0], 'total_weight': 3.0,
                               'item_count': 3},
                              {'box_index': 2, 'items': [2, 1], 'weight': [2, 1], 'total_weight': 3, 'item_count': 2}]
            assert layer_deployment == [[3, 3, 0], [2, 1]]

    def test_compute_balanced_pack_redundancy(self):
        info = self.info
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        layer_workloads = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        expert_num = layer_workloads.shape[1]
        num_redundancy_expert = 1
        for layer in range(layer_num):
            # 获取当前层专家ID和对应负载，负载需要进行正则化处理, 每个卡加一个冗余专家
            weights = np.zeros((expert_num,), dtype='object')
            for expert_id, workload_weight in enumerate(layer_workloads[layer]):
                weights[expert_id] = (expert_id, workload_weight)

            # 获取每一层全局计算均衡的放置策略
            result, layer_deployment = self.dynamic_ep.compute_balanced_pack_redundancy(
                weights, num_npus, num_redundancy_expert
            )
            assert result == [{'box_index': 1, 'items': [2, 3], 'weight': [2, 1.5], 'total_weight': 3.5,
                               'item_count': 2},
                              {'box_index': 2, 'items': [3, 1, 0], 'weight': [1.5, 1, 0], 'total_weight': 2.5,
                               'item_count': 3}]

            assert layer_deployment == [[2, 3], [3, 1, 0]]

    def test_compute_balanced_pack(self):
        info = self.info
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        layer_workloads = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        expert_num = layer_workloads.shape[1]
        for layer in range(layer_num):
            # 获取当前层专家ID和对应负载，负载需要进行正则化处理, 每个卡加一个冗余专家
            weights = np.zeros((expert_num,), dtype='object')
            for expert_id, workload_weight in enumerate(layer_workloads[layer]):
                weights[expert_id] = (expert_id, workload_weight)

            # 获取每一层全局计算均衡的放置策略
            result, layer_deployment = DynamicEP.compute_balanced_pack(weights, num_npus)
            assert result == [{'box_index': 1, 'items': [3, 0], 'weight': [3, 0], 'total_weight': 3, 'item_count': 2},
                              {'box_index': 2, 'items': [2, 1], 'weight': [2, 1], 'total_weight': 3, 'item_count': 2}]
            assert layer_deployment == [[3, 0], [2, 1]]

    def test_rebalance_experts(self):

        current_expert_table = np.array([[[1, 2],
                                          [3, 4]],

                                         [[1, 2],
                                          [3, 4]]])
        expert_workload = np.array([[[1, 2],
                                     [3, 4]],

                                    [[1, 2],
                                     [3, 4]]])

        self.dynamic_ep.rebalance_experts(current_expert_table, expert_workload)

    # EplbPolicy
    def test_eplb_policy_rebalance_experts(self):
        current_expert_table = torch.randint(1, 255, [58, 16, 64])
        expert_workload = torch.randint(1, 255, [58, 16, 64])
        instance0 = EplbPolicy(MagicMock())
        instance0.rebalance_experts(current_expert_table, expert_workload)  # 无返回值

    # MockLoadBalance
    def test_mock_load_balance_rebalance_experts(self):
        current_expert_table = np.array([[[127, 200, 214],
                                          [223, 223, 44]],

                                         [[247, 196, 135],
                                          [79, 126, 222]],

                                         [[148, 145, 72],
                                          [50, 70, 71]]])

        expert_workload = np.array([[[220, 159, 156],
                                     [52, 29, 100]],

                                    [[137, 42, 35],
                                     [96, 124, 18]],

                                    [[136, 35, 132],
                                     [120, 153, 177]]])

        instance0 = MockLoadBalance(MagicMock())
        results = instance0.rebalance_experts(current_expert_table, expert_workload)

        assert results.change == 1
        assert (results.priority == [0, 1, 2]).all()
        assert results.deployment_table.tolist() == [[[127, 200, 44],
                               [223, 223, 214]],
                              [[247, 196, 222],
                               [79, 126, 135]],
                              [[148, 145, 71],
                               [50, 70, 72]]]

    # PolicyFactory
    def test_generate_policys(self):
        instance0 = PolicyFactory()
        instance0.generate_policy(0, MagicMock())


if __name__ == '__main__':
    unittest.main()
