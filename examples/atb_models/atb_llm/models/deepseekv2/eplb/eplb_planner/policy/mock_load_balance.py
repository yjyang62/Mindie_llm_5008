# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
import random
import numpy as np

from .eplb_policy import EplbPolicy, DynamicConfig, EplbResult


class MockLoadBalance(EplbPolicy):
    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def rebalance_experts(self, current_expert_table, expert_workload):
        new_table = copy.deepcopy(current_expert_table)
        num_layers = len(current_expert_table)
        num_card = len(current_expert_table[0])

        for i in range(num_layers):
            # 随机选两个卡
            indices = random.sample(range(num_card), 2)

            # 交换冗余专家
            new_table[i][indices[0]][-1], new_table[i][indices[1]][-1] = (
                new_table[i][indices[1]][-1],
                new_table[i][indices[0]][-1]
            )
        results = EplbResult(change=True, priority=np.arange(num_layers), deployment_table=np.array(new_table))
        return results
