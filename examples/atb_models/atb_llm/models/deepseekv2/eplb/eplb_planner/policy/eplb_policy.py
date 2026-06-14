# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


class DynamicConfig:
    placement_policy = None

    # Maximum number of experts that can be transferred per layer
    max_transferred_expert_per_layer = 100

    ep_worldsize = 64  # Total number of dies across the entire expert-parallel cluster
    num_die_per_host = 8  # Number of dies per machine

    n_group = 8  # Number of expert groups under GroupBasedRouting
    num_layer = 58  # Total number of adjustable MoE layers

    buffer_expert_layer_num = 58  # Max number of layers to adjust in a single update
    enable_pointer_lb = False  # Whether to enable pointer-based expert load balancing
    max_stage_window = 24  # Maximum number of historical stages for tracking expert load
    threshold_ratio = 1.15  # Fluctuation threshold to trigger expert redistribution


@dataclass
class EplbResult:
    change: bool
    priority: Optional[np.ndarray] = None
    deployment_table: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


class EplbPolicy:
    def __init__(self, config: DynamicConfig):
        self.config = config

    @abstractmethod
    def rebalance_experts(self, current_expert_table, expert_workload) -> EplbResult:
        """
        传入weight并返回相关限制条件下的专家复制和放置
        INPUT:
        current_expert_table: [layerId, rankId, expert_num_i]
        expert_workload = expert_table[layer0][rankId][expert_num_i]

        RETURNED: (res, expert_table)
        res:
        1 -- table_changed
        0 -- not_changed

        expert_table: [layerId, rankId, expert_num_i]
        expert_num_i --- [0, MaxExpertPerRank]
        expertID = expert_table[layer0][rankId][expert_num_i]
        array_values:
        [0, 1, 2, 3, 248]
        [4, 5, 6, 7, 254]
        [8, 9, 10, 11, 71]
        ...
        [252, 253, 254, 255, 0]
        """
        pass