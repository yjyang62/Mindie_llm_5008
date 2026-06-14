# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from .eplb_policy import EplbPolicy, DynamicConfig
from .mock_load_balance import MockLoadBalance
from .dynamic_ep import DynamicEP
from .flash_lb import FlashLB


class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: int, config: DynamicConfig) -> EplbPolicy:
        policy = {
            0: MockLoadBalance,
            1: DynamicEP,
            2: FlashLB,
        }
        return policy.get(policy_type, MockLoadBalance)(config)