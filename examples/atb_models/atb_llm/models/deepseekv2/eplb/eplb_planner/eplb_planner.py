# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
from atb_llm.utils.log import logger
from .policy.policy_factory import PolicyFactory, DynamicConfig
from .eplb_planner_process import EplbPlannerProcess
from .policy.eplb_policy import EplbResult


class EplbPlanner:
    def __init__(self, config: DynamicConfig, policy_type=1, enable_eplb_multi_process=False):
        self.policy = PolicyFactory.generate_policy(policy_type, config)
        self.enable_eplb_multi_process = enable_eplb_multi_process
        if self.enable_eplb_multi_process:
            self.plan_process = EplbPlannerProcess(self.policy)

    @staticmethod
    def parse_ep_file(ep_file_path):
        experts_table = []
        with open(ep_file_path) as handle:
            ep_file = json.load(handle)
            layer_count = ep_file["moe_layer_count"]
            layer_list = ep_file["layer_list"]

            for layer_list_idx in range(layer_count):
                layer_info = layer_list[layer_list_idx]
                device_count = layer_info["device_count"]
                device_list = layer_info["device_list"]
                layer_expert_table = []

                for device_list_idx in range(device_count):
                    device_info = device_list[device_list_idx]
                    device_expert = device_info["device_expert"]
                    layer_expert_table.append(device_expert)

                experts_table.append(layer_expert_table)

        return experts_table

    @staticmethod
    def _do_rebalance_experts2(policy, current_expert_table, expert_workload, q):
        try:
            results = policy.rebalance_experts(current_expert_table, expert_workload)
            q.put((results))
        except Exception as e:
            import traceback
            logger.error(f"Error in _do_rebalance_experts: {str(e)}\n{traceback.format_exc()}")

    def calculate_rebalance_experts(self, load_info, old_map) -> EplbResult:
        results = self.rebalance_experts(old_map, load_info)
        return results

    def rebalance_experts(self, current_expert_table, expert_workload):
        logger.debug("rebalance_experts start")
        if self.enable_eplb_multi_process:
            if self.plan_process.is_alive:
                results = self.plan_process.rebalance_experts(
                    current_expert_table, expert_workload
                )
            else:
                results = EplbResult(change=False)
        else:
            results = self.policy.rebalance_experts(current_expert_table, expert_workload)
        logger.debug(f"rebalance_experts end, changed:{results.change}, sort_layers:{results.priority}")
        return results

    def shutdown(self):
        if self.enable_eplb_multi_process:
            self.plan_process.shutdown()
