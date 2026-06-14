# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from collections import defaultdict, deque
import logging
import numpy as np
from numba import njit
from atb_llm.utils.log import logger
from .eplb_policy import EplbPolicy, DynamicConfig, EplbResult


numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@njit
def compute_piece_counts(hotness, pieces, stage_weights):
    '''
    Calculation experts which should be redundant.
    Inputs: 
    pieces: Number of experts deployed.
    stage_weights: The weights to be considered at different hotness stages.
    Outputs:
    pieces: The number of deployments required for each logic expert. Shape:[Number of logic experts]
    '''
    n_stage, num_experts = hotness.shape
    num_replicas = pieces - num_experts
    pieces = np.ones(num_experts, dtype=np.int32)
    unit = hotness / pieces  # unit[i, j] = hotness[i, j] / pieces[j]

    for _ in range(num_replicas):
        deltas = np.zeros(num_experts, dtype=np.float32)
        for i in range(n_stage):
            # 找 top1 和 top2
            idx1 = -1
            idx2 = -1
            val1 = -1.0
            val2 = -1.0
            for j in range(num_experts):
                v = unit[i, j]
                if v > val1:
                    val2 = val1
                    idx2 = idx1
                    val1 = v
                    idx1 = j
                elif v > val2:
                    val2 = v
                    idx2 = j
            origin = unit[i, idx1]
            secv = unit[i, idx2]
            # Calculate the hotness and benefits after redundancy of the expert
            alt = hotness[i, idx1] / (pieces[idx1] + 1)
            delta = origin - (alt if alt > secv else secv)
            deltas[idx1] += delta * stage_weights[i]

        # Calculate the redundant experts with the highest benefits in different stages
        if np.sum(deltas) == 0:
            deltas = np.sum(unit, axis=0, dtype=np.float32)
        max_idx = np.argmax(deltas)
        pieces[max_idx] += 1
        for i in range(n_stage):
            unit[i, max_idx] = hotness[i, max_idx] / pieces[max_idx]

    return pieces


@njit
def jsq_placement(hotness, pieces, num_device, stage_weights):
    n_stage, num_experts = hotness.shape
    total_piece = pieces.sum()
    num_per_group = total_piece // num_device

    # 1. 计算 unit_hotness
    unit_hotness = np.empty((n_stage, num_experts), dtype=np.float32)
    for i in range(num_experts):
        for s in range(n_stage):
            unit_hotness[s, i] = hotness[s, i] / pieces[i]

    # 2. 按总热度排序
    scores = np.zeros(num_experts, dtype=np.float32)
    for i in range(num_experts):
        for s in range(n_stage):
            scores[i] += unit_hotness[s, i]
    idx = np.argsort(-scores)

    # 3. 初始化
    loads = np.zeros((n_stage, num_device), dtype=np.float32)
    dev_phy_exp_n = np.zeros(num_device, dtype=np.int32)
    deployment = -np.ones((num_device, num_per_group), dtype=np.int32)
    dep_ptr = np.zeros(num_device, dtype=np.int32)

    # 4. 主循环
    for t in range(num_experts):
        i = idx[t]
        for _ in range(pieces[i]):
            # 4.1 构造 w 向量
            w = np.empty(n_stage, dtype=np.float32)
            for s in range(n_stage):
                w[s] = unit_hotness[s, i]

            # 4.2 计算阶段级最大负载 (loads.max(axis=1))
            stage_max = np.empty(n_stage, dtype=np.float32)
            for s in range(n_stage):
                max_val = loads[s, 0]
                for k in range(1, num_device):
                    if loads[s, k] > max_val:
                        max_val = loads[s, k]
                stage_max[s] = max_val

            # 4.3 计算分母 denom_s = mean_j (loads[s,j] + w[s]) + eps
            denom = np.empty(n_stage, dtype=np.float32)
            for s in range(n_stage):
                sum_tmp = 0.0
                for j in range(num_device):
                    sum_tmp += loads[s, j] + w[s]
                denom[s] = sum_tmp / num_device + 1e-2

            # 4.4 找最佳设备 j
            best_j = -1
            best_val = 1e30
            for j in range(num_device):
                # 达到上限跳过
                if dev_phy_exp_n[j] >= num_per_group:
                    continue

                # 计算 numer/denom 并累加加权和
                score = 0.0
                for s in range(n_stage):
                    tmp_sj = loads[s, j] + w[s]
                    numer_sj = tmp_sj if tmp_sj > stage_max[s] else stage_max[s]
                    score += stage_weights[s] * (numer_sj / denom[s])

                if score < best_val:
                    best_val = score
                    best_j = j

            # 4.5 更新 loads、deployment、计数
            for s in range(n_stage):
                loads[s, best_j] += w[s]
            ptr = dep_ptr[best_j]
            deployment[best_j, ptr] = i
            dep_ptr[best_j] += 1
            dev_phy_exp_n[best_j] += 1

    return deployment


@njit
def slice_values(hotness, pieces):
    total_len = 0
    for i in range(hotness.shape[0]):
        total_len += pieces[i]
    result = np.empty(total_len, dtype=np.float32)
    idx = 0
    for i in range(hotness.shape[0]):
        val = hotness[i] / pieces[i]
        for _ in range(pieces[i]):
            result[idx] = val
            idx += 1
    return result


@njit
def group_based_adaptive_bloating_kernel(hotness, pieces, simulated_pieces, simulated_deployment, stage_weights):
    num_device = simulated_deployment.shape[0]
    n_stage, num_experts = hotness.shape
    num_group = pieces // num_device

    hotness_all = np.zeros(num_experts, dtype=np.float32)
    for i in range(n_stage):
        for j in range(num_experts):
            hotness_all[j] += hotness[i, j]

    sort_idx = np.argsort(np.negative(hotness_all))
    hotness_sorted = hotness[:, sort_idx]

    unit_load = np.empty(num_experts, dtype=np.float32)
    for j in range(num_experts):
        unit_load[j] = hotness_all[j] / simulated_pieces[j]

    flat_deployment = simulated_deployment.reshape(-1)
    simulated_load = np.zeros(num_device, dtype=np.float32)
    for i in range(flat_deployment.shape[0]):
        simulated_load[i // (flat_deployment.shape[0] // num_device)] += unit_load[flat_deployment[i]]

    slice_vals = slice_values(hotness_all, simulated_pieces)
    sorted_slices = np.sort(slice_vals)[::-1]
    simulated_slopes = (sorted_slices[:-num_device + 1] - sorted_slices[num_device - 1:]) / num_device

    cumulative_slices_used = np.zeros(num_experts, dtype=np.int32)
    acc = 0
    for i in range(num_experts):
        acc += simulated_pieces[sort_idx[i]]
        cumulative_slices_used[i] = acc

    group_boundary_indices = np.empty(num_group, dtype=np.int32)
    for i in range(1, num_group + 1):
        for j in range(num_experts):
            if cumulative_slices_used[j] >= i * num_device:
                group_boundary_indices[i - 1] = j
                break

    slices_used_per_group = np.empty(num_group, dtype=np.int32)
    slices_used_per_group[0] = group_boundary_indices[0]
    for i in range(1, num_group):
        slices_used_per_group[i] = group_boundary_indices[i] - group_boundary_indices[i - 1]
    slices_used_per_group = num_device - slices_used_per_group

    loads = np.zeros(num_device, dtype=np.float32)
    pieces_per_expert = np.zeros(num_experts, dtype=np.int32)
    num_remain_slice = pieces - num_experts
    current_idx = 0

    for _ in range(num_group):
        window = hotness_sorted[:, current_idx: current_idx + 2 * num_device]
        low = max(0, current_idx + num_device - num_experts)
        high = min(num_remain_slice, num_device - 1)

        while high - low > 1:
            mid = (high + low) // 2
            keep = num_device - mid
            current_group = window[:, :keep]
            current_pieces = compute_piece_counts(current_group, num_device, stage_weights)
            current_slice = slice_values(current_group.sum(0), current_pieces)
            current_slice_sorted = np.sort(current_slice)
            current_loads = loads + current_slice_sorted
            current_slope = (np.max(current_loads) - np.min(current_loads)) / num_device
            next_slope = np.max(simulated_slopes[current_idx + keep:])

            if abs(current_slope) > abs(next_slope):
                low = mid
            else:
                high = mid

        num_replicas = high
        keep = num_device - num_replicas
        current_group = window[:, :keep]
        current_pieces = compute_piece_counts(current_group, num_device, stage_weights)

        for i in range(keep):
            pieces_per_expert[sort_idx[current_idx + i]] = current_pieces[i]

        current_slice = slice_values(current_group.sum(0), current_pieces)
        current_slice_sorted = np.sort(current_slice)
        loads += current_slice_sorted
        loads = np.sort(loads)[::-1]

        current_idx += keep
        num_remain_slice -= num_replicas

    return pieces_per_expert


@njit
def compute_objective(deployment, hotness, pieces_per_expert):
    num_device, pieces = deployment.shape
    loads = np.zeros(num_device)

    for i in range(num_device):
        for j in range(pieces):
            expert = deployment[i, j]
            if pieces_per_expert[expert] == 0:
                continue
            loads[i] += hotness[expert] / pieces_per_expert[expert]

    mean_load = np.mean(loads)
    max_load = np.max(loads)
    obj = max_load / mean_load
    return obj, loads


@njit
def local_swap(deployment, hotness):
    new_deployment = deployment[:, :-1].copy()
    _, pieces = new_deployment.shape

    mask = np.ones_like(deployment)
    mask[:, -1] = 0

    flat = new_deployment.reshape(-1)
    max_expert = np.max(flat) + 1
    pieces_per_expert = np.zeros(max_expert, dtype=np.int32)
    for i in range(flat.shape[0]):
        pieces_per_expert[flat[i]] += 1

    best_obj, loads = compute_objective(new_deployment, hotness, pieces_per_expert)

    indices = np.argsort(loads)

    for i in indices:
        best_delta = 0.0
        best_j = -1
        current_expert = deployment[i, -1]

        for j in range(pieces):
            old_expert = new_deployment[i, j]
            if pieces_per_expert[old_expert] <= 1:
                continue

            new_deployment[i, j] = current_expert
            pieces_per_expert[old_expert] -= 1
            pieces_per_expert[current_expert] += 1

            new_obj, _ = compute_objective(new_deployment, hotness, pieces_per_expert)
            delta = best_obj - new_obj
            if delta > best_delta:
                best_delta = delta
                best_j = j

            new_deployment[i, j] = old_expert
            pieces_per_expert[old_expert] += 1
            pieces_per_expert[current_expert] -= 1

        if best_j != -1:
            old_expert = new_deployment[i, best_j]
            new_deployment[i, best_j] = current_expert
            pieces_per_expert[old_expert] -= 1
            pieces_per_expert[current_expert] += 1
            best_obj -= best_delta

            mask[i, best_j] = 0
            mask[i, -1] = 1

    return mask, best_obj


class FlashLB(EplbPolicy):
    par_history = defaultdict(float)
    hotness_window = {}

    def __init__(self, config: DynamicConfig):
        super().__init__(config)

    def compute_expert_hotness(self, num_of_expert: int, deployment: np.ndarray, rank_load: np.ndarray):
        hotness = np.zeros(num_of_expert, dtype=rank_load.dtype)
        deployment_flat = deployment.ravel()
        rank_load_flat = rank_load.ravel()
        np.add.at(hotness, deployment_flat, rank_load_flat)
        return hotness
    
    def compute_rank_load(self, deployment: np.ndarray, hotness: np.ndarray):
        n_stage, _ = hotness.shape
        unit_hotness = hotness / np.bincount(deployment.reshape(-1))
        stage_par = np.zeros(n_stage)
        for i in range(n_stage):
            stage_load = unit_hotness[i][deployment].sum(-1)
            stage_par[i] = stage_load.max() / stage_load.mean()
        return stage_par.mean()
    
    def group_based_adaptive_bloating(self, hotness, pieces, num_device, stage_weights=None, recorsive=False):
        n_stage, _ = hotness.shape
        if stage_weights is None:
            stage_weights = np.ones(n_stage, dtype=np.float32)

        # Calculate the redundant experts and expert map.
        if recorsive:
            simulated_deployment, simulated_pieces = \
                self.group_based_adaptive_bloating(hotness, pieces, num_device, stage_weights, recorsive=False)
        else:
            simulated_pieces = compute_piece_counts(hotness, pieces, stage_weights)
            simulated_deployment = jsq_placement(hotness, simulated_pieces, num_device, stage_weights)

        
        # Find out if there is a better way to deploy redundant experts based on the calculated expert routing table.
        pieces = group_based_adaptive_bloating_kernel(
            hotness.astype(np.float32),
            pieces,
            simulated_pieces.astype(np.int32),
            simulated_deployment.astype(np.int32),
            stage_weights.astype(np.float32),
        )

        deployment = jsq_placement(hotness, pieces, num_device, stage_weights)

        # Choose which calculation expert table to use based on the degree of imbalance
        hotness_all = hotness.sum(0)
        unit_load = hotness_all / pieces
        load = unit_load[deployment].sum(-1)

        sim_unit_load = hotness_all / simulated_pieces
        sim_load = sim_unit_load[simulated_deployment].sum(-1)

        if load.max() > sim_load.max():
            return simulated_deployment, simulated_pieces
        return deployment, pieces
        

    def need_update(self, current_par, layer_id=0):
        threshold = self.par_history.get(layer_id, 0.0)
        return current_par >= self.config.threshold_ratio * threshold or current_par > 1.3

    def compute_stage_weight(self, hotness):
        # The comprehensive ratio of heat in each stage is used as the weight of the stage.
        n_stage = hotness.shape[0]
        stage_weights = np.zeros(n_stage)
        for i in range(n_stage):
            stage_weights[i] = hotness[i].sum()

        stage_weights = stage_weights / stage_weights.max()
        return stage_weights

    def rebalance_layer(self, deployment, hotness, layer_id=0):
        num_rank, expert_per_rank = deployment.shape
        num_expert = np.unique(deployment.reshape(-1)).shape[0]
        num_of_redundant_expert = num_rank * expert_per_rank - num_expert

        current_par = self.compute_rank_load(deployment, hotness)

        # If the current imbalance does not meet the update threshold,
        # it will directly return to the original expert routing table.
        if not self.need_update(current_par, layer_id):
            return deployment, current_par, current_par
        
        # If an update is needed, calculate the new expert routing table and imbalance
        stage_weights = self.compute_stage_weight(hotness)
        new_deployment, _ = self.group_based_adaptive_bloating(
            hotness,
            num_expert + num_of_redundant_expert,
            num_rank,
            stage_weights,
            recorsive=False)

        new_par = self.compute_rank_load(new_deployment, hotness)
        
        return new_deployment, new_par, current_par
    
    def place_pointer_experts(self, hotness, num_device, deployment):
        middle = np.abs(hotness - np.mean(hotness))
        middle_idx = np.argpartition(middle, num_device)[:num_device]
        return np.concatenate([deployment, middle_idx.reshape((-1, 1))], axis=1)
    
    def register_hotness(self, deployment, rank_load, num_layer, num_expert):
        for layer in range(num_layer):
            if layer not in self.hotness_window:
                self.hotness_window[layer] = deque(maxlen=self.config.max_stage_window)
            hotness = self.compute_expert_hotness(num_expert, deployment[layer], rank_load[layer])
            self.hotness_window[layer].append(hotness)

    def compress_by_avg_pooling_fast_nd(self, arr, m):
        n, d = arr.shape
        idx = (np.arange(n) * m // n)
        result = np.zeros((m, d))
        counts = np.zeros((m, 1))
        np.add.at(result, idx, arr)
        np.add.at(counts, idx, 1)
        return result / counts
    
    def rebalance_experts(self, current_expert_table, expert_workload):
        '''
        Generate a new experts balancing loading table.
        Input:
        current_expert_table: Current experts balancing loading table. 
                            Shape: [moe_layer_num, rank_num, expert_per_rank_num]
        expert_workload: Hotness of every experts.
                        Shape: [moe_layer_num, rank_num, expert_per_rank_num]
        Output:
            EplbResult(
            change: Whether the table changes,
            priority: Layers which need to be changed,
            deployment_table: New experts balancing loading table,
            mask: Experts whick need to be activated
            )

        '''
        current_deployment = np.array(current_expert_table)
        expert_workload = np.array(expert_workload)
        num_layer, _ = expert_workload.shape[0], expert_workload.shape[1]
        num_expert = np.unique(current_expert_table[0].reshape(-1)).shape[0]

        # Convert the hotness of physical experts into hotness of logical experts. 
        # Saving up to 16 collection windows.
        self.register_hotness(current_deployment, expert_workload, num_layer, num_expert)

        new_deployment = current_deployment.copy()
        
        # If you overload experts, enable the ones you really need.
        new_mask = None
        repointer_par = np.zeros(num_layer)
        if self.config.enable_pointer_lb:
            new_mask = np.ones_like(new_deployment)
            for layer in range(num_layer):
                new_mask[layer], repointer_par[layer] = local_swap(
                    current_deployment[layer],
                    np.array(self.hotness_window[layer][-1])
                    )
            repointer_par = repointer_par[:self.config.num_layer]

        layers_need_update = np.arange(min(num_layer, self.config.num_layer))

        # Based on the existing load balancing table and hotness,
        # each layer calculates a new load balancing table, new imbalance, and old imbalance separately.
        new_par = np.zeros(layers_need_update.shape[0])
        current_par = np.zeros(layers_need_update.shape[0])
        for i, layer in enumerate(layers_need_update):
            hotness = np.array(self.hotness_window[layer])
        
            if self.config.enable_pointer_lb:
                deploy, npar, cpar = self.rebalance_layer(
                    current_deployment[layer][:, :-1],
                    hotness=hotness,
                    layer_id=layer,
                    )
                deploy = self.place_pointer_experts(
                    hotness.sum(0),
                    current_deployment[layer].shape[0],
                    deploy
                    )

                new_deployment[layer] = deploy
                new_par[i] = npar
                current_par[i] = cpar
            else:
                new_deployment[layer], new_par[i], current_par[i] = self.rebalance_layer(
                    current_deployment[layer],
                    hotness,
                    layer_id=layer
                    )
        
        # Calculate imbalance and sort. Filter out the top buffer_expert_layer_num layers 
        # where the imbalance is reduced and the optimization is more obvious.
        priority = new_par / current_par
        priority_idx = np.argsort(priority)
        priority_idx = priority_idx[priority[priority_idx] < 1][:self.config.buffer_expert_layer_num]

        logger.info(
            f"new_par: {new_par.mean()}, cur_par: {current_par.mean()}, ptr_par: {repointer_par.mean()}, change: {layers_need_update[priority_idx]}")
        # Record new imbalance.
        change = len(priority_idx) > 0
        if change:
            for idx in priority_idx:
                self.par_history[layers_need_update[idx]] = new_par[idx]
    
        results = EplbResult(
            change=change,
            priority=layers_need_update[priority_idx],
            deployment_table=new_deployment,
            mask=new_mask)

        return results