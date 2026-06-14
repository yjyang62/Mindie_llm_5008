# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import queue
import threading
import numpy as np

import torch

from atb_llm.utils.eplb_expert_data_collect import EplbExpertDataCollect
from atb_llm.utils.log import logger


class EplbForwarder:
    def __init__(self, model, eplb_loader, **kwargs):

        self.aggregate_count = 0
        self.load_info_queue = queue.Queue()
        self.planner_block_queue = queue.Queue()

        self.start_reblance_prepare = False
        self.update_flag = False
        self.update_flag_lock = threading.Lock()
        self.block_update_queue = queue.Queue()
        self.current_table = None
        self.new_map = None
        self.mask = None  # Boolean mask specifying the active experts for data routing
        self.priority = None  # List of layer IDs specifying the order of expert weight transfer;
        self._model = model
        self.expert_weight_updator = ExpertWeightUpdator(self, eplb_loader)
        self.npu_synced = False
        self.aggregate_threshold = model.llm_config.models.deepseekv2.eplb.aggregate_threshold
        self.update_times = eplb_loader.h2d_update_times(self._model)

    def fetch_and_sum_load_info(self):
        load_infos = []
        while not self.load_info_queue.empty():
            load_infos.append(self.load_info_queue.get_nowait())
        if len(load_infos) == 0:
            return None
        if len(load_infos) > 1:
            load_info = torch.stack(load_infos).sum(dim=0)
        else:
            load_info = load_infos[0]
        return load_info

    def put_load_info(self, info):
        self.load_info_queue.put_nowait(info)

    def check_aggregate(self):
        if self.aggregate_threshold < 0:
            return False

        self.aggregate_count += 1
        if self.aggregate_count >= self.aggregate_threshold:
            self.aggregate_count = 0
            return True
        return False

    def reset_forward_count(self):
        self.aggregate_count = 0

    def set_update_flag(self, flag):
        with self.update_flag_lock:
            self.update_flag = flag

    def get_update_flag(self):
        tmp = None
        with self.update_flag_lock:
            tmp = self.update_flag
        return tmp

    def do_aggregate(self):
        if not self.npu_synced:
            torch.npu.synchronize()
            self.npu_synced = True
        flash_deepseekv2_model = self._model
        if not flash_deepseekv2_model.warmup_is_end or self.check_aggregate():
            load_info = EplbExpertDataCollect().all_gather_token_num_per_expert(False)
            if EplbExpertDataCollect().is_data_integrity_valid and load_info is not None:
                self.put_load_info(load_info.cpu())
                self.planner_block_queue.put_nowait("start")
                self.start_reblance_prepare = True
                self.expert_weight_updator.expert_update_ready_count = \
                    -self.expert_weight_updator.num_expert_update_ready_countdown

        self.expert_weight_updator.expert_update_ready_count_down()


class ExpertWeightUpdator:
    def __init__(self, eplb_forwarder, eplb_loader):
        self.flash_deepseekv2_model = eplb_forwarder._model
        self.eplb_forwarder = eplb_forwarder
        self.num_expert_update_ready_countdown = self.flash_deepseekv2_model.num_expert_update_ready_countdown
        self.expert_update_ready_count = -1
        self.eplb_loader = eplb_loader
        self.update_times_count = 0
        self.update_flag_true = torch.tensor([1]).npu()
        self.update_flag_false = torch.tensor([0]).npu()
        self.h2d_count = 0
        self.moe_layer_num = self.flash_deepseekv2_model.num_layers - \
            getattr(self.flash_deepseekv2_model, 'first_k_dense_replace', 0)

    def expert_get_prepare_update_flag(self):
        if self.eplb_forwarder.get_update_flag():
            return self.update_flag_true
        else:
            return self.update_flag_false   # ignore when test

    def expert_update_ready_count_down(self):
        if not self.flash_deepseekv2_model.warmup_is_end:
            # warmup时需要阻塞等待planner完成
            prepare_updated_status = self.expert_get_prepare_update_flag()
            result = self.flash_deepseekv2_model.execute_expert_all_gather(prepare_updated_status)[0].cpu()
            all_ready_prod = torch.prod(result.flatten())
            self.eplb_forwarder.set_update_flag(False)
            self.eplb_forwarder.start_reblance_prepare = False
            return
        if not self.eplb_forwarder.start_reblance_prepare:
            return

        self.expert_update_ready_count = (self.expert_update_ready_count + 1) % self.num_expert_update_ready_countdown

        if self.expert_update_ready_count == 0:
            prepare_updated_status = self.expert_get_prepare_update_flag()
            result = self.flash_deepseekv2_model.execute_expert_all_gather(prepare_updated_status)[0].cpu()
            all_ready_prod = torch.prod(result.flatten())
            if all_ready_prod == 1:
                logger.debug(f"h2d took count:{self.h2d_count}")
                if self.eplb_forwarder.priority is not None:
                    self.moe_layer_num = len(self.eplb_forwarder.priority)
                start_layer = self.flash_deepseekv2_model.buffer_expert_layer_num * self.h2d_count
                end_layer = min((self.h2d_count + 1) * self.flash_deepseekv2_model.buffer_expert_layer_num,
                                self.moe_layer_num)
                self.h2d_count = self.h2d_count + 1
                logger.debug(f"weight_memory_copy, start_layer:{start_layer}, end_layer:{end_layer}")
                self.eplb_loader.weight_memory_copy(start_layer, end_layer)

                # 4. 更新old_map， 在H2D的时候异步做了专家分布表创建和设置
                if self.eplb_forwarder.mask is None:
                    expert_routing_maps = (
                        self.build_experts_map_acl_input(self.eplb_forwarder.new_map, start_layer, end_layer))
                else:
                    expert_routing_maps = (
                        self.build_experts_map_with_mask_local_first_acl_input(
                            self.eplb_forwarder.new_map[self.eplb_forwarder.priority],
                            self.eplb_forwarder.mask[self.eplb_forwarder.priority],
                            start_layer,
                            end_layer)
                        )
                logger.debug("expert_routing_map update, range :"
                             f"[{expert_routing_maps.min()}, {expert_routing_maps.max()}]")
                selected_layers = self.eplb_forwarder.priority[start_layer:end_layer]
                self.flash_deepseekv2_model.expert_routing_map[selected_layers] = expert_routing_maps

                EplbExpertDataCollect().is_data_integrity_valid = False
                self.eplb_forwarder.set_update_flag(False)
                self.update_times_count = self.update_times_count + 1
                self.eplb_forwarder.block_update_queue.put("go on")
                if self.update_times_count == self.eplb_forwarder.update_times:
                    # 做完专家权重和分布表更新，重置标志
                    self.eplb_forwarder.start_reblance_prepare = False
                    self.update_times_count = 0
                    self.expert_update_ready_count = -1
                    self.h2d_count = 0
                    self.eplb_forwarder.load_info_queue.queue.clear()
                    self.eplb_forwarder.reset_forward_count()
                    EplbExpertDataCollect().reset_expert_data()
                    EplbExpertDataCollect().is_data_integrity_valid = True

    def build_experts_map_acl_input(self, new_expert_table, start_layer, end_layer):
        update_layer_num = end_layer - start_layer
        new_expert_table = new_expert_table[self.eplb_forwarder.priority[start_layer: end_layer]]
        expert_routing_maps_dict = {}
        expert_routing_maps = [None] * (update_layer_num)
        for layer_id in range(update_layer_num):
            expert_routing_map = {}
            for i, v in enumerate(list(np.array(new_expert_table[layer_id]).flatten())):
                if v not in expert_routing_map:
                    expert_routing_map[v] = [i]
                else:
                    expert_routing_map[v].append(i)

            for key in expert_routing_map.keys():
                num_of_duplications = len(expert_routing_map[key])
                expert_routing_map[key] = expert_routing_map[key][
                    self.flash_deepseekv2_model.mapping.moe_ep.rank % num_of_duplications
                    ]

            expert_routing_map = torch.scatter(torch.zeros(len(expert_routing_map.keys()), dtype=torch.int32),
                                               0,
                                               torch.tensor(list(expert_routing_map.keys()), dtype=torch.int64),
                                               torch.tensor(list(expert_routing_map.values()), dtype=torch.int32)
                                               )
            expert_routing_maps_dict[layer_id] = expert_routing_map

        for i in expert_routing_maps_dict.keys():
            expert_routing_maps[i] = torch.tensor(expert_routing_maps_dict[i], dtype=torch.int32).unsqueeze(0)

        expert_routing_maps = torch.cat(expert_routing_maps, dim=0).npu()
        return expert_routing_maps
    
    # Prefer local (intra-NPU) expert routing, then use round-robin for cross-node routing
    def build_experts_map_with_mask_local_first_acl_input(self, new_expert_table, mask, start_layer, end_layer):
        rank = self.flash_deepseekv2_model.mapping.moe_ep.rank
        num_experts = np.unique(new_expert_table[0].reshape(-1)).shape[0]

        block = np.array(new_expert_table[start_layer:end_layer])
        block_mask = np.array(mask[start_layer:end_layer])
        num_layer, _, num_local_expert = block.shape

        expert_maps = np.zeros((num_layer, num_experts), dtype=np.int32)

        for layer in range(num_layer):
            tbl = block[layer].ravel()
            tbl_mask = block_mask[layer].ravel()

            order = np.argsort(tbl)
            sorted_vals = tbl[order]
            sorted_mask = tbl_mask[order]
            counts = np.bincount(sorted_vals[sorted_mask == 1], minlength=num_experts)
            cum_counts = np.concatenate([[0], np.cumsum(counts)])

            # === 新增部分：优先选用本地 rank 的位置（按 expert） ===
            local_tbl = block[layer, rank, :]
            local_mask = block_mask[layer, rank, :]

            # 构造 local picks：如果本地 expert 出现且有效，就记录它的位置索引
            local_picks = -np.ones(num_experts, dtype=np.int32)

            for j in range(num_local_expert):
                expert_id = local_tbl[j]
                if local_mask[j] == 1:
                    local_picks[expert_id] = rank * num_local_expert + j

            # 计算 relative_rank，避免使用 for 循环
            relative_rank = np.full(num_experts, rank)
            relative_rank -= np.bincount(block[layer, :rank, :].reshape(-1), minlength=num_experts)

            # fallback 使用原先的 pick 策略
            nonzero = counts > 0
            offsets = np.zeros_like(counts)
            offsets[nonzero] = relative_rank[nonzero] % counts[nonzero]
            picks_fallback = cum_counts[:-1] + offsets
            picks_final = np.where(local_picks != -1, local_picks, order[sorted_mask == 1][picks_fallback])

            expert_maps[layer] = picks_final

        return torch.from_numpy(expert_maps).to(torch.int32).npu()
