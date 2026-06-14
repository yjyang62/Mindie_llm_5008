# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import numpy as np
import torch_npu
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger


def do_eplb(args, rank_id: int, device: int, old_map):
    pa_runner, eplb_loader, eplb_planner = args
    logger.info(f"[rank{rank_id}] device_id[{device}] planner thread start!")
    torch_npu.npu.set_device(device)
    copy_stream = torch_npu.npu.Stream(device)

    flash_deepseekv2_model = pa_runner.model.model
    old_map = np.array(old_map)
    exit_flag = False
    try:
        while True:
            # 1. 通过阻塞队列阻塞，等待forward唤醒
            pa_runner.eplb_forwarder.planner_block_queue.get()
            load_info = pa_runner.eplb_forwarder.fetch_and_sum_load_info()
            if load_info is None or (not flash_deepseekv2_model.warmup_is_end):
                continue

            # 2. 调取planner, 获取专家表
            results = eplb_planner.calculate_rebalance_experts(load_info, old_map)
            
            if results.mask is not None:
                logger.debug("expert_routing_map changed")
                expert_routing_maps = \
                    pa_runner.eplb_forwarder.expert_weight_updator.build_experts_map_with_mask_local_first_acl_input(
                        old_map, 
                        start_layer=0, 
                        end_layer=len(old_map),
                        mask=results.mask
                        )
                flash_deepseekv2_model.expert_routing_map = expert_routing_maps

            # 如果专家分布表没有更新，不继续执行专家权重更新操作
            if results.change == 0:
                continue
            
            new_map, priority = results.deployment_table, results.priority

            pa_runner.eplb_forwarder.new_map = new_map
            pa_runner.eplb_forwarder.priority = priority
            if results.mask is not None:
                mask = np.ones_like(new_map)
                mask[:, :, -1] = 0 # By default, the last expert is not activated.
                pa_runner.eplb_forwarder.mask = mask
            eplb_loader.priority = priority
            
            transpose_new_map = np.transpose(new_map, (1, 0, 2)) # layer, rank, expert -> rank, layer, expert

            # 3. 下传update操作
            update_times = eplb_loader.h2d_update_times(flash_deepseekv2_model)
            pa_runner.eplb_forwarder.update_times = update_times
            for i in range(update_times):
                if not flash_deepseekv2_model.warmup_is_end:
                    pa_runner.eplb_forwarder.set_update_flag(True)
                    break
                logger.debug(f"[rank{ENV.rank}] start round{i} H2D update")
                need_update = True
                try:
                    need_update = eplb_loader.do_load_prepare_h2d(copy_stream, transpose_new_map[rank_id], i)
                except Exception as e:
                    if not flash_deepseekv2_model.warmup_is_end:
                        exit_flag = True
                    raise Exception from e

                if not need_update:
                    continue

                pa_runner.eplb_forwarder.set_update_flag(True)
                # 阻塞等待D2D完成
                pa_runner.eplb_forwarder.block_update_queue.get()
                logger.debug(f"[rank{ENV.rank}] finish round{i} H2D/D2D update")

            # 4. 更新old_map
            if flash_deepseekv2_model.warmup_is_end:
                old_map[priority] = new_map[priority] 
                
    except Exception as e:
        logger.warning(f"[rank{ENV.rank}] eplb planner thread raise exception: {e}")
        import traceback
        traceback.print_exc()
        if exit_flag:
            raise Exception from e
