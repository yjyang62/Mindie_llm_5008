# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import threading
from ..eplb_planner.eplb_forwarder import EplbForwarder
from ..eplb_planner.eplb_planner import EplbPlanner, DynamicConfig
from ..eplb_planner.eplb_thread_handler import do_eplb
from ..eplb_loader.eplb_loader import EplbRebalanceLoader


class EplbWorker:
    def __init__(self, model_runner, rank, model_id, device):
        self.model = model_runner
        self.model.model.warmup_is_end = False
        self.enable_eplb_multi_process = True
        self.eplb_loader = EplbRebalanceLoader(model_runner.model, self.enable_eplb_multi_process)
        self.eplb_forwarder = EplbForwarder(model_runner.model, self.eplb_loader)

        config = DynamicConfig()
        config.buffer_expert_layer_num = 4  # Max number of layers to adjust in a single update
        config.max_stage_window = 128  # Maximum number of historical stages for tracking expert load
        config.threshold_ratio = 1.15  # Fluctuation threshold to trigger expert redistribution
        # Whether to enable pointer-based expert load balancing
        config.enable_pointer_lb = model_runner.model.num_redundant_experts > model_runner.model.mapping.world_size
        # Total number of adjustable MoE layers
        config.num_layers = model_runner.model.num_layers - model_runner.model.config.first_k_dense_replace 
        self.eplb_planner = EplbPlanner(config, policy_type=2, enable_eplb_multi_process=True)

        # 起do eplb线程,收集信息与执行权重更新操作
        t_eplb_load = threading.Thread(
            target=do_eplb,
            args=(
                (self, self.eplb_loader, self.eplb_planner),
                int(rank),
                device,
                self.model.model.init_expert_table
            )
        )
        t_eplb_load.daemon = True
        t_eplb_load.start()