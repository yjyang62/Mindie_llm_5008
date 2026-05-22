# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import os
import re

from dataclasses import dataclass, field
import ipaddress
import torch_npu

from . import file_utils

DEVICE = "device"


@dataclass
class EnvVar:
    """
    Environment Variables
    """
    # Size of dynamically allocated memory pool during model runtime (unit: GB)
    reserved_memory_gb: int = int(os.getenv("RESERVED_MEMORY_GB", "3"))
    # Which devices to use
    visible_devices: str = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
    # Whether to bind CPU cores
    bind_cpu: bool = os.getenv("BIND_CPU", "1") == "1"
    # Whether to remove post-processing parameters after generation
    remove_generation_config_dict: bool = os.getenv("REMOVE_GENERATION_CONFIG_DICT", "0") == "1"

    cpu_binding_num: int | None = os.getenv("CPU_BINDING_NUM", None)

    memory_fraction = float(os.getenv("NPU_MEMORY_FRACTION", "1.0"))

    lcoc_enable: bool = os.getenv("ATB_LLM_LCOC_ENABLE", "1") == "1"

    compress_head_enable = os.getenv("ATB_LLM_RAZOR_ATTENTION_ENABLE", "0") == "1"
    compress_head_rope = os.getenv("ATB_LLM_RAZOR_ATTENTION_ROPE", "0") == "1"

    profiling_level = os.getenv("PROFILING_LEVEL", "Level0")
    profiling_enable: bool = os.getenv("ATB_PROFILING_ENABLE", "0") == "1"
    profiling_filepath = os.getenv("PROFILING_FILEPATH", os.path.join(os.getcwd(), "profiling"))

    benchmark_enable: bool = os.getenv("ATB_LLM_BENCHMARK_ENABLE", "0") == "1"
    benchmark_filepath = os.getenv("ATB_LLM_BENCHMARK_FILEPATH", None)

    logits_save_enable: bool = os.getenv("ATB_LLM_LOGITS_SAVE_ENABLE", "0") == "1"
    logits_save_folder = os.getenv("ATB_LLM_LOGITS_SAVE_FOLDER", './')

    token_ids_save_enable: bool = os.getenv("ATB_LLM_TOKEN_IDS_SAVE_ENABLE", "0") == "1"
    token_ids_save_folder = os.getenv("ATB_LLM_TOKEN_IDS_SAVE_FOLDER", './')

    modeltest_dataset_specified = os.getenv("MODELTEST_DATASET_SPECIFIED", None)
    modeltest_pd_split_enable: bool = os.getenv("MODELTEST_PD_SPLIT_ENABLE", "0") == "1"

    hccl_enable = os.getenv("ATB_LLM_HCCL_ENABLE", "0") == "1"

    auto_transpose_enable: bool = os.getenv("ATB_LLM_ENABLE_AUTO_TRANSPOSE", "1") == "1"

    atb_speed_home_path: str = os.getenv("ATB_SPEED_HOME_PATH", None)
    ailbi_mask_enable: bool = os.getenv("IS_ALIBI_MASK_FREE", "0") == "1"
    long_seq_enable: bool = os.getenv("LONG_SEQ_ENABLE", "0") == "1"

    rank: int = int(os.getenv("RANK", "0"))
    local_rank: int = int(os.getenv("LOCAL_RANK", "0"))
    world_size: int = int(os.getenv("WORLD_SIZE", "1"))
    rank_table_file: str = os.getenv("RANK_TABLE_FILE", "")

    enable_mc2 = False
    enable_greedy_search_opt = False

    # omni_attention environment
    omni_attention_enable = False
    omni_shift_windows_enable = False
    omni_attention_pattern_file = None

    enable_dp_move_up: bool = os.getenv("DP_MOVE_UP_ENABLE", "0") == "1"
    enable_dp_partition_up: bool = os.getenv("DP_PARTITION_UP_ENABLE", "0") == "1"
    lm_head_local_tp: bool = os.getenv("LM_HEAD_LOCAL_TP", "0") == "1"
    deepseek_mtp: int = int(os.getenv("DEEPSEEK_MTP", "0"))

    enable_expert_hotpot_gather: bool = os.getenv("MINDIE_ENABLE_EXPERT_HOTPOT_GATHER", "0") == "1"
    expert_hotpot_dump_path: str = os.getenv("MINDIE_EXPERT_HOTPOT_DUMP_PATH", None)

    enable_inf_nan_mode: bool = os.getenv("INF_NAN_MODE_ENABLE", "1") == "1"

    master_ip: str = os.getenv("MASTER_IP", None)
    master_port: int | None = os.getenv("MASTER_PORT", None)
    if master_port is not None:
        master_port = int(master_port)

    def __post_init__(self):
        # Validation
        if self.reserved_memory_gb >= 64 or self.reserved_memory_gb < 0:
            raise ValueError("RESERVED_MEMORY_GB should be in the range of 0 to 64, 64 is not inclusive.")

        if self.visible_devices is not None:
            try:
                self.visible_devices = list(map(int, self.visible_devices.split(',')))
            except ValueError as e:
                raise ValueError("ASCEND_RT_VISIBLE_DEVICES should be in format "
                                 "{device_id},{device_id},...,{device_id}") from e

        if self.memory_fraction <= 0 or self.memory_fraction > 1.0:
            raise ValueError("NPU_MEMORY_FRACTION should be in the range of 0 to 1.0, 0.0 is not inclusive.")

        if self.atb_speed_home_path is not None: 
            self.atb_speed_home_path = file_utils.standardize_path(self.atb_speed_home_path)
            file_utils.check_path_permission(self.atb_speed_home_path)

        if self.world_size <= 0 or self.world_size > 1048576:
            raise ValueError("WORLD_SIZE should not be a number in the range of 0 to 1048576, 0 is not inclusive.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("RANK should be in the range of 0 to WORLD_SIZE, WORLD_SIZE is not inclusive.")
        if self.local_rank < 0 or self.local_rank >= self.world_size:
            raise ValueError("LOCAL_RANK should be in the range of 0 to WORLD_SIZE, WORLD_SIZE is not inclusive.")
            
        if self.cpu_binding_num is not None:
            if not isinstance(self.cpu_binding_num, int):
                raise ValueError("CPU_BINDING_NUM should be an int type variable or None")
        
        valid_profiler_level = torch_npu.profiler.ProfilerLevel
        if not hasattr(valid_profiler_level, self.profiling_level):
            raise ValueError("The specified profiling level is not implemented in torch_npu.profiler.ProfilerLevel")
        
        self.profiling_filepath = file_utils.standardize_path(self.profiling_filepath)
        if self.profiling_enable and not os.path.exists(self.profiling_filepath):
            os.makedirs(self.profiling_filepath, mode=0o750, exist_ok=True)
        file_utils.check_file_safety(self.profiling_filepath, 'w')
        if self.expert_hotpot_dump_path is not None:
            self.expert_hotpot_dump_path = file_utils.standardize_path(self.expert_hotpot_dump_path)
            file_utils.check_file_safety(self.expert_hotpot_dump_path, 'w')
        self.check_ranktable(self.rank_table_file)

        if os.getenv("INF_NAN_MODE_ENABLE", "1") == "0":
            os.environ['INF_NAN_MODE_FORCE_DISABLE'] = "1"
        if self.omni_attention_enable:
            self.compress_head_enable = False
            self.compress_head_rope = False
        if self.omni_attention_pattern_file is not None:
            self.omni_attention_pattern_file = file_utils.standardize_path(self.omni_attention_pattern_file)
            file_utils.check_file_safety(self.omni_attention_pattern_file)
        if self.master_ip is not None:
            if not self.is_valid_ip(self.master_ip):
                raise ValueError("Master ip is invalid. Please check environment MASTER_IP.")
        if self.master_port is not None:
            if not 0 <= self.master_port <= 65535:
                raise ValueError("Master port is invalid. Please check environment MASTER_PORT.")

    @staticmethod
    def is_valid_ip(ip):
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def dict(self):
        return self.__dict__

    def update(self):
        self.logits_save_enable = os.getenv("ATB_LLM_LOGITS_SAVE_ENABLE", "0") == "1"
        self.logits_save_folder = os.getenv("ATB_LLM_LOGITS_SAVE_FOLDER", './')
        self.token_ids_save_enable = os.getenv("ATB_LLM_TOKEN_IDS_SAVE_ENABLE", "0") == "1"
        self.token_ids_save_folder = os.getenv("ATB_LLM_TOKEN_IDS_SAVE_FOLDER", './')
        self.modeltest_dataset_specified = os.getenv("MODELTEST_DATASET_SPECIFIED", None)

    def check_ranktable(self, rank_table_file):
        if rank_table_file:
            with file_utils.safe_open(rank_table_file, 'r', encoding='utf-8') as device_file:
                ranktable = json.load(device_file)
            
            world_size = 0
            server_list = ranktable["server_list"]
            for server in server_list:
                server_devices = server[DEVICE]
                world_size += len(server_devices)
            for server in server_list:
                server_devices = server[DEVICE]
                for device in server_devices:
                    if int(device["rank_id"]) < world_size:
                        continue
                    else:
                        raise ValueError("rank_id should be a number less than world size.")
            
            for server in server_list:
                server_devices = server[DEVICE]
                for device in server_devices:
                    if self.is_valid_ip(device["device_ip"]):
                        continue
                    else:
                        raise ValueError("device_ip is invalid.")
            
            for server in server_list:
                if self.is_valid_ip(server["server_id"]):
                    continue
                else:
                    raise ValueError("server_id is invalid.")


ENV = EnvVar()
