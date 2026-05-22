# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import csv
import json
import math
import os
from enum import Enum

import numpy as np

from atb_llm.utils.env import ENV
from atb_llm.utils import file_utils
from atb_llm.utils.prof.profiler import is_profiler_enable, prof_expert_hot
from .log import logger


class EPLBType(int, Enum):
    NO_EPLB = 0
    STATIC_EPLB = 1
    DYNAMIC_EPLB = 2
    FORCE_EPLB = 3  # no accuracy guarantee in the forced load balance case


class ExpertParallelDegree(int, Enum):
    NO_EP = 0
    STATIC_EP = 1
    DYNAMIC_EP = 2
    MIX_EP = 3


def assign(expert_count, world_size):
    per_device = math.ceil(expert_count / world_size)
    assignment = []
    if expert_count % world_size == 0:
        for i in range(world_size):
            assignment.append([i * per_device + j for j in range(per_device)])
    else:
        for i in range(world_size - 1):
            assignment.append([i * per_device + j for j in range(per_device)])
        assignment.append([])
        for i in range(expert_count % world_size):
            assignment[-1].append(per_device * (world_size - 1) + i)
    return assignment


def is_ep_file_valid(ep_file_path, n_device=1, layer_id=0):
    if (not os.path.exists(ep_file_path)):
        logger.error(f"EPLB file path '{ep_file_path}' does not exist.")
        return False
    flag_current_layer_exist = False
    with file_utils.safe_open(ep_file_path) as handle:
        ep_file = json.load(handle)
        if ("moe_layer_count" not in ep_file or "layer_list" not in ep_file):
            logger.error("Required fields 'moe_layer_count' or 'layer_list' are missing.")
            return False

        layer_count = ep_file["moe_layer_count"]
        layer_list = ep_file["layer_list"]
        if (len(layer_list) != layer_count or layer_count < 1):
            logger.error(
                f"Invalid 'moe_layer_count'. It should be greater than 0. "
                f"Found: {layer_count}, but 'layer_list' has {len(layer_list)} elements."
            )
            return False

        for list_idx in range(layer_count):
            layer_info = layer_list[list_idx]
            if ("layer_id" not in layer_info or "device_count" not in layer_info or
                    "device_list" not in layer_info):
                logger.error(
                    f"Missing required fields "
                    f"'layer_id', 'device_count', or 'device_list' in layer {list_idx}."
                )
                return False
            if layer_id == int(layer_info["layer_id"]):
                flag_current_layer_exist = True
            device_count = layer_info["device_count"]
            device_list = layer_info["device_list"]
            if (len(device_list) != device_count or device_count < 1 or n_device != device_count):
                logger.error(
                    f"Mismatch in device list. "
                    f"Expected {n_device} devices, "
                    f"but found {device_count} devices in layer {list_idx}."
                )
                return False

            for device_list_idx in range(device_count):
                device_info = device_list[device_list_idx]
                if ("device_id" not in device_info or "device_expert" not in device_info):
                    logger.error(
                        f"Missing 'device_id' or 'device_expert' "
                        f"in device {device_list_idx} of layer {list_idx}."
                    )
                    return False
    if not flag_current_layer_exist:
        logger.error(f"Layer {layer_id} not found in the EPLB balance file.")
    return flag_current_layer_exist


def parse_ep_file(ep_file_path):
    experts_table = []
    try: 
        with file_utils.safe_open(ep_file_path) as handle:
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
                
    except Exception as e:
        msg = f"Failed to load or parse EPLB file '{ep_file_path}': {type(e).__name__}: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    return experts_table


def parse_ep_balance_file(ep_file_path, n_device=1, layer_id=0):
    if (is_ep_file_valid(ep_file_path, n_device=n_device, layer_id=layer_id)):
        return parse_ep_file(ep_file_path)
    else:
        logger.error("[moe_util.py] parse_ep_balance_file: ep balance file is invalid.")
        return None


def save_eplb_data(rank, data, pd_type, save_count, is_topk=False):
    if is_profiler_enable() and ENV.enable_expert_hotpot_gather:
        if not is_topk and save_count % 8 == 0:
            data = data.cpu().numpy().tolist()
            prof_expert_hot(data, rank)
        return
    enable_save = ENV.enable_expert_hotpot_gather and ENV.expert_hotpot_dump_path is not None
    if enable_save and save_count % 8 == 0: # 8 means save file when decode 8 times
        if is_topk:
            data = [topk.cpu().numpy().tolist() for topk in data]
        else:
            data = data.cpu().numpy().tolist()
        file_path = os.path.join(ENV.expert_hotpot_dump_path, pd_type)
        os.makedirs(file_path, exist_ok=True)
        file_name = os.path.join(file_path, f'{pd_type}{"_topk" if is_topk else ""}_{rank}.csv')
        with file_utils.safe_open(file_name, "a", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
            os.fsync(file.fileno())
        file_name = os.path.join(file_path, f'{pd_type}{"_topk" if is_topk else ""}_{rank}_bak.csv')
        with file_utils.safe_open(file_name, "a", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
            os.fsync(file.fileno())


def random_generation(n_layer=58, n_expert=256, device_count=64, n_redundant=64, **kwargs):
    def split_and_insert(n, k, m):
        all_experts = np.arange(n)
        groups = np.array_split(all_experts, k)
        for i in range(m):
            j = i % k + 1
            if len(groups[-j]) == 0:
                groups[-j] = np.append(groups[-j], j)
            else:
                groups[-j] = np.append(groups[-j], groups[-j][0])
        return np.concatenate(groups)

    n_dangling = kwargs.get("n_dangling", 0)
    mix_shared_routing = kwargs.get("mix_shared_routing", False)
    experts_table = []
    for _ in range(n_layer):
        random_placement = split_and_insert(n_expert, device_count, n_redundant)
        step = random_placement.shape[0] // device_count
        layer_expert_table = []
        for _ in range(n_dangling):
            layer_expert_table.append([0])
        for j in range(device_count):
            device_expert = random_placement[j * step: (j + 1) * step].tolist()
            if mix_shared_routing:
                device_expert.append(n_expert)
            layer_expert_table.append(device_expert)
        experts_table.append(layer_expert_table)

    return experts_table


def calculate_eplb_param(file_path, n_routed_experts):
    table = parse_ep_file(file_path)[0]
    expert_nums = []
    mix_shared_routing = False
    num_dangling_experts = 0
    for expert_list in table:
        if expert_list == [0]:
            num_dangling_experts += 1
        expert_nums.append(len(expert_list))
        if max(expert_list) >= n_routed_experts:
            mix_shared_routing = True
    if len(expert_nums) == sum(expert_nums):
        num_dangling_experts -= 1
    num_redundant_experts = sum(expert_nums) - num_dangling_experts - n_routed_experts
    return (mix_shared_routing, num_dangling_experts, num_redundant_experts)