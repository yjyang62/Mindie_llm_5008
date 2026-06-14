#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import acl
from dataclasses import dataclass, field
from itertools import accumulate
from atb_llm.utils.layerwise_disaggregated.cloud_cut_policy import CloudCutModelType


@dataclass
class RatioInfo:
    ratio_list: list[int] = field(default_factory=list)
    min_unit: int = 128


class ChunkPrefilPolicy:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ChunkPrefilPolicy, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name_or_path="qwen", batch_p_num=1, moe_quantize=None):
        self.soc_name = acl.get_soc_name()
        if not hasattr(self, "initialized"):
            self.model_type = self.__get_model_name(model_name_or_path)
            self.batch_p_num = batch_p_num
            self.multi_nodes_enable = False
            self.moe_quantize = moe_quantize
            self.cp_size = 1
            # the map means to {prefill len(K) : ratio list for chunk ([chunk_ratio] * chunk num)}
            self.ratio_list_default_edge_map = {
                125: [1] * 33,
                63: [1] * 20,
                31: [1] * 10,
                15: [1] * 6,
                7: [1] * 2,
            }
            self.__ajust_prefill_chunk_map_for_diff_npu_soc_qwen()
            if self.model_type == CloudCutModelType.DEEP_SEEK:
                self.ratio_list_default_edge_map = {
                    31: [1] * 20,
                    15: [1] * 5,
                    7: [1] * 2,
                }
            # 当前认为边云侧的chunk ratio list是一致的; 其实边云可以不一样长, 但是必须保持云的每一段是边段的整数倍长
            # 如云: [10, 6], 边: [6, 4, 2, 2, 2], 也就是云10 = 边6+4; 云6 = 边2+2+2; 边/云的总数也得相等
            # 也就是云的第一段等于边的前两段之和, 云的第二段等于边段的后三段之和
            self.ratio_list_default_cloud_map = self.ratio_list_default_edge_map

    @staticmethod
    def __get_model_name(model_name_or_path):
        model_name_or_path_ = model_name_or_path.lower()
        if "qwen" in model_name_or_path_:
            return CloudCutModelType.QWEN
        elif "deepseek" in model_name_or_path_ or "ds" in model_name_or_path_:
            return CloudCutModelType.DEEP_SEEK
        return CloudCutModelType.QWEN

    @staticmethod
    def get_cloud_eq_edge_chunk_num_list(edge_list: list[int], cloud_list: list[int]):
        # 找到云侧一段chunk等于边侧多少段chunk数的list
        eq_chunk_num_list = []
        tmp_edge_list = [0] + list(accumulate(edge_list))
        tmp_cloud_list = [0] + list(accumulate(cloud_list))

        cloud_index = 1
        last_chunk_end_index = 0
        for cloud_end_offset in tmp_cloud_list[cloud_index:]:
            for i in range(last_chunk_end_index, len(tmp_edge_list)):
                end_offset = tmp_edge_list[i]
                if end_offset >= cloud_end_offset:
                    eq_chunk_num_list.append(i - last_chunk_end_index)
                    last_chunk_end_index = i
                    break

        return eq_chunk_num_list

    @staticmethod
    def get_edge_chunk_length_by_ratio(total_len, ratio_list, unit):
        """按照边侧每个段的比例, 计算每个段的长度, 单位为unit; total_len是云每段的长度, ratio_list是边侧每个段的比例"""
        if total_len % unit != 0:
            raise ValueError(f"总长度{total_len}不能被单位{unit}整除")

        total_ratio = sum(ratio_list)
        total_units = total_len // unit
        allocated_units = []
        remaining_units = total_units

        for ratio in ratio_list:
            raw_units = (total_units * ratio) / total_ratio
            base_units = int(raw_units)

            allocated_units.append(base_units)
            remaining_units -= base_units

        for i in range(remaining_units):
            allocated_units[i] += 1

        allocation = [u * unit for u in allocated_units]
        return allocation

    @staticmethod
    def split_long_seq_by_ratio(total_len, edge_info: RatioInfo, cloud_info: RatioInfo):
        # 1. 计算比例总和, 检查有效性
        edge_ratio_sum = sum(edge_info.ratio_list)
        cloud_ratio_sum = sum(cloud_info.ratio_list)
        if edge_ratio_sum != cloud_ratio_sum or edge_ratio_sum == 0:
            raise ValueError(f"边侧和云侧的总比例和不一致, 边侧{edge_ratio_sum}, 云侧{cloud_ratio_sum}")

        edge_valid = (total_len / edge_info.min_unit) >= len(edge_info.ratio_list)
        cloud_valid = (total_len / cloud_info.min_unit) >= len(cloud_info.ratio_list)
        if not edge_valid or not cloud_valid:
            raise ValueError(f"比例总和不能为0, 至少要有{len(edge_info.ratio_list)}个单位")

        # 2. 按比例计算每份基础长度（非整数）
        cloud_base_lengths = [total_len * r / cloud_ratio_sum for r in cloud_info.ratio_list]
        cloud_split_lengths = []

        # 3. 调整为最小单位的整数倍（向下取min_unit）
        for cloud_bl in cloud_base_lengths:
            # 向下取到最近的min_unit倍数
            rounded = (int(cloud_bl) // cloud_info.min_unit) * cloud_info.min_unit
            # 确保至少为最小单位（避免0）
            cloud_split_lengths.append(max(rounded, cloud_info.min_unit))

        # 4. 调整云侧每份长度，先按比例分配, 将min_unit整数倍全部分完
        current_sum = sum(cloud_split_lengths)
        remaining = total_len - current_sum
        remain_unit_num = remaining // cloud_info.min_unit
        for i in range(len(cloud_split_lengths)):
            if remain_unit_num > 0 and cloud_split_lengths[i] < cloud_base_lengths[i]:
                cloud_split_lengths[i] += cloud_info.min_unit
                remain_unit_num -= 1

            if remain_unit_num == 0:
                break

        # 5. 云侧的剩余长度要比边侧的长, 因此仅留下边的非单位长度不分; 云侧的剩余长度要补到最后一份
        cloud_remaining = remaining % cloud_info.min_unit
        edge_remaining = remaining % edge_info.min_unit
        cloud_split_lengths[-1] = cloud_split_lengths[-1] + cloud_remaining - edge_remaining

        # 6. 计算边侧的切分策略
        eq_chunk_num_list = ChunkPrefilPolicy.get_cloud_eq_edge_chunk_num_list(
            edge_info.ratio_list, cloud_info.ratio_list
        )

        # 7. 调整策略, 如果出现云侧长度没有边侧N个chunk * unit那么多, 则需要调整到后面
        eq_list_len = len(eq_chunk_num_list)
        for i in range(eq_list_len):
            eq_edge_unit_num = cloud_split_lengths[i] // edge_info.min_unit
            if eq_edge_unit_num < eq_chunk_num_list[i] and i < eq_list_len - 1:
                eq_chunk_num_list[-1] += eq_chunk_num_list[i] - eq_edge_unit_num
                eq_chunk_num_list[i] = eq_edge_unit_num

        # 8. 计算边侧的切分长度
        edge_split_lengths = []
        ratio_list_start_idx = 0
        for i in range(len(cloud_split_lengths)):
            tmp_ratio_list = edge_info.ratio_list[ratio_list_start_idx : ratio_list_start_idx + eq_chunk_num_list[i]]
            ratio_list_start_idx += eq_chunk_num_list[i]
            edge_chunk_lengths = ChunkPrefilPolicy.get_edge_chunk_length_by_ratio(
                cloud_split_lengths[i], tmp_ratio_list, edge_info.min_unit
            )
            edge_split_lengths += edge_chunk_lengths

        # 9. 剩余长度，全部补到最后一份
        cloud_split_lengths[-1] += edge_remaining
        edge_split_lengths[-1] += edge_remaining

        return edge_split_lengths, cloud_split_lengths, eq_chunk_num_list

    def initialize(self, multi_nodes_enable, cp_size=1):
        self.multi_nodes_enable = multi_nodes_enable
        self.cp_size = cp_size
        self.__ajust_prefill_chunk_map_for_multi_nodes()

    def initialize_standard_card(self):
        self.ratio_list_default_edge_map = {
            125: [1] * 33,
            64: [1] * 20,
            32: [1] * 10,
            16: [1] * 4,
            8: [1] * 2,
        }
        if self.model_type == CloudCutModelType.DEEP_SEEK:
            self.ratio_list_default_edge_map = {
                31.5: [1] * 20,
                15.5: [1] * 5,
                7.5: [1] * 2,
            }
        self.ratio_list_default_cloud_map = self.ratio_list_default_edge_map

    def get_chunk_len_policy(self, prefill_seq_len):
        tmp_k_len = math.ceil(prefill_seq_len / 1024)
        edge_info = RatioInfo([1, 1], 128)  # 默认两段, 1:1, 按128对齐
        cloud_info = RatioInfo([1, 1], 128)  # 默认两段, 1:1, 按128对齐

        for (
            key,
            value,
        ) in self.ratio_list_default_edge_map.items():  # 边/云从比例key必须保持一致, 否则会报错
            if tmp_k_len >= key:
                edge_info.ratio_list = value
                cloud_info.ratio_list = self.ratio_list_default_cloud_map[key]
                break

        if self.cp_size > 1:  # 开cp之后, 按比例切分
            edge_info.min_unit = 1024
            cloud_info.min_unit = 2048

        edge_split_lengths, cloud_split_lengths, eq_chunk_num_list = self.split_long_seq_by_ratio(
            prefill_seq_len, edge_info, cloud_info
        )
        return edge_split_lengths, cloud_split_lengths, eq_chunk_num_list

    def __ajust_prefill_chunk_map_for_multi_nodes(self):
        # DS双机INT4以及单/双机INT8
        if self.model_type == CloudCutModelType.DEEP_SEEK:
            if self.moe_quantize == "w4a8_dynamic" and self.multi_nodes_enable:
                self.ratio_list_default_edge_map = {
                    31: [1] * 20,
                    15: [1] * 2,
                    7: [1] * 2,
                }
                self.ratio_list_default_cloud_map = self.ratio_list_default_edge_map
            elif self.moe_quantize != "w4a8_dynamic":
                # 双机DS INT8: 注意, 开cp之后会传入切分cp之后的长度; 这里是指cp切分之后的chunk ratio list(如32K开cp之后是16K)
                self.ratio_list_default_edge_map = {
                    63: [6] + [2] * 13 + [1] * 32,
                    31: [6] + [2] * 13,
                    15: [6, 4, 2, 2, 2],
                }
                self.ratio_list_default_cloud_map = {
                    63: [8] * 3 + [4] * 7 + [2] * 6,
                    31: [8, 8, 8, 4, 4],
                    15: [10, 6],
                }

    def __ajust_prefill_chunk_map_for_diff_npu_soc_qwen(self):
        if self.soc_name.startswith("Ascend910B4") and self.batch_p_num == 2:
            self.ratio_list_default_edge_map = {
                125: [1] * 33,
                63: [1] * 20,
                31: [1] * 10,
                15: [1] * 4,
                7: [1] * 2,
            }
            return
