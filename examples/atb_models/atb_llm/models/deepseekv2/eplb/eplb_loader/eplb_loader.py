# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import numpy as np
import torch
import torch_npu

from atb_llm.utils.log import logger
from atb_llm.utils.weights import ProcessGroupType
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader_process import EplbLoaderProcess
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear
)


class EplbRebalanceLoader:

    def __init__(self, flash_causal_model, enable_eplb_multi_process=False):
        self.flash_causal_model = flash_causal_model
        self.layer_group_size = flash_causal_model.buffer_expert_layer_num

        # as compare with multi thread
        self.enable_eplb_multi_process = enable_eplb_multi_process
        layer_idx = self.flash_causal_model.config.first_k_dense_replace
        prefix_name = f"model.layers.{layer_idx}.mlp.experts"
        linear_names = [f'{prefix_name}.0.up_proj', f'{prefix_name}.0.gate_proj']
        pack_name = f'{prefix_name}.0.gate_up_proj'
        layer_prefix = '.'.join(prefix_name.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(self.flash_causal_model.weights, linear_names, norm_name, pack_name)
        self.replicate_experts_num_per_expert_model = None
        self.expert_id_map_model = None
        self.start_layer = None
        self.end_layer = None
        self.priority = None  # List of layer IDs specifying the order of expert weight transfer;
        self.num_moe_layers = flash_causal_model.num_layers - layer_idx
        self.num_experts = flash_causal_model.num_of_experts
        if self.enable_eplb_multi_process:
            self.eplb_loader_process = EplbLoaderProcess(flash_causal_model.weights)

    # 计算D2D的批次，需要与H2D分批的算法保持一致
    def h2d_update_times(self, model):
        if self.priority is None:
            self.num_moe_layers = model.num_layers - model.first_k_dense_replace
            self.priority = np.arange(self.num_moe_layers)
        else:
            self.num_moe_layers = len(self.priority)
        update_times = math.ceil(self.num_moe_layers / self.layer_group_size)
        return update_times

    def do_load_prepare_h2d(self, copy_stream, new_expert_map, i):
        self.start_layer = i * self.layer_group_size
        self.end_layer = min((i + 1) * self.layer_group_size, self.num_moe_layers)
        return self.update(copy_stream, new_expert_map)

    def load_weight_row_linear_from_ssd(self, config, prefix_list, process_group, weights, bias):
        if self.enable_eplb_multi_process:
            result = self.eplb_loader_process.load_weight_row_linear_from_ssd(config, prefix_list, process_group, bias)
            if not self.eplb_loader_process.is_alive:
                raise Exception("eplb_loader_process is shut down")
        else:
            result = TensorParallelRowLinear.load_moe(
                config,
                prefix_list=prefix_list,
                process_group=process_group,
                weights=weights,
                bias=False
            )
        return result

    def load_weight_column_linear_from_ssd(self, config, prefix_list, weights, bias):
        if self.enable_eplb_multi_process:
            result = self.eplb_loader_process.load_weight_column_linear_from_ssd(
                config, prefix_list, bias
            )
            if not self.eplb_loader_process.is_alive:
                logger.warning("eplb_loader_process is shut down")
                raise Exception("eplb_loader_process is shut down")
        else:
            result = TensorParallelColumnLinear.load_moe(
                config,
                prefix_list=prefix_list,
                weights=weights,
                bias=False
            )
        return result

    def load_weight_ssd2host(self, update_expert_list, layer_idx):
        update_expert_weight = []
        if len(update_expert_list) == 0:
            return update_expert_weight

        if self.flash_causal_model.ep:
            self.flash_causal_model.weights.switch_process_group(ProcessGroupType.MOE_EP)

        priority = self.priority + self.flash_causal_model.config.first_k_dense_replace
        for update_experts in update_expert_list:
            update_expert_weight.append([])
            prefix_name = f"model.layers.{priority[layer_idx]}.mlp.experts"
            if self.pack_type in [
                PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
                PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
                PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
                PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI
            ]:
                pack_prefixes = [[f"{prefix_name}.{i}.gate_proj", f"{prefix_name}.{i}.up_proj"] \
                                 for i in update_experts]
                # Replace "experts.{self.num_experts}" with "shared_experts" in all shared expert prefixes (DeepSeekV3)
                for i, _ in enumerate(pack_prefixes):
                    for j, _ in enumerate(pack_prefixes[i]):
                        if f"experts.{self.num_experts}" in pack_prefixes[i][j]:
                            pack_prefixes[i][j] = \
                                pack_prefixes[i][j].replace(f"experts.{self.num_experts}", "shared_experts")

                gate_up_proj = self.load_weight_column_linear_from_ssd(
                    self.flash_causal_model.config,
                    prefix_list=pack_prefixes,
                    weights=self.flash_causal_model.weights,
                    bias=False
                )

                update_expert_weight[-1].append(gate_up_proj.linear.weight.data)
                update_expert_weight[-1].append(gate_up_proj.linear.weight_offset.data)
                update_expert_weight[-1].append(gate_up_proj.linear.weight_scale.data)
            else:
                msg = f"Dynamic EPLB not support this pack_type {self.pack_type}."
                raise TypeError(msg)

            down_prefixes = [f"{prefix_name}.{i}.down_proj" \
                             for i in update_experts]
            # Replace "experts.{self.num_experts}" with "shared_experts" in all shared expert prefixes (DeepSeekV3)
            for i, _ in enumerate(down_prefixes):
                if f"experts.{self.num_experts}" in down_prefixes[i]:
                    down_prefixes[i] = down_prefixes[i].replace(f"experts.{self.num_experts}", "shared_experts")

            down_proj = self.load_weight_row_linear_from_ssd(
                self.flash_causal_model.config,
                prefix_list=down_prefixes,
                process_group=self.flash_causal_model.weights.process_group,
                weights=self.flash_causal_model.weights,
                bias=False
            )
            update_expert_weight[-1].append(down_proj.linear.weight.data)
            update_expert_weight[-1].append(down_proj.linear.weight_offset.data)
            update_expert_weight[-1].append(down_proj.linear.weight_scale.data)
            layer_idx += 1
        return update_expert_weight

    def update(
            self,
            copy_stream,
            new_expert_map: torch.tensor,
    ):

        new_experts_cpu = self.load_weight_ssd2host(
            new_expert_map[self.priority[self.start_layer:self.end_layer]],
            self.start_layer)

        if len(new_experts_cpu) == 0:
            return False
        # 开始数据拷贝操作
        with torch_npu.npu.stream(copy_stream):
            index = 0
            for layer_tensors in new_experts_cpu:
                for tensor in layer_tensors:
                    self.flash_causal_model.ascend_buffer_weight[index].copy_(tensor)
                    index += 1

        return True

    def weight_memory_copy(self, start_layer, end_layer):
        # DEVICE BUFFER -> DEVICE MODEL WEIGHT
        logger.debug(
            f"--------d2d start------\n"
            f"start_layer:{start_layer}, end_layer:{end_layer}, priority: {self.priority}")
        old_expert_weight_ids = []
        for i in range(start_layer, end_layer):
            start_idx = self.flash_causal_model.buffer_replace_weights_ids[self.priority[i]][0]
            end_idx = self.flash_causal_model.buffer_replace_weights_ids[self.priority[i]][1]
            for old_expert_weight_id in range(start_idx, end_idx):
                if (self.flash_causal_model.ascend_weight[old_expert_weight_id].data_ptr()
                        != self.flash_causal_model.placeholder_dataptr):
                    old_expert_weight_ids.append(old_expert_weight_id)

        self.flash_causal_model.acl_encoder_operation.update_weights_ptr(
            self.flash_causal_model.ascend_buffer_weight[:len(old_expert_weight_ids)], old_expert_weight_ids)
        self.flash_causal_model.acl_decoder_operation.update_weights_ptr(
            self.flash_causal_model.ascend_buffer_weight[:len(old_expert_weight_ids)], old_expert_weight_ids)
        new_expert_weight_id = 0
        for i in old_expert_weight_ids:
            tmp_tensor = torch.tensor([],
                                      device=self.flash_causal_model.ascend_weight[i].device,
                                      dtype=self.flash_causal_model.ascend_weight[i].dtype
                                      ).set_(self.flash_causal_model.ascend_weight[i])
            self.flash_causal_model.ascend_weight[i].set_(
                self.flash_causal_model.ascend_buffer_weight[new_expert_weight_id])
            self.flash_causal_model.ascend_buffer_weight[new_expert_weight_id].set_(tmp_tensor)
            new_expert_weight_id += 1
        logger.debug("--------d2d finished------")

    def shutdown(self):
        if self.enable_eplb_multi_process:
            self.eplb_loader_process.shutdown()
