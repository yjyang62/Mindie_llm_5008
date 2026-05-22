# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from torch import multiprocessing as mp
import torch

from atb_llm.utils.log import logger
from atb_llm.utils.weights import ProcessGroupType, Weights
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear
)


class EplbLoaderProcess():
    def __init__(self, weights: Weights):
        self.is_alive = True
        spawn_ctx = mp.get_context("spawn")
        self.weight_column_linear_in = spawn_ctx.Queue()
        self.weight_column_linear_out = spawn_ctx.Queue()
        self.weight_column_linear_process \
            = spawn_ctx.Process(target=EplbLoaderProcess._do_load_weight_column_linear_from_ssd,
                                args=(weights, self.weight_column_linear_in, self.weight_column_linear_out))
        self.weight_column_linear_process.start()

        self.weight_row_linear_in = spawn_ctx.Queue()
        self.weight_row_linear_out = spawn_ctx.Queue()
        self.weight_row_linear_process = (
            spawn_ctx.Process(target=EplbLoaderProcess._do_load_weight_row_linear_from_ssd,
                              args=(weights, self.weight_row_linear_in, self.weight_row_linear_out))
        )
        self.weight_row_linear_process.start()

    @staticmethod
    def _do_load_weight_column_linear_from_ssd(weights, in_q, out_q):
        weights.switch_process_group(ProcessGroupType.MOE_EP)

        while True:
            logger.debug("do_load_weight_column_linear_from_ssd process start.")
            args = in_q.get()
            if args == "exist":
                break

            config, pack_prefixes, bias = args
            result, tensor_weight, tensor_weight_scale, tensor_weight_offset \
                = EplbLoaderProcess._do_load_weight_column_linear_from_ssd_(weights, config, pack_prefixes, bias)
            logger.debug(f"do_load_weight_column_linear_from_ssd process finished: {result}.")
            out_q.put((result, tensor_weight, tensor_weight_scale, tensor_weight_offset))

    @staticmethod
    def _do_load_weight_column_linear_from_ssd_(weights, config, pack_prefixes, bias):
        result = TensorParallelColumnLinear.load_moe(
            config,
            prefix_list=pack_prefixes,
            weights=weights,
            bias=bias,
            routing_expert_dim=1
        )

        tensor_weight = result.linear.weight
        tensor_weight.share_memory_()
        tensor_weight_scale = result.linear.weight_scale
        tensor_weight_scale.share_memory_()
        tensor_weight_offset = result.linear.weight_offset
        tensor_weight_offset.share_memory_()

        # 清除linear buffer，避免复杂tensor序列化
        result.linear.weight = torch.zeros(1)
        result.linear.weight_scale = torch.zeros(1)
        result.linear.weight_offset = torch.zeros(1)

        return result, tensor_weight, tensor_weight_scale, tensor_weight_offset

    @staticmethod
    def _do_load_weight_row_linear_from_ssd(weights, in_q, out_q):
        weights.switch_process_group(ProcessGroupType.MOE_EP)
        
        while True:
            logger.debug("do_load_weight_row_linear_from_ssd_ process start.")
            args = in_q.get()
            if args == "exist":
                break
            config, prefix_list, process_group, bias = args
            result, tensor_weight, tensor_weight_scale, tensor_weight_offset = (
                EplbLoaderProcess._do_load_weight_row_linear_from_ssd_(
                    weights, config, prefix_list, process_group, bias
                )
            )
            logger.debug(f"do_load_weight_row_linear_from_ssd_ process finished: result {result}.")
            out_q.put((result, tensor_weight, tensor_weight_scale, tensor_weight_offset))

    @staticmethod
    def _do_load_weight_row_linear_from_ssd_(weights, config, prefix_list, process_group, bias):
        result = TensorParallelRowLinear.load_moe(
            config,
            prefix_list=prefix_list,
            process_group=process_group,
            weights=weights,
            bias=bias
        )
        # 将张量的存储放入共享内存
        tensor_weight = result.linear.weight
        tensor_weight.share_memory_()  # 标记为共享内存
        tensor_weight_scale = result.linear.weight_scale
        tensor_weight_scale.share_memory_()  # 标记为共享内存
        tensor_weight_offset = result.linear.weight_offset
        tensor_weight_offset.share_memory_()  # 标记为共享内存
        # 清除linear buffer，避免复杂tensor序列化
        result.linear.weight = torch.zeros(1)
        result.linear.weight_scale = torch.zeros(1)
        result.linear.weight_offset = torch.zeros(1)

        return result, tensor_weight, tensor_weight_scale, tensor_weight_offset

    def graceful_exit(self, signum, frame):
        self.is_alive = False
        self.weight_column_linear_in.put("exist")
        self.weight_row_linear_in.put("exist")
        self.weight_column_linear_process.terminate()
        self.weight_row_linear_process.terminate()

    def shutdown(self):
        self.graceful_exit(0, 0)

    def load_weight_column_linear_from_ssd(self, config, pack_prefixes, bias):
        self.weight_column_linear_in.put((config, pack_prefixes, bias))
        result = None
        if self.is_alive and self.weight_column_linear_process.is_alive():
            result, tensor_weight, tensor_weight_scale, tensor_weight_offset = self.weight_column_linear_out.get()
            # 组装回result
            result.linear.weight = tensor_weight
            result.linear.weight_scale = tensor_weight_scale
            result.linear.weight_offset = tensor_weight_offset
        else:
            result = None
        return result

    def load_weight_row_linear_from_ssd(self, config, prefix_list, process_group, bias):

        self.weight_row_linear_in.put((config, prefix_list, process_group, bias))
        if self.is_alive and self.weight_row_linear_process.is_alive():
            result, tensor_weight, tensor_weight_scale, tensor_weight_offset = self.weight_row_linear_out.get()
            # 组装回result
            result.linear.weight = tensor_weight
            result.linear.weight_scale = tensor_weight_scale
            result.linear.weight_offset = tensor_weight_offset
        else:
            result = None
        return result
