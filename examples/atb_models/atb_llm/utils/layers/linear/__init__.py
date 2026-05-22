# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#implement class TensorParallelHead based on text-generation-inference
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List
from functools import wraps

import torch
from torch import nn

from atb_llm.utils.log import logger
from atb_llm.utils.layers.attention.process_mla_linear import preprocess_linear_for_rope, preprocess_kv_weights
from .fast_linear import FastLinear
from ...quantize.pack_type import TransposeType
from ...quantize.quant_type import QuantType
from ...quantize.w4a16 import W4A16LinearStatic
from ...quantize.w8a16 import W8A16LinearStatic
from ...quantize.w8a8 import W8A8LinearStatic
from ...quantize.w8a8sc import W8A8SparseCompressedLinear
from ...quantize.w8a8_dynamic import W8A8LinearDynamic
from ...quantize.w8a8_pdmix import W8A8PDMixLinear
from ...quantize.w4a8 import W4A8LinearDynamic
from ...quantize.w16a16sc import W16A16SparseCompressedLinear


IS_NZCASTED = "is_nzcasted"
MODULE_NAME = "module_name"


def support_load_sharded_weight(is_classmethod=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            module_name = kwargs.get(MODULE_NAME, kwargs.get("prefix", ""))
            weights = kwargs.get("weights", None)
            config = kwargs.get("config", None)
            if config is None:
                config = args[1] if is_classmethod else args[0]
            if weights is not None and weights.sharded:
                return TensorReplicatedLinear.load(
                    config, prefix=module_name,
                    weights=kwargs.get("weights", None),
                    bias=kwargs.get("bias", False)
                )
            else:
                if 'module_name' in kwargs:
                    del kwargs['module_name']
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_linear(weight, bias, quantize, inter_type=None, need_flatten=True, is_norm=False, **kwargs):
    nd_weight = kwargs.get('nd_weight', False)
    if quantize is None:
        linear = FastLinear(weight, bias, is_norm, nd_weight=nd_weight)
    elif quantize in [QuantType.W8A8, QuantType.W8A8S]:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, deq_scale, quant_bias, input_scale, input_offset = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w8a8` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A8LinearStatic(
                weight=qweight,
                deq_scale=deq_scale,
                input_scale=input_scale,
                quant_bias=quant_bias,
                input_offset=input_offset,
                bias=bias,
                inter_type=inter_type
            )
    elif quantize == QuantType.W4A16:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, weight_scale, weight_offset = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w4a16` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W4A16LinearStatic(
                weight=qweight,
                weight_scale=weight_scale,
                weight_offset=weight_offset,
                bias=bias
            )
    elif quantize == QuantType.W8A16:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, weight_scale, weight_offset = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w8a16` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A16LinearStatic(
                weight=qweight,
                weight_scale=weight_scale,
                weight_offset=weight_offset,
                bias=bias
            )
    elif quantize == QuantType.W8A8SC:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, deq_scale, quant_bias, input_scale, input_offset, index = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w8a8sc` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A8SparseCompressedLinear(
                weight=qweight,
                deq_scale=deq_scale,
                input_scale=input_scale,
                quant_bias=quant_bias,
                input_offset=input_offset,
                index=index
            )
    elif quantize == QuantType.W8A8_DYNAMIC:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, weight_scale, weight_offset = weight
            except Exception as err:
                logger.error(
                    "The passed weight is not `w8a8 dynamic` compatible, loader needs to be updated."
                )
                raise AssertionError from err
            linear = W8A8LinearDynamic(
                weight=qweight,
                weight_scale=weight_scale,
                weight_offset=weight_offset,
                bias=bias,
                need_flatten=need_flatten,
                nd_weight=nd_weight
            )
    elif quantize == QuantType.W8A8_PDMIX:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, weight_scale, weight_offset, deq_scale, quant_bias, input_scale, input_offset = weight
            except Exception as e:
                logger.error(
                    "The passed weight is not `w8a8 pdmix` compatible, loader needs to be updated."
                )
                raise AssertionError from e
            linear = W8A8PDMixLinear(
                weight=qweight,
                weight_scale=weight_scale,
                weight_offset=weight_offset,
                deq_scale=deq_scale,
                quant_bias=quant_bias,
                input_scale=input_scale,
                input_offset=input_offset,
                bias=bias
            )
    elif quantize == QuantType.W4A8_DYNAMIC:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                if len(weight) == 4:
                    qweight, scale, scale_second, bias = weight
                else:
                    qweight, scale, bias = weight
                    scale_second = None
            except Exception as e:
                logger.error(
                    "The passed weight is not `w4a8` compatible, loader needs to be updated."
                )
                raise AssertionError from e
            linear = W4A8LinearDynamic(
                weight=qweight,
                scale=scale,
                scale_second=scale_second,
                bias=bias,
                is_sharded=kwargs.get('is_sharded', False)
            )
    elif quantize == QuantType.W16A16S:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)                    
    elif quantize == QuantType.W16A16SC:
        if isinstance(weight, torch.Tensor):
            linear = FastLinear(weight, bias, is_norm)
        else:
            try:
                qweight, index, quant_bias = weight
            except Exception as e:
                logger.error(
                    "The passed weight is not `w16a16sc` compatible, loader needs to be updated."
                )
                raise AssertionError from e
            linear = W16A16SparseCompressedLinear(
                weight=qweight,
                index=index,
                quant_bias=quant_bias
            )
    else:
        raise AssertionError(
            f"Quantization `{quantize}` is not implemented yet. This filed is obtained from "
            f"the `quantize` field in config.json. If weights are not quantized, "
            f"this field is not required in config.json. If weights are quantized, "
            f"please refer to the model README file for more details for supported quantization types."
        )
    
    # 更新Linear metainfo
    linear.prefixes = kwargs.get("prefixes", [])
    linear.num_linear_before_pack = kwargs.get("num_linear_before_pack", 1)
    linear.tensor_parallel_dim = kwargs.get("tensor_parallel_dim", 0)
    linear.align_size = kwargs.get("align_size", 1)
    return linear


class SuperLayer(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, input_tensor):
        return self.linear.forward(input_tensor)


class TensorHead(SuperLayer):
    def __init__(self, linear):
        super().__init__(linear)

    @staticmethod
    def load_weight(config, prefix: str, weights, is_norm=False):
        weight = weights.get_whole_tensor(f"{prefix}.weight", dim=0)

        # GPTQ doesn't quantize heads (nor embeddings)
        if config.quantize == "gptq":
            quantize = None
        else:
            quantize = config.quantize
        return TensorHead(
            get_linear(weight, bias=None, quantize=quantize, inter_type=config.torch_dtype, is_norm=is_norm),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        out = torch.mm(input_tensor, self.linear.weight.T)
        return out


class TensorParallelHead(SuperLayer):
    def __init__(self, linear, process_group, should_gather: bool):
        super().__init__(linear)
        self.process_group = process_group
        self.should_gather = should_gather

    @staticmethod
    def load_weight(config, prefix: str, weights, is_norm=False):
        weight = weights.get_tensor(f"{prefix}.weight")
        should_gather = False
        # GPTQ doesn't quantize heads (nor embeddings)
        quantize = None if config.quantize == "gptq" else config.quantize
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=quantize, inter_type=config.torch_dtype, is_norm=is_norm),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    @staticmethod
    def load(config, prefix: str, weights, is_norm=False):
        should_gather = True
        if weights.process_group.size() > 1:
            try:
                weight = weights.get_sharded(f"{prefix}.weight", dim=0)
            except AssertionError:
                # If the vocab size is not divisible by number of shards
                # just load the entire thing.
                weight = weights.get_tensor(f"{prefix}.weight")
                should_gather = False
        else:
            weight = weights.get_tensor(f"{prefix}.weight")
            should_gather = False

        # GPTQ doesn't quantize heads (nor embeddings)
        quantize = None if config.quantize == "gptq" else config.quantize
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=quantize, inter_type=config.torch_dtype, is_norm=is_norm),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if not self.should_gather:
            return super().forward(input_tensor)

        world_size = self.process_group.size()
        if len(input_tensor.shape) == 2 and isinstance(self.linear, FastLinear):
            out_dim = self.linear.weight.shape[0]
            if input_tensor.shape[0] == 1:
                world_out = input_tensor.new_empty(1, out_dim * world_size)
                local_out = input_tensor.new_empty(1, out_dim)
                gather_input = local_out
            else:
                world_out = input_tensor.new_empty(out_dim * world_size, input_tensor.shape[0])
                gather_input = input_tensor.new_empty(out_dim, input_tensor.shape[0])
                local_out = gather_input.T

            torch.mm(input_tensor, self.linear.weight.T, out=local_out)
            torch.distributed.all_gather_into_tensor(
                world_out, gather_input, group=self.process_group
            )

            if input_tensor.shape[0] == 1:
                return world_out
            return world_out.T

        output = super().forward(input_tensor)
        world_output = [
            torch.empty_like(output)
            for _ in range(self.process_group.size())
        ]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        return world_output


class TensorParallelColumnLinear(SuperLayer):
    @classmethod
    def load_qkv(cls, config, prefix: str, weights, bias: bool, hidden_size, num_heads, num_kv_heads=None, dim=0):
        """Specific method when the QKV was joined after the fact"""
        if num_kv_heads is None:
            num_kv_heads = num_heads
        weight = weights.get_weights_col_packed_qkv(
            prefix, quantize=config.quantize, hidden_size=hidden_size, 
            num_heads=num_heads, num_kv_heads=num_kv_heads, dim=dim
        )
        if bias:
            bias = weights.get_tensor_col_packed_qkv(
                f"{prefix}.bias", hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads
            )
        else:
            bias = None
        linear = get_linear(
            weight, bias, config.quantize, inter_type=config.torch_dtype, prefixes=[prefix], num_linear_before_pack=3,
            tensor_parallel_dim=dim, align_size=hidden_size // num_heads,
        )
        return cls(linear)

    @classmethod
    def load_gate_up(cls, config, prefix: str, weights, bias: bool):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_mlp(
            prefix, quantize=config.quantize
        )
        if bias:
            bias = weights.get_tensor_col_packed_mlp(f"{prefix}.bias")
        else:
            bias = None
        linear = get_linear(
            weight, bias, config.quantize, inter_type=config.torch_dtype, prefixes=[prefix], num_linear_before_pack=2,
            tensor_parallel_dim=1, align_size=1
        )
        return cls(linear)

    @classmethod
    @support_load_sharded_weight()
    def load(cls, config, prefix: str, weights, bias: bool, dim=0):
        return cls.load_multi(config, [prefix], weights, bias, dim=dim)

    @classmethod
    @support_load_sharded_weight()
    def load_multi(cls, config, prefixes: List[str], weights, bias, **kwargs):
        dim = kwargs.get('dim', 0)
        proj_name = kwargs.get('proj_name', "")
        weight = weights.get_multi_weights_col(
            prefixes, quantize=config.quantize, dim=dim
        )

        if bias:
            if config.quantize == QuantType.W8A8SC:
                b = [weights.get_tensor(f"{p}.bias") for p in prefixes]
            else:
                b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
            bias = torch.cat(b, dim=0)
        else:
            bias = None

        if proj_name == "projq":
            weight = preprocess_linear_for_rope(weight, config, "projq")
        elif proj_name in ["projk", "projv"]:
            weight = preprocess_kv_weights(weight, config, proj_name)

        linear = get_linear(
            weight, bias, config.quantize, inter_type=config.torch_dtype,
            prefixes=prefixes, num_linear_before_pack=len(prefixes),
            tensor_parallel_dim=dim, align_size=1
        )
        return cls(linear)

    @classmethod
    @support_load_sharded_weight()
    def load_moe(cls, config, prefix_list: List[str], weights, bias: bool, **kwargs):
        last_dot_index = prefix_list[0][0].rfind(".")
        before_last_dot = prefix_list[0][0][:last_dot_index]
        second_last_dot_index = before_last_dot.rfind('.')
        expert_prefix = before_last_dot[:second_last_dot_index]
        routing_expert_dim = kwargs.get("routing_expert_dim", 0)
        is_nzcasted = kwargs.get(IS_NZCASTED, False)

        if len(prefix_list[0]) == 2:
            linear_index = "gate_up_proj"
        else:
            linear_index = prefix_list[0][0][last_dot_index + 1:]
        weight_list = [[] for _ in range(5)]
        for prefixes in prefix_list:
            if is_nzcasted:
                weight = weights.get_nzcasted_weights(config, prefixes)
            else:
                weight = weights.get_multi_weights_col(
                    prefixes, quantize=config.quantize, dim=0, routing_expert_dim=routing_expert_dim
                )
            if isinstance(weight, tuple):
                for i, element in enumerate(weight):
                    weight_list[i].append(element)
            else:
                weight_list[0].append(weight)
        if isinstance(weight, tuple):
            weight_stacked = []
            for i in range(len(weight)):
                weight_stacked.append(torch.stack(weight_list[i], dim=0))
        else:
            weight_stacked = torch.stack(weight_list[0], dim=0)

        if bias:
            for prefixes in prefix_list:
                if config.quantize == QuantType.W8A8SC:
                    b = [weights.get_tensor(f"{p}.bias") for p in prefixes]
                else:
                    b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
            bias = torch.cat(b, dim=0)
        else:
            bias = None

        linear = get_linear(
            weight_stacked, bias, config.quantize, need_flatten=False, prefixes=[f"{expert_prefix}.{linear_index}"],
            num_linear_before_pack=len(prefixes), tensor_parallel_dim=0, align_size=1
        )
        return cls(linear)


class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group):
        super().__init__(linear)
        self.process_group = process_group

    @classmethod
    @support_load_sharded_weight()
    def load(cls, config, prefix: str, weights, bias: bool, bias_pre_add=False, gqa_size=1,
        dim=1, transpose_b=TransposeType.INVALID, nd_weight=False, is_nzcasted=False):
        if is_nzcasted and ('layers.61' not in prefix or hasattr(config, "mtp_quantize")):
            weight = weights.get_nzcasted_weights(config, [prefix])
            nd_weight = True
        else:
            weight = weights.get_multi_weights_row(prefix, quantize=config.quantize, gqa_size=gqa_size, dim=dim)
        if bias and bias_pre_add:
            bias = weights.get_tensor(f"{prefix}.bias")
        elif bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        linear = get_linear(
            weight, bias, config.quantize, inter_type=config.torch_dtype, prefixes=[prefix],
            tensor_parallel_dim=dim, align_size=gqa_size, nd_weight=nd_weight
        )
        linear.set_transpose(transpose_b)
        return cls(linear, process_group=weights.process_group)

    @classmethod
    @support_load_sharded_weight()
    def load_moe(cls, config, prefix_list: str, process_group, weights, bias: bool, **kwargs):
        is_nzcasted = kwargs.get(IS_NZCASTED, False)
        weight_list = [[] for _ in range(5)]
        for prefixes in prefix_list:
            if is_nzcasted:
                weight = weights.get_nzcasted_weights(config, [prefixes])
            else:
                weight = weights.get_multi_weights_row(
                    prefixes, quantize=config.quantize, gqa_size=1, dim=1
                )
            if isinstance(weight, tuple):
                for i, element in enumerate(weight):
                    weight_list[i].append(element)
            else:
                weight_list[0].append(weight)
        if isinstance(weight, tuple):
            weight_stacked = []
            for i in range(len(weight)):
                weight_stacked.append(torch.stack(weight_list[i], dim=0))
        else:
            weight_stacked = torch.stack(weight_list[0], dim=0)

        if bias:
            for prefixes in prefix_list:
                if config.quantize == QuantType.W8A8SC:
                    b = [weights.get_tensor(f"{p}.bias") for p in prefixes]
                else:
                    b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
                bias = torch.cat(b, dim=0)
        else:
            bias = None

        prefix_split = prefix_list[0].split(".")
        del prefix_split[-2]
        prefix = ".".join(prefix_split)

        linear = get_linear(
            weight_stacked, bias, config.quantize, inter_type=config.torch_dtype, need_flatten=False,
            prefixes=[prefix], tensor_parallel_dim=1, nd_weight=is_nzcasted # 在RowLinear下，dim为1
        )
        return cls(linear, process_group=process_group)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = super().forward(input_tensor)
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


class TensorReplicatedLinear(SuperLayer):
    def __init__(self, linear):
        super().__init__(linear)

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_replicated_weights(prefix, quantize=config.quantize)
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None

        if "kv_a_proj" in prefix and config.model_type != "minicpm3" and not weights.sharded: # 用于MLA场景
            weight = preprocess_linear_for_rope(weight, config, "projk")

        return cls(get_linear(weight, bias, config.quantize, 
                inter_type=config.torch_dtype, prefixes=[prefix], need_flatten=not weights.sharded,
                is_sharded=weights.sharded))
