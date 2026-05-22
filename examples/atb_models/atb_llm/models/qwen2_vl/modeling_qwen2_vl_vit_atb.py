# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement part of this file based on from transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""PyTorch Qwen2-VL-VIT model."""

import json
import math
from collections import OrderedDict, defaultdict

import _libatb_torch as atb
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch_npu
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import (
    CommonOpBuilderType,
    NormType,
)
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import (
    ParallelType,
    TensorParallelInfo
)
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorReplicatedLinear
)
from atb_llm.utils.quantize.quant_type import QuantType
from transformers.modeling_utils import PreTrainedModel

PADDING_HEAD_DIM = 128
PATCH_MERGER_FACTOR = 4
OUT_DATA_TYPE = {
    torch.bfloat16: "ACL_BF16",
    torch.float16: "ACL_FLOAT16"
}
QUANT_TYPE = {
    QuantType.W8A8: "QUANT_INT8"
}


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, device, dtype, dim: int, theta: float = 10000.0, seqlen: int = 100000) -> None:
        super().__init__()
        self._seq_len_cached = 0
        self.scaling_factor = 1.0
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))

        t = torch.arange(seqlen, device=device, dtype=torch.float)
        t = t / self.scaling_factor
        # Don't do einsum, it converts fp32 to fp16 # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        self.cos_cache = self.freqs.cos().to(dtype)
        self.sin_cache = self.freqs.sin().to(dtype)


class FastPatchEmbed(nn.Module):
    def __init__(self, config, weights, prefix, backend=None) -> None:
        super().__init__()
        self.prefix = prefix
        self.proj = TensorReplicatedLinear.load(
            config,
            prefix=f"{prefix}.proj",
            weights=weights,
            bias=False
        )

    def get_weights(self):
        weights_dict = OrderedDict()
        linear_weight = self.proj.linear.get_weights(f"{self.prefix}.proj")[f"{self.prefix}.proj.weight"]
        new_weight_dict = OrderedDict()
        linear_weight = linear_weight.contiguous()
        linear_weight = torch_npu.npu_format_cast(linear_weight, 2)
        new_weight_dict[f"{self.prefix}.proj.weight"] = linear_weight.view(linear_weight.shape[0], -1)
        weights_dict.update(new_weight_dict)
        return weights_dict

    def build_graph(self, graph, input_tensor_name, output_tensor_name):
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": False}),
            op_name="patch_embed_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            [input_tensor_name, f"{self.prefix}.proj.weight"],
            [output_tensor_name]
        )


class PatchMerger(nn.Module):
    def __init__(self, config, weights, prefix, backend):
        super().__init__()
        self.config = config
        self.weights = weights
        self.dtype = weights.dtype
        self.backend = backend

        self.prefix = prefix
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        if config.enable_atb_vit_tp:
            self.patch_merger_mlp_0 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.mlp.0",
                weights=weights,
                bias=True
            )

            self.patch_merger_mlp_2 = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.mlp.2",
                weights=weights,
                bias=True
            )
        else:
            self.patch_merger_mlp_0 = TensorReplicatedLinear.load(
                config,
                prefix=f"{prefix}.mlp.0",
                weights=weights,
                bias=True
            )

            self.patch_merger_mlp_2 = TensorReplicatedLinear.load(
                config,
                prefix=f"{prefix}.mlp.2",
                weights=weights,
                bias=True
            )

        self.patch_merger_ln_q = LayerNormATB(
            weights,
            f"{prefix}.ln_q"
        )

    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.patch_merger_ln_q.get_weights(f"{self.prefix}.ln_q"))
        weights_dict.update(self.patch_merger_mlp_0.linear.get_weights(f"{self.prefix}.mlp.0"))
        weights_dict.update(self.patch_merger_mlp_2.linear.get_weights(f"{self.prefix}.mlp.2"))
        return weights_dict

    def build_mlp0_graph(self, graph, input_tensor_name, output_tensor_name):
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="patch_merger_mlp_0_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            [input_tensor_name, f"{self.prefix}.mlp.0.weight", f"{self.prefix}.mlp.0.bias"],
            [output_tensor_name]
        )

    def build_activation_graph(self, graph, input_tensor_name, output_tensor_name):
        act = atb.BaseOperation(
            op_type="Activation",
            op_param=json.dumps({"activationType": "ACTIVATION_GELU"}),
            op_name="patch_merger_activation",
        )
        graph.operations.append(act)
        graph.add_operation(
            act,
            [input_tensor_name],
            [output_tensor_name],
        )

    def build_mlp2_graph(self, graph, input_tensor_name, output_tensor_name):
        mlp2_param = {
            "op_name": "patch_merger_mlp_2",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.patch_merger_mlp_2.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        mlp2_tensor_map = {
            "input": input_tensor_name,
            "linear_out": output_tensor_name,
        }

        mlp2_parallel_param = {
            "op_name": "patch_merger_mlp_2_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend
            ),
            "linear_param": mlp2_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            mlp2_parallel_param
        )
        graph = linear_parallel_builder.build(graph, mlp2_tensor_map)

    def build_mlp2_graph_without_tp(self, graph, input_tensor_name, output_tensor_name):
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="patch_merger_mlp_2"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            [input_tensor_name, f"{self.prefix}.mlp.2.weight", f"{self.prefix}.mlp.2.bias"],
            [output_tensor_name]
        )

    def reshape_out(self, org_shape):
        return [1, org_shape[1] // PATCH_MERGER_FACTOR, org_shape[2] * PATCH_MERGER_FACTOR]

    def build_graph(self, graph, input_tensor_name, output_tensor_name):
        self.patch_merger_ln_q.build_graph(graph, input_tensor_name, f"{self.prefix}.ln_q_out")
        graph.add_reshape(f"{self.prefix}.ln_q_out", f"{self.prefix}.ln_q_out_reshape", self.reshape_out)
        self.build_mlp0_graph(graph, f"{self.prefix}.ln_q_out_reshape", f"{self.prefix}.mlp_0_out")
        self.build_activation_graph(graph, f"{self.prefix}.mlp_0_out", f"{self.prefix}.activation_out")
        if self.config.enable_atb_vit_tp:
            self.build_mlp2_graph(graph, f"{self.prefix}.activation_out", output_tensor_name)
        else:
            self.build_mlp2_graph_without_tp(graph, f"{self.prefix}.activation_out", output_tensor_name)


class LayerNormATB(nn.Module):
    def __init__(self, weights, prefix, quantize=None):
        super().__init__()
        self.layer_norm_eps = 1e-6
        self.prefix = prefix
        self.weights = weights
        self.quantize = quantize

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict[f"{prefix}.weight"] = self.weights.get_tensor(f"{prefix}.weight").data.npu()
        weights_dict[f"{prefix}.bias"] = self.weights.get_tensor(f"{prefix}.bias").data.npu()
        if self.quantize == QuantType.W8A8:
            if "norm1" in prefix: 
                prefix = prefix.replace("norm1", "attn.qkv")
            elif "norm2" in prefix:
                prefix = prefix.replace("norm2", "mlp.fc1")
            weights_dict[f"{self.prefix}.input_scale"] = self.weights.get_tensor(f"{prefix}.input_scale").npu()
            weights_dict[f"{self.prefix}.input_offset"] = self.weights.get_tensor(f"{prefix}.input_offset") \
                                                            .to(dtype=torch.int8).npu()
        return weights_dict

    def build_graph(self, graph, input_tensor_name, output_tensor_name):
        input_key_list = [input_tensor_name, f"{self.prefix}.weight", f"{self.prefix}.bias"]
        if self.quantize == QuantType.W8A8:
            input_key_list.extend([f"{self.prefix}.input_scale", f"{self.prefix}.input_offset"])
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": QUANT_TYPE.get(self.quantize, "QUANT_UNDEFINED"),
                "epsilon": self.layer_norm_eps,
                "beginParamsAxis": 2,
                "beginNormAxis": 2
            },
        }
        norm_op = atb.BaseOperation(
            op_type=NormType.LAYERNORM,
            op_param=json.dumps(norm_param),
            op_name="layer_norm",
        )
        graph.operations.append(norm_op)
        graph.add_operation(
            norm_op,
            input_key_list,
            [output_tensor_name]
        )


class VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, weights, prefix, norm_prefix, backend):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.norm_prefix = norm_prefix
        self.backend = backend
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.num_heads = config.num_heads
        self.hidden_size = config.embed_dim
        self.embed_dim = config.embed_dim
        self.head_dim_ori = self.embed_dim // self.num_heads
        self.head_dim = PADDING_HEAD_DIM
        self.quantize = config.quantize
        if config.enable_atb_vit_tp and self.quantize != QuantType.W8A8:
            self.num_heads_pre_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        else:
            self.num_heads_pre_rank = self.num_heads
        if config.enable_atb_vit_tp and self.quantize != QuantType.W8A8:
            self.qkv = TensorParallelColumnLinear.load_qkv(
                config,
                prefix=f"{self.prefix}.qkv",
                weights=weights,
                bias=True,
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
            )

            self.proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{self.prefix}.proj",
                weights=weights,
                bias=True,
                gqa_size=self.head_dim_ori,
            )
        else:
            self.qkv = TensorReplicatedLinear.load(
                config,
                prefix=f"{self.prefix}.qkv",
                weights=weights,
                bias=True,
            )

            self.proj = TensorReplicatedLinear.load(
                config,
                prefix=f"{self.prefix}.proj",
                weights=weights,
                bias=True,
            )

    def pad_qkv_bias(self, bias):
        first_half = bias.reshape(self.num_heads_pre_rank, 3, 80)[:, :, :40]
        second_half = bias.reshape(self.num_heads_pre_rank, 3, 80)[:, :, 40:]
        first_half_padded = torch.nn.functional.pad(first_half, (0, 24))
        second_half_padded = torch.nn.functional.pad(second_half, (0, 24))
        bias_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        bias_final = bias_padded.reshape(self.num_heads_pre_rank * 128 * 3)
        return bias_final
        
    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        qkv_weights_dict = self.qkv.linear.get_weights(f"{prefix}.qkv")
        qkv_proj_weight = qkv_weights_dict[f"{prefix}.qkv.weight"]
        qkv_proj_bias = qkv_weights_dict[f"{prefix}.qkv.bias"]

        # padding head_dim from 80 to 128
        first_half = qkv_proj_weight.reshape(self.num_heads_pre_rank, 3, 80, self.embed_dim)[:, :, :40, :]
        second_half = qkv_proj_weight.reshape(self.num_heads_pre_rank, 3, 80, self.embed_dim)[:, :, 40:, :]
        first_half_padded = torch.nn.functional.pad(first_half, (0, 0, 0, 24))
        second_half_padded = torch.nn.functional.pad(second_half, (0, 0, 0, 24))
        qkv_proj_weight_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        qkv_proj_weight_final = qkv_proj_weight_padded.reshape(self.num_heads_pre_rank * 128 * 3, self.embed_dim)

        qkv_proj_bias_final = self.pad_qkv_bias(qkv_proj_bias)
        if self.quantize == QuantType.W8A8:
            qkv_proj_quant_bias = qkv_weights_dict[f"{prefix}.qkv.quant_bias"]
            qkv_proj_deq_scale = qkv_weights_dict[f"{prefix}.qkv.deq_scale"]
            qkv_proj_quant_bias_final = self.pad_qkv_bias(qkv_proj_quant_bias)
            qkv_proj_deq_scale_final = self.pad_qkv_bias(qkv_proj_deq_scale)
            qkv_weights_dict[f"{prefix}.qkv.quant_bias"] = qkv_proj_quant_bias_final
            qkv_weights_dict[f"{prefix}.qkv.deq_scale"] = qkv_proj_deq_scale_final

        qkv_weights_dict[f"{prefix}.qkv.weight"] = qkv_proj_weight_final
        qkv_weights_dict[f"{prefix}.qkv.bias"] = qkv_proj_bias_final

        out_proj_weights_dict = self.proj.linear.get_weights(f"{prefix}.proj")
        out_proj_weight = out_proj_weights_dict[f"{prefix}.proj.weight"]
        soc_info = NPUSocInfo()
        if not self.config.enable_atb_vit_tp or self.tp_world_size == 1 or soc_info.need_nz:
            out_proj_weight = torch.nn.functional.pad(
                out_proj_weight.reshape(self.embed_dim, self.num_heads_pre_rank * 2, 40), (0, 24, 0, 0)
            ).reshape(self.embed_dim, self.num_heads_pre_rank * 128)
        elif self.tp_world_size > 1:
            first_half = out_proj_weight.reshape(self.num_heads_pre_rank, 80, self.embed_dim)[:, :40, :]
            second_half = out_proj_weight.reshape(self.num_heads_pre_rank, 80, self.embed_dim)[:, 40:, :]
            first_half_padded = torch.nn.functional.pad(first_half, (0, 0, 0, 24))
            second_half_padded = torch.nn.functional.pad(second_half, (0, 0, 0, 24))
            out_proj_weight_padded = torch.cat([first_half_padded, second_half_padded], dim=1)
            out_proj_weight = out_proj_weight_padded.reshape(self.num_heads_pre_rank * 128, self.embed_dim)

        out_proj_weights_dict[f"{prefix}.proj.weight"] = out_proj_weight
        weights_dict.update(qkv_weights_dict)
        weights_dict.update(out_proj_weights_dict)
        return weights_dict

    def build_qkv_graph(self, graph):
        if self.quantize == QuantType.W8A8:
            input_key_list = [f"{self.norm_prefix}_out", f"{self.prefix}.qkv.weight", 
                              f"{self.prefix}.qkv.quant_bias", f"{self.prefix}.qkv.deq_scale"]
            out_data_type = OUT_DATA_TYPE.get(self.dtype, "ACL_DT_UNDEFINED")
        else:
            input_key_list = [f"{self.norm_prefix}_out", f"{self.prefix}.qkv.weight", f"{self.prefix}.qkv.bias"]
            out_data_type = "ACL_DT_UNDEFINED"
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True,
                "outDataType": out_data_type}),
            op_name="layer_qkv_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            ["qkv_linear_out"]
        )
        split_op = atb.BaseOperation(
            op_type="Split",
            op_param=json.dumps({
                "splitDim": -1,
                "splitNum": 3
            }),
            op_name="layer_qkv_split"
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            ["qkv_linear_out"],
            ["q_split", "k_split", "v_split"],
        )

    def build_rope_graph(self, graph):
        rope_param = {
            "op_name": "layer_rope",
            "head_num": self.num_heads_pre_rank,
            "kv_head_num": self.num_heads_pre_rank,
            "category": CommonOpBuilderType.ROPE,
            "is_fa": True,
            "atb_rope_param": {"rotaryCoeff": 2},
        }
        rope_tensor_map = {
            "q": "q_split",
            "k": "k_split",
            "cos_embedding": "cos_embedding",
            "sin_embedding": "sin_embedding",
            "seq_len": "seq_len",
            "q_out": "q_split",
            "k_out": "k_split",
        }
        rope_builder = CommonOpBuilderManager.get_builder(rope_param)
        graph = rope_builder.build(graph, rope_tensor_map)

    def reshape_qkv(self, org_shape):
        return [org_shape[0] * org_shape[1], self.num_heads_pre_rank, self.head_dim]

    def reshape_out(self, org_shape):
        return [1, org_shape[0], self.num_heads_pre_rank * org_shape[2]]

    def build_attention_graph(self, graph):
        attention_op = atb.BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps({
                "headNum": self.num_heads_pre_rank,
                "kvHeadNum": self.num_heads_pre_rank,
                "qkScale": 1.0 / math.sqrt(self.head_dim_ori),
                "calcType": "PA_ENCODER"}),
            op_name="layer_selfattention"
        )
        graph.add_reshape("q_split", "q_split_reshape", self.reshape_qkv)
        graph.add_reshape("k_split", "k_split_reshape", self.reshape_qkv)
        graph.add_reshape("v_split", "v_split_reshape", self.reshape_qkv)
        input_key_list = ["q_split_reshape", "k_split_reshape", "v_split_reshape", "seq_len"]
        output_key_list = ["attn_out"]
        graph.operations.append(attention_op)
        graph.add_operation(attention_op, input_key_list, output_key_list)
        graph.add_reshape("attn_out", "attn_out_reshape", self.reshape_out)

    def build_output_proj_graph(self, graph):
        output_proj_linear_param = {
            "op_name": "layer_output_proj_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.proj.linear,
            "enable_quant_input": True if self.quantize == QuantType.W8A8 else False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        output_proj_linear_tensor_map = {
            "input": "attn_out_reshape",
            "linear_out": "output_proj_out"
        }
        output_proj_linear_parallel_param = {
            "op_name": "layer_output_proj_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": output_proj_linear_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(output_proj_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, output_proj_linear_tensor_map)

    def build_output_proj_graph_without_tp(self, graph):
        if self.quantize == QuantType.W8A8:
            input_quant_op = atb.BaseOperation(
                op_type="Elewise",
                op_param=json.dumps({"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"}),
                op_name="attn_proj_elewise_quant"
            )
            graph.operations.append(input_quant_op)
            graph.add_operation(
                input_quant_op,
                ["attn_out_reshape", f"{self.prefix}.proj.input_scale", f"{self.prefix}.proj.input_offset"],
                ["attn_out_reshape_quant_out"]
            )
        
        input_key_list = ["attn_out_reshape", f"{self.prefix}.proj.weight", f"{self.prefix}.proj.bias"]
        out_data_type = "ACL_DT_UNDEFINED"
        transpose_b = True
        if self.quantize == QuantType.W8A8:
            input_key_list = ["attn_out_reshape_quant_out", f"{self.prefix}.proj.weight", 
                              f"{self.prefix}.proj.quant_bias", f"{self.prefix}.proj.deq_scale"]
            out_data_type = OUT_DATA_TYPE.get(self.dtype, "ACL_DT_UNDEFINED")
            if self.tp_world_size > 1:
                transpose_b = False
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": transpose_b,
                "hasBias": True,
                "outDataType": out_data_type}),
            op_name="layer_output_proj_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            ['output_proj_out']
        )

    def build_graph(self, graph):
        attn_res_add = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="layer_attn_res_add",
        )
        self.build_qkv_graph(graph)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph)
        if self.config.enable_atb_vit_tp and self.quantize != QuantType.W8A8:
            self.build_output_proj_graph(graph)
        else:
            self.build_output_proj_graph_without_tp(graph)
        graph.operations.append(attn_res_add)
        graph.add_operation(
            attn_res_add, ["hidden_states", "output_proj_out"], ["hidden_states"]
        )


class VisionMlp(nn.Module):
    def __init__(self, config, weights, prefix, norm_prefix, backend):
        super().__init__()
        self.norm_prefix = norm_prefix
        self.config = config
        self.weights = weights
        self.dtype = weights.dtype
        self.backend = backend
        self.prefix = prefix
        self.quantize = config.quantize
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        if self.config.enable_atb_vit_tp:
            self.fc1 = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.fc1",
                weights=weights,
                bias=True
            )
            self.fc2 = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.fc2",
                weights=weights,
                bias=True
            )
        else:
            self.fc1 = TensorReplicatedLinear.load(
                config,
                prefix=f"{prefix}.fc1",
                weights=weights,
                bias=True
            )
            self.fc2 = TensorReplicatedLinear.load(
                config,
                prefix=f"{prefix}.fc2",
                weights=weights,
                bias=True
            )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.fc1.linear.get_weights(f"{prefix}.fc1"))
        weights_dict.update(self.fc2.linear.get_weights(f"{prefix}.fc2"))
        return weights_dict

    def build_fc1_graph(self, graph, input_tensor_name, output_tensor_name):
        if self.quantize == QuantType.W8A8:
            input_key_list = [input_tensor_name, f"{self.prefix}.fc1.weight", f"{self.prefix}.fc1.quant_bias", 
                              f"{self.prefix}.fc1.deq_scale"]
            out_data_type = OUT_DATA_TYPE.get(self.dtype, "ACL_DT_UNDEFINED")
        else: 
            input_key_list = [input_tensor_name, f"{self.prefix}.fc1.weight", f"{self.prefix}.fc1.bias"]
            out_data_type = "ACL_DT_UNDEFINED"
        linear_out = [output_tensor_name]
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True,
                "outDataType": out_data_type}),
            op_name="layer_fc1_Linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            linear_out
        )

    def build_activation_graph(self, graph, input_tensor_name, output_tensor_name):
        act = atb.BaseOperation(
            op_type="Activation",
            op_param=json.dumps({"activationType": "ACTIVATION_GELU"}),
            op_name="layer_activation",
        )
        graph.operations.append(act)
        graph.add_operation(
            act,
            [input_tensor_name],
            [output_tensor_name],
        )

    def build_fc2_graph(self, graph, input_tensor_name, output_tensor_name):
        fc2_param = {
            "op_name": "layer_fc2_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.fc2.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        fc2_tensor_map = {
            "input": input_tensor_name,
            "linear_out": output_tensor_name,
        }

        fc2_parallel_param = {
            "op_name": "layer_fc2_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend
            ),
            "linear_param": fc2_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            fc2_parallel_param
        )
        graph = linear_parallel_builder.build(graph, fc2_tensor_map)

    def build_fc2_graph_without_tp(self, graph, input_tensor_name, output_tensor_name):
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="layer_fc2_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            [input_tensor_name, f"{self.prefix}.fc2.weight", f"{self.prefix}.fc2.bias"],
            [output_tensor_name]
        )

    def build_graph(self, graph):
        mlp_res_add = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="mlp_res_add",
        )
        setattr(graph, "mlp_res_add", mlp_res_add)
        self.build_fc1_graph(graph, f"{self.norm_prefix}_out", f"{self.prefix}.fc1_out")
        self.build_activation_graph(graph, f"{self.prefix}.fc1_out", f"{self.prefix}.activation_out")
        if self.config.enable_atb_vit_tp:
            self.build_fc2_graph(graph, f"{self.prefix}.activation_out", f"{self.norm_prefix}.fc2_out")
        else:
            self.build_fc2_graph_without_tp(graph, f"{self.prefix}.activation_out", f"{self.norm_prefix}.fc2_out")
        graph.operations.append(graph.mlp_res_add)
        graph.add_operation(
            graph.mlp_res_add, ["hidden_states", f"{self.norm_prefix}.fc2_out"], ["layer_out"]
        )


class Qwen2VLVisionLayerATB(nn.Module):
    def __init__(self, config, layer_idx, weights, layer_prefix, backend) -> None:
        super().__init__()
        prefix = f"{layer_prefix}.{layer_idx}"
        self.prefix = prefix
        self.layer_idx = layer_idx
        self.config = config
        self.weight_names = None
        self.layer_graph = None
        self.norm1 = LayerNormATB(
            weights,
            f"{prefix}.norm1",
            quantize=config.quantize
        )

        self.attn = VisionAttention(
            config=config,
            weights=weights,
            prefix=f"{prefix}.attn",
            norm_prefix=f"{prefix}.norm1",
            backend=backend,
        )

        self.norm2 = LayerNormATB(
            weights,
            f"{prefix}.norm2",
            quantize=config.quantize
        )

        self.mlp = VisionMlp(
            config=config,
            weights=weights,
            prefix=f"{prefix}.mlp",
            norm_prefix=f"{prefix}.norm2",
            backend=backend,
        )

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def get_in_tensor_names(self):
        return ["hidden_states", "seq_len", "cos_embedding", "sin_embedding"]

    def get_out_tensor_names(self):
        return ["hidden_states"]

    def build_graph(self, graph):
        self.layer_graph = AtbGraph(f"Qwen2VL_VIT_layer_{self.layer_idx}_graph")
        self.layer_graph.add_input_output(
            input=self.weight_names + self.get_in_tensor_names(),
            output=["layer_out"],
        )

        self.norm1.build_graph(self.layer_graph, self.get_in_tensor_names()[0], f"{self.prefix}.norm1_out")
        self.attn.build_graph(self.layer_graph)
        self.norm2.build_graph(self.layer_graph, self.get_in_tensor_names()[0], f"{self.prefix}.norm2_out")
        self.mlp.build_graph(self.layer_graph)
        self.layer_graph.build()

        graph.operations.append(self.layer_graph)
        graph.add_operation(
            self.layer_graph,
            self.weight_names
            + self.get_in_tensor_names(),
            self.get_out_tensor_names(),
        )


class Qwen2VLVisionEncoderATB(nn.Module):

    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        self.weights = weights
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.soc_info = NPUSocInfo()
        self.backend = self.soc_info.communication_backend
        self.layer_prefix = "visual.blocks"
        self.patch_embed_prefix = "visual.patch_embed"
        self.patch_merger_prefix = "visual.merger"
        layers = []
        for layer_idx in range(config.depth):
            layers.append(Qwen2VLVisionLayerATB(config, layer_idx, weights, self.layer_prefix, self.backend))
        self.layers = nn.ModuleList(layers)
        self.patch_embed = FastPatchEmbed(config, weights, self.patch_embed_prefix)
        self.patch_merger = PatchMerger(config, weights, self.patch_merger_prefix, self.backend)

        self.graph = None
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()

    def prepare_inputs(self, hidden_states, seqlens, cos_vit_mrope, sin_vit_mrope):
        self.graph_inputs.update({"patch_embed_in_tensor": hidden_states})
        self.graph_inputs.update({"seq_len": seqlens})
        self.graph_inputs.update({"cos_embedding": cos_vit_mrope})
        self.graph_inputs.update({"sin_embedding": sin_vit_mrope})
        self.graph_param["seq_len"] = seqlens.cpu().to(torch.int32)
        self.graph_outputs["patch_merger_out_tensor"] = torch.ones(
            1,
            hidden_states.shape[1] // PATCH_MERGER_FACTOR,
            self.config.hidden_size,
            dtype=hidden_states.dtype,
            device="npu"
        )

    def forward(
            self,
            hidden_states,
            seqlens,
            cos_vit_mrope,
            sin_vit_mrope
    ):

        self.prepare_inputs(hidden_states, seqlens, cos_vit_mrope, sin_vit_mrope)
        hidden_states = self.graph.forward(self.graph_inputs, self.graph_outputs, self.graph_param)
        return hidden_states

    def get_in_tensor_names(self):
        return ["patch_embed_in_tensor", "seq_len", "cos_embedding", "sin_embedding"]

    def get_out_tensor_names(self):
        return ["patch_merger_out_tensor"]

    def get_weights(self):
        weights_dict = OrderedDict()
        layer_idx = 0
        weights_dict.update(self.patch_embed.get_weights())
        for layer in self.layers:
            weights = layer.get_weights(f"{self.layer_prefix}.{layer_idx}")
            weights_dict.update(weights)
            layer_idx += 1
        weights_dict.update(self.patch_merger.get_weights())
        return weights_dict

    def build_graph(self):
        self.graph.add_input_output(
            input=list(self.weight.keys()) + self.get_in_tensor_names(),
            output=self.get_out_tensor_names()
        )
        self.patch_embed.build_graph(self.graph, self.get_in_tensor_names()[0], "hidden_states")
        for layer in self.layers:
            layer.build_graph(self.graph)
        self.patch_merger.build_graph(self.graph, "hidden_states", self.get_out_tensor_names()[0])

        self.graph.execute_as_single = False
        self.graph.build()
        self.graph.set_weights(self.weight)

    def init_graph(self):
        self.weight = self.get_weights()
        self.graph = AtbGraph("Qwen2VL_VIT_graph")
        self.build_graph()


class Qwen2VisionTransformerPretrainedModelATB(PreTrainedModel):

    def __init__(self, config, weights, max_seq_len) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.weights = weights
        self.head_dim = config.embed_dim // config.num_heads
        self.encoder = Qwen2VLVisionEncoderATB(config, self.weights)
        self.rotary_pos_emb = VisionRotaryEmbedding(
            weights.device,
            weights.dtype,
            self.head_dim // 2,
            seqlen=max_seq_len
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        cos = self.rotary_pos_emb.cos_cache[pos_ids].flatten(1)
        sin = self.rotary_pos_emb.sin_cache[pos_ids].flatten(1)
        cos = torch.nn.functional.pad(cos, (0, 24))
        sin = torch.nn.functional.pad(sin, (0, 24))
        cos_vit_mrope = cos.repeat(1, 2)
        sin_vit_mrope = sin.repeat(1, 2)
        return cos_vit_mrope, sin_vit_mrope

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        cos_vit_mrope, sin_vit_mrope = self.rot_pos_emb(grid_thw.to(torch.int32))
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        vision_features = self.encoder(
            hidden_states.unsqueeze(0),
            seqlens.to(torch.int32),
            cos_vit_mrope.unsqueeze(0),
            sin_vit_mrope.unsqueeze(0)
        )
        return vision_features[self.encoder.get_out_tensor_names()[0]].squeeze()
