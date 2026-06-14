# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#          http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement part of this file based on transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
from collections import OrderedDict, defaultdict
from functools import lru_cache
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_npu
import _libatb_torch as atb

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, NormType, ActivationType
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import (
    ParallelType,
    TensorParallelInfo
)
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.models.base.model_utils import MlpLinearInfo, AttnLinearInfo, LinearInfo
from atb_llm.models.base.modeling_atb import BaseRMSNorm, BaseMLP
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorReplicatedLinear,
    load_column_multi,
)


A2_SOCS = (220, 221, 222, 223, 224, 225)
A3_SOCS = (250, 251, 252, 253, 254, 255)
INITIAL_MAX_GRID_SIZE = 8192


class Glm41vVisionPatchEmbed(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.prefix = prefix
        self.dtype = weights.dtype
        self.embed_dim = config.hidden_size
        self.proj = TensorReplicatedLinear.load(
            config,
            prefix=f"{self.prefix}.proj",
            weights=weights,
            bias=True
        )
        self.proj.linear.weight.data = \
            self.proj.linear.weight.data.view(self.embed_dim, -1)
    
    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.proj.linear.get_weights(f"{self.prefix}.proj"))
        return weights_dict
    
    def build_graph(self, graph, input_tensor_name, output_tensor_name):
        out_data_type = "ACL_DT_UNDEFINED"
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True,
                "outDataType": out_data_type}),
            op_name="patch_embed"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_tensor_name + [f"{self.prefix}.proj.weight", f"{self.prefix}.proj.bias"],
            output_tensor_name
        )


class Glm41vRMSNorm(BaseRMSNorm):
    def __init__(self, prefix, config, weights, linear_info):
        super().__init__(prefix, config, weights, linear_info)
        self.has_bias = False
        self.bias = None


class Glm41vVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_grid_size = INITIAL_MAX_GRID_SIZE
        self.freqs = None

    def build_freq_table(self):
        seq = torch.arange(self.max_grid_size, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        self.freqs = torch.outer(seq, self.inv_freq)

    def forward(self, seqlen: int) -> torch.Tensor:
        if self.freqs is None or seqlen > self.max_grid_size:
            self.max_grid_size = max(seqlen, self.max_grid_size)
            self.build_freq_table()
        return self.freqs


class Glm41vVisionEmbeddings(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.soc_info = NPUSocInfo()
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_embedding.weight.data = weights.get_tensor(f"{prefix}.position_embedding.weight")
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, lengths, image_shapes, h_coords, w_coords) -> torch.Tensor:
        # Get position embedding parameters
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        dtype = pos_embed_weight.dtype
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.numpy(), w_coords.numpy()
        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(0, hidden_size, device=device, dtype=dtype)
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = np.array(lengths, dtype=np.int64)
            if not isinstance(image_shapes, np.ndarray):
                image_shapes = np.array(image_shapes, dtype=np.int64)
            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            # Calculate target dimensions for each patch
            target_h = np.concatenate(
                [np.repeat(image_shapes[i, 1], lengths[i]) for i in range(len(lengths))]
            ).astype(np.float32)
            target_w = np.concatenate(
                [np.repeat(image_shapes[i, 2], lengths[i]) for i in range(len(lengths))]
            ).astype(np.float32)
            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.astype(np.float32)
            w_coords = w_coords.astype(np.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1
            # Create sampling grid
            grid = np.stack((norm_w, norm_h), axis=-1)
            grid = torch.tensor(grid, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)
            # Perform bicubic interpolation
            if self.soc_info.soc_version in A2_SOCS + A3_SOCS:
                interpolated_embed_fp32 = F.grid_sample(
                    pos_embed_2d, grid, mode="bicubic", align_corners=False, padding_mode="border"
                )
            else:
                # GridSample2D in bicubic mode is not supported in 300I DUO cards
                upsampled = F.interpolate(pos_embed_2d, scale_factor=2, mode='bilinear', align_corners=False)
                interpolated_embed_fp32 = F.grid_sample(upsampled, grid, 
                                                        mode='bilinear', align_corners=False,
                                                        padding_mode='zeros')
            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            adapted_pos_embed = adapted_pos_embed_fp32.to(dtype)

        return adapted_pos_embed


class Glm41vVisionAttention(nn.Module):
    def __init__(self, config, weights, prefix, norm_prefix, comm_backend):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.prefix = prefix
        self.norm_prefix = norm_prefix
        self.dtype = weights.dtype
        self.comm_backend = comm_backend
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_heads = config.num_heads
        self.num_heads_pre_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{self.prefix}.qkv",
            weights=weights,
            bias=False,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
        )
        self.proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{self.prefix}.proj",
            weights=weights,
            bias=False,
            gqa_size=self.head_dim,
        )
        self.linear_info = AttnLinearInfo()
        self.linear_info.is_pack = True
        self.linear_info.is_all_float = True
        self.linear_info.pack_linear = self.qkv.linear
        self.linear_info.dense_linear = self.proj.linear
        
    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        qkv_weights_dict = self.qkv.linear.get_weights(f"{prefix}.qkv")
        out_proj_weights_dict = self.proj.linear.get_weights(f"{prefix}.proj")
        weights_dict.update(qkv_weights_dict)
        weights_dict.update(out_proj_weights_dict)
        return weights_dict
    
    def build_qkv_graph(self, graph):
        out_data_type = "ACL_DT_UNDEFINED"
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": False,
                "outDataType": out_data_type}),
            op_name="block_qkv"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            [f"{self.norm_prefix}_out", f"{self.prefix}.qkv.weight"],
            ["qkv_out"]
        )
        split_op = atb.BaseOperation(
            op_type="Split",
            op_param=json.dumps({
                "splitDim": -1,
                "splitNum": 3
            }),
            op_name="block_qkv_split"
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            ["qkv_out"],
            ["q", "k", "v"],
        )

    def build_rope_graph(self, graph):
        rope_param = {
            "op_name": "block_rope",
            "head_num": self.num_heads_pre_rank,
            "kv_head_num": self.num_heads_pre_rank,
            "category": CommonOpBuilderType.ROPE,
            "is_fa": False,
            "atb_rope_param": {"rotaryCoeff": 2},
        }
        rope_tensor_map = {
            "q": "q",
            "k": "k",
            "cos_embedding": "cos_embedding",
            "sin_embedding": "sin_embedding",
            "seq_len": "seq_len",
            "q_out": "q",
            "k_out": "k",
        }
        rope_builder = CommonOpBuilderManager.get_builder(rope_param)
        graph = rope_builder.build(graph, rope_tensor_map)

    def reshape_qkv(self, org_shape):
        return [org_shape[0], self.num_heads_pre_rank, self.head_dim]

    def reshape_out(self, org_shape):
        return [org_shape[0], org_shape[1] * org_shape[2]]

    def build_attention_graph(self, graph):
        attention_op = atb.BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps({
                "headNum": self.num_heads_pre_rank,
                "kvHeadNum": self.num_heads_pre_rank,
                "qkScale": self.scaling,
                "calcType": "PA_ENCODER"}),
            op_name="block_selfattention"
        )
        graph.add_reshape("q", "q_reshape", self.reshape_qkv)
        graph.add_reshape("k", "k_reshape", self.reshape_qkv)
        graph.add_reshape("v", "v_reshape", self.reshape_qkv)
        input_key_list = ["q_reshape", "k_reshape", "v_reshape", "seq_len"]
        output_key_list = ["attn_out"]
        graph.operations.append(attention_op)
        graph.add_operation(attention_op, input_key_list, output_key_list)
        graph.add_reshape("attn_out", "attn_out_reshape", self.reshape_out)
        
    def build_proj_graph(self, graph):
        proj_linear_param = {
            "op_name": "block_proj",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.proj.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        proj_linear_tensor_map = {
            "input": "attn_out_reshape",
            "linear_out": "attn_proj_out"
        }
        proj_linear_parallel_param = {
            "op_name": "block_proj_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.comm_backend),
            "linear_param": proj_linear_param,
            "enable_lcoc": self.comm_backend == "lccl",
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(proj_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, proj_linear_tensor_map)
        
    def build_graph(self, graph):
        self.build_qkv_graph(graph)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph)
        self.build_proj_graph(graph)


class Glm41vVisionMlp(BaseMLP):
    def __init__(self, config, weights, prefix, norm_prefix, comm_backend):
        super().__init__(prefix, config, weights, norm_prefix, comm_backend)
        if config.quantize == "w8a8sc":
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_up_proj",
                weights=weights,
                bias=False
            )
        else:
            self.gate_up_proj = load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1,
                bias=False
            )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.linear_info.is_pack = True
        self.linear_info.is_all_float = True
        self.linear_info.pack_linear = self.gate_up_proj.linear
        self.linear_info.down_linear = self.down_proj.linear
    
    def build_activation_graph(self, graph: atb.GraphOperation):
        act_param = {
            "op_name": "activation",
            "category": CommonOpBuilderType.ACTIVATION,
            "is_pack": self.linear_info.is_pack,
            "up_weight_only": self.linear_info.up_weight_only,
            "activation_type": ActivationType.SWIGLU
        }
        act_tensor_map = {
            "input": 'gate_up_out',
            "act_out": 'mul_out'
        }
        if not self.linear_info.is_pack:
            act_tensor_map.update({"other_input": 'up_out'})
        act_builder = CommonOpBuilderManager.get_builder(act_param)
        graph = act_builder.build(graph, act_tensor_map)

    def build_down_graph(self, graph: atb.GraphOperation, is_prefill: bool):
        down_linear_param = {
            "op_name": "down_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_info.down_linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0
        }
        down_linear_tensor_map = {
            "input": 'mul_out',
            "linear_out": 'mlp_out'
        }

        down_linear_parallel_param = {
            "op_name": "down_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": down_linear_param,
            "enable_lcoc": self.backend == "lccl",
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(down_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, down_linear_tensor_map)


class Glm41vVisionBlock(nn.Module):
    def __init__(self, config, weights, prefix, comm_backend):
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.dtype = weights.dtype
        self.weight_names = None
        self.block_graph = None
        self.attn = Glm41vVisionAttention(
            config=config,
            weights=weights,
            prefix=f"{self.prefix}.attn",
            norm_prefix=f"{prefix}.norm1",
            comm_backend=comm_backend,
        )
        self.mlp = Glm41vVisionMlp(
            config=config,
            weights=weights,
            prefix=f"{prefix}.mlp",
            norm_prefix=f"{prefix}.norm2",
            comm_backend=comm_backend,
        )
        self.norm1 = Glm41vRMSNorm(
            f"{prefix}.norm1",
            config,
            weights,
            self.attn.linear_info
        )
        self.norm2 = Glm41vRMSNorm(
            f"{prefix}.norm2",
            config,
            weights,
            self.mlp.linear_info
        )
        self.input_names = ["hidden_states", "seq_len", "cos_embedding", "sin_embedding"]
        self.output_names = ["hidden_states"]

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict
    
    def build_add_norm_graph(self, graph):
        add_norm_op = atb.BaseOperation(
            op_type="AddRmsNorm",
            op_param=json.dumps({"epsilon": self.config.rms_norm_eps}),
            op_name="block_add_rms_norm",
        )
        graph.operations.append(add_norm_op)
        graph.add_operation(
            add_norm_op,
            ["attn_proj_out", "hidden_states", f"{self.norm2.prefix}.weight"],
            [f"{self.prefix}.norm2_out", f"{self.prefix}.norm2_rstdout", "hidden_states"]
        )

    def build_graph(self, graph):
        self.block_graph = AtbGraph("_".join(self.prefix.split(".")) + "_graph")
        self.block_graph.add_input_output(
            input=self.weight_names + self.input_names,
            output=["layer_out"]
        )
        self.norm1.build_graph(self.block_graph, is_prefill=True)
        self.attn.build_graph(self.block_graph)
        self.build_add_norm_graph(self.block_graph)
        self.mlp.build_graph(self.block_graph, is_prefill=False)
        self.block_graph.build()
        graph.operations.append(self.block_graph)
        graph.add_operation(
            self.block_graph,
            self.weight_names + self.input_names,
            self.output_names,
        )


class LayerNorm(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.config = config
        self.prefix = prefix
        weight = weights.get_tensor(f"{prefix}.weight")
        bias = weights.get_tensor(f"{prefix}.bias")
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict[f"{prefix}.weight"] = self.weight.data
        weights_dict[f"{prefix}.bias"] = self.bias.data
        return weights_dict

    def build_graph(self, graph, input_tensor_name, output_tensor_name):
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_UNDEFINED",
                "epsilon": self.config.rms_norm_eps,
                "beginParamsAxis": 1,
                "beginNormAxis": 1
            },
        }
        norm_op = atb.BaseOperation(
            op_type=NormType.LAYERNORM,
            op_param=json.dumps(norm_param),
            op_name="layernorm",
        )
        graph.operations.append(norm_op)
        graph.add_operation(
            norm_op, input_tensor_name + [f"{self.prefix}.weight", f"{self.prefix}.bias"], output_tensor_name
        )


class Glm41vVisionPatchMerger(nn.Module):
    def __init__(self, config, weights, prefix, comm_backend) -> None:
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.dtype = weights.dtype
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.comm_backend = comm_backend
        self.proj = TensorReplicatedLinear.load(
            config,
            prefix=f"{self.prefix}.proj",
            weights=weights,
            bias=False
        )
        self.post_projection_norm = LayerNorm(config, weights, f"{prefix}.post_projection_norm")
        self.linear_info = MlpLinearInfo()
        if config.quantize == "w8a8sc":
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_up_proj",
                weights=weights,
                bias=False
            )
        else:
            self.gate_up_proj = load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1,
            )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.linear_info.is_pack = True
        self.linear_info.pack_linear = self.gate_up_proj.linear
        self.linear_info.down_linear = self.down_proj.linear
        self.linear_info.is_all_float = True

    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.proj.linear.get_weights(f"{self.prefix}.proj"))
        weights_dict.update(self.post_projection_norm.get_weights(f"{self.prefix}.post_projection_norm"))
        weights_dict.update(self.gate_up_proj.linear.get_weights(f"{self.prefix}.gate_up_proj"))
        weights_dict.update(self.down_proj.linear.get_weights(f"{self.prefix}.down_proj"))
        return weights_dict
    
    def build_gateup_graph(self, graph):
        linear_param = {
            "op_name": "patch_merger_gate_up_linear",
            "category": CommonOpBuilderType.LINEAR,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0
        }
        linear_param.update({"linear_module": self.linear_info.pack_linear})
        gate_up_linear_param = {
            "op_name": "patch_merger_gate_up_linear",
            "category": CommonOpBuilderType.GATE_UP,
            "is_pack": self.linear_info.is_pack,
            "linear_param": linear_param
        }
        gate_up_linear_tensor_map = {
            "input": "patch_merger_gate_up_in",
            "gate_up_out": 'patch_merger_gate_up_out'
        }
        builder = CommonOpBuilderManager.get_builder(gate_up_linear_param)
        graph = builder.build(graph, gate_up_linear_tensor_map)

    def build_activation_graph(self, graph):
        act_param = {
            "op_name": "patch_merger_activation",
            "category": CommonOpBuilderType.ACTIVATION,
            "is_pack": self.linear_info.is_pack,
            "up_weight_only": False,
            "activation_type": ActivationType.SWIGLU
        }
        act_tensor_map = {
            "input": 'patch_merger_gate_up_out',
            "act_out": 'patch_merger_mul_out'
        }
        act_builder = CommonOpBuilderManager.get_builder(act_param)
        graph = act_builder.build(graph, act_tensor_map)

    def build_down_graph(self, graph):
        down_linear_param = {
            "op_name": "patch_merger_down_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_info.down_linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0
        }
        down_linear_tensor_map = {
            "input": 'patch_merger_mul_out',
            "linear_out": 'patch_merger_out_tensor'
        }

        down_linear_parallel_param = {
            "op_name": "patch_merger_down_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.comm_backend),
            "linear_param": down_linear_param,
            "enable_lcoc": self.comm_backend == "lccl",
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(down_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, down_linear_tensor_map)
    
    def build_graph(self, graph):
        out_data_type = "ACL_DT_UNDEFINED"
        proj = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": False,
                "outDataType": out_data_type}),
            op_name="patch_merger_proj"
        )
        graph.operations.append(proj)
        graph.add_operation(
            proj,
            ["patch_merger_proj_in", f"{self.prefix}.proj.weight"],
            ["patch_merger_proj_out"]
        )
        self.post_projection_norm.build_graph(
            graph,
            ["patch_merger_proj_out"],
            [f"{self.prefix}.post_projection_norm_out"]
        )
        act = atb.BaseOperation(
            op_type="Activation",
            op_param=json.dumps({
                "activationType": "ACTIVATION_GELU",
                "geluMode": "NONE_MODE"
            }),
            op_name="patch_embed_act"
        )
        graph.operations.append(act)
        graph.add_operation(
            act,
            [f"{self.prefix}.post_projection_norm_out"],
            ["patch_merger_gate_up_in"]
        )
        self.build_gateup_graph(graph)
        self.build_activation_graph(graph)
        self.build_down_graph(graph)


class Glm41vVisionModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        if not hasattr(self.config, "quantize"):
            setattr(self.config, "quantize", None)
        if config.quantize == "w8a8sc":
            self.prefix = "visual"
        else:
            self.prefix = "model.visual"
        self.dtype = weights.dtype
        self.soc_info = NPUSocInfo()
        self.comm_backend = self.soc_info.communication_backend
        self.pos_emb_cache = dict()
        self.spatial_merge_size = self.config.spatial_merge_size
        self.patch_embed = Glm41vVisionPatchEmbed(self.config, weights, f"{self.prefix}.patch_embed")
        norm_info = LinearInfo(is_pack=False, is_all_float=True)
        self.post_conv_layernorm = Glm41vRMSNorm(f"{self.prefix}.post_conv_layernorm", self.config, weights, norm_info)
        head_dim = self.config.hidden_size // self.config.num_heads
        self.rotary_pos_emb = Glm41vVisionRotaryEmbedding(head_dim // 2)
        self.embeddings = Glm41vVisionEmbeddings(self.config, weights, f"{self.prefix}.embeddings")
        self.blocks = nn.ModuleList([Glm41vVisionBlock(
            self.config,
            weights,
            f"{self.prefix}.blocks.{i}",
            self.comm_backend
        ) for i in range(self.config.depth)])
        self.post_layernorm = Glm41vRMSNorm(f"{self.prefix}.post_layernorm", self.config, weights, norm_info)
        self.merger = Glm41vVisionPatchMerger(self.config, weights, f"{self.prefix}.merger", self.comm_backend)
        self.downsample = TensorReplicatedLinear.load(
            self.config,
            prefix=f"{self.prefix}.downsample",
            weights=weights,
            bias=True
        )
        self.downsample.linear.weight.data = \
            self.downsample.linear.weight.data.view(self.config.out_hidden_size, -1)

        self.graph = None
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()

    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.patch_embed.get_weights())
        weights_dict.update(self.post_conv_layernorm.get_weights(f"{self.prefix}.post_conv_layernorm"))
        for i, block in enumerate(self.blocks):
            weights = block.get_weights(f"{self.prefix}.blocks.{i}")
            weights_dict.update(weights)
        weights_dict.update(self.post_layernorm.get_weights(f"{self.prefix}.post_layernorm"))
        weights_dict.update(self.downsample.linear.get_weights(f"{self.prefix}.downsample"))
        weights_dict.update(self.merger.get_weights())
        return weights_dict
    
    def init_graph(self):
        self.weight = self.get_weights()
        self.graph = AtbGraph("Glm41v_VIT_graph")
        self.build_graph()

    def get_in_tensor_names(self):
        return ["patch_embed_in_tensor", "adapted_pos_embed", "seq_len", "cos_embedding", "sin_embedding"]

    def get_out_tensor_names(self):
        return ["patch_merger_out_tensor"]
    
    def reshape_in(self, org_shape):
        return [org_shape[0] // (self.spatial_merge_size ** 2), self.spatial_merge_size,
                self.spatial_merge_size, org_shape[1]]

    def reshape_out(self, org_shape):
        return [org_shape[0], org_shape[1] * org_shape[2] * org_shape[3]]
    
    def build_graph(self):
        self.graph.add_input_output(
            input=list(self.weight.keys()) + self.get_in_tensor_names(),
            output=self.get_out_tensor_names()
        )
        self.patch_embed.build_graph(self.graph, ["patch_embed_in_tensor"], ["hidden_states"])
        self.post_conv_layernorm.build_graph(self.graph, is_prefill=True)
        pos_embed_add = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="pos_embed_add",
        )
        self.graph.operations.append(pos_embed_add)
        self.graph.add_operation(
            pos_embed_add, [f"{self.prefix}.post_conv_layernorm_out", "adapted_pos_embed"], ["hidden_states"]
        )
        for block in self.blocks:
            block.build_graph(self.graph)
        self.post_layernorm.build_graph(self.graph, is_prefill=True)
        self.graph.add_reshape(f"{self.prefix}.post_layernorm_out", "post_layernorm_out_reshape", self.reshape_in)
        transpose_op = atb.BaseOperation(
            op_type="Transpose",
            op_param=json.dumps({"perm": [0, 3, 1, 2]}),
            op_name="downsample_transpose",
        )
        self.graph.operations.append(transpose_op)
        self.graph.add_operation(
            transpose_op,
            ["post_layernorm_out_reshape"],
            ["downsample_in"]
        )
        self.graph.add_reshape("downsample_in", "downsample_in_reshape", self.reshape_out)
        out_data_type = "ACL_DT_UNDEFINED"
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True,
                "outDataType": out_data_type}),
            op_name="downsample"
        )
        self.graph.operations.append(linear_op)
        self.graph.add_operation(
            linear_op,
            ["downsample_in_reshape", f"{self.prefix}.downsample.weight", f"{self.prefix}.downsample.bias"],
            ["patch_merger_proj_in"]
        )
        self.merger.build_graph(self.graph)
        self.graph.execute_as_single = True
        self.graph.build()
        self.graph.set_weights(self.weight)

    def prepare_inputs(self, hidden_states, seqlens, position_embeddings, adapted_pos_embed):
        self.graph_inputs.update({"patch_embed_in_tensor": hidden_states.to(self.dtype)})
        self.graph_inputs.update({"seq_len": seqlens.to(torch.int32).npu()})
        self.graph_inputs.update({"cos_embedding": position_embeddings[0].to(self.dtype).npu()})
        self.graph_inputs.update({"sin_embedding": position_embeddings[1].to(self.dtype).npu()})
        self.graph_inputs.update({"adapted_pos_embed": adapted_pos_embed.to(self.dtype).npu()})
        self.graph_param["seq_len"] = seqlens.to(torch.int32)
        self.graph_outputs["patch_merger_out_tensor"] = torch.ones(
            seqlens.sum().item() // (self.spatial_merge_size ** 2),
            self.config.out_hidden_size,
            dtype=self.dtype,
            device="npu"
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
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb, pos_ids
    
    @lru_cache(maxsize=128)
    def get_adapted_pos_embed(self, grid_thw):
        grid_thw = torch.tensor(np.array(json.loads(grid_thw)))
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.npu()
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        adapted_pos_embed = self.embeddings(seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1])
        return seqlens, position_embeddings, adapted_pos_embed

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw = json.dumps(grid_thw.tolist())
        seqlens, position_embeddings, adapted_pos_embed = self.get_adapted_pos_embed(grid_thw)
        self.prepare_inputs(hidden_states, seqlens, position_embeddings, adapted_pos_embed)
        graph_out = self.graph.forward(self.graph_inputs, self.graph_outputs, self.graph_param)
        hidden_states = graph_out["patch_merger_out_tensor"]
        return hidden_states