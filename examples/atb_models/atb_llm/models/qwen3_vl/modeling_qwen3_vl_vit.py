# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, NormType
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import (
    ParallelType,
    TensorParallelInfo
)
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorReplicatedLinear,
)
from atb_llm.utils.quantize.quant_type import QuantType

OUT_DATA_TYPE = {
    torch.bfloat16: "ACL_BF16",
    torch.float16: "ACL_FLOAT16"
}
INITIAL_MAX_POS_EMBEDS = 8192
ATTN_PADDED_HEAD_DIM = 128


class LayerNorm(nn.Module):
    def __init__(self, weights, prefix, quantize=None):
        super().__init__()
        self.layer_norm_eps = 1e-6
        self.prefix = prefix
        self.weights = weights
        self.quantize = quantize == QuantType.W8A8
        weight = weights.get_tensor(f"{prefix}.weight")
        bias = weights.get_tensor(f"{prefix}.bias")
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict[f"{self.prefix}.weight"] = self.weight.data
        weights_dict[f"{self.prefix}.bias"] = self.bias.data
        if self.quantize:
            if "norm1" in self.prefix: 
                prefix = self.prefix.replace("norm1", "attn.qkv")
            elif "norm2" in self.prefix:
                prefix = self.prefix.replace("norm2", "mlp.linear_fc1")
            weights_dict[f"{self.prefix}.input_scale"] = self.weights.get_tensor(f"{prefix}.input_scale").data.npu()
            weights_dict[f"{self.prefix}.input_offset"] = \
                self.weights.get_tensor(f"{prefix}.input_offset").to(dtype=torch.int8).data.npu()
        return weights_dict

    def build_graph(self, graph, input_tensor_name, output_tensor_name):
        input_key_list = input_tensor_name + [f"{self.prefix}.weight", f"{self.prefix}.bias"]
        if self.quantize:
            input_key_list.extend([f"{self.prefix}.input_scale", f"{self.prefix}.input_offset"])
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_INT8" if self.quantize else "QUANT_UNDEFINED",
                "epsilon": self.layer_norm_eps,
                "beginParamsAxis": 1,
                "beginNormAxis": 1
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
            output_tensor_name
        )


class Qwen3VLVisionPatchEmbed(nn.Module):
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


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0, max_hw: int = INITIAL_MAX_POS_EMBEDS):
        super().__init__()
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="npu") / dim))
        self.max_hw = max_hw
        self.freqs = None
    
    def build_freq_table(self):
        seq = torch.arange(self.max_hw, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        self.freqs = torch.outer(seq, self.inv_freq)

    def forward(self, seqlen: int) -> torch.Tensor:
        if self.freqs is None or seqlen > self.max_hw:
            self.max_hw = seqlen
            self.build_freq_table()
        return self.freqs


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config, weights, prefix, norm_prefix, comm_backend):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.norm_prefix = norm_prefix
        self.comm_backend = comm_backend
        self.quantize = config.quantize == QuantType.W8A8
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.num_heads = config.num_heads
        self.embed_dim = config.hidden_size
        self.original_head_dim = self.embed_dim // self.num_heads
        self.padded_head_dim = ATTN_PADDED_HEAD_DIM
        self.pad_size = (self.padded_head_dim - self.original_head_dim) // 2
        self.scaling = self.original_head_dim ** -0.5
        self.num_heads_pre_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        if config.quantize == QuantType.W8A8SC:
            self.qkv = TensorParallelColumnLinear.load(
                config,
                prefix=f"{self.prefix}.qkv",
                weights=weights,
                bias=True,
            )
        else:
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
            gqa_size=self.original_head_dim,
        )

    def get_weights(self):
        weights_dict = OrderedDict()
        qkv_weights_dict = self.qkv.linear.get_weights(f"{self.prefix}.qkv")
        qkv_proj_weight = qkv_weights_dict[f"{self.prefix}.qkv.weight"]
        qkv_proj_bias = qkv_weights_dict[f"{self.prefix}.qkv.bias"]
        qkv_proj_weight_final = self._pad_qkv_weight(qkv_proj_weight)
        qkv_proj_bias_final = self._pad_qkv_bias(qkv_proj_bias)
        qkv_weights_dict[f"{self.prefix}.qkv.weight"] = qkv_proj_weight_final
        qkv_weights_dict[f"{self.prefix}.qkv.bias"] = qkv_proj_bias_final
        if self.quantize:
            qkv_proj_quant_bias = qkv_weights_dict[f"{self.prefix}.qkv.quant_bias"]
            qkv_proj_deq_scale = qkv_weights_dict[f"{self.prefix}.qkv.deq_scale"]
            qkv_proj_quant_bias_final = self._pad_qkv_bias(qkv_proj_quant_bias)
            qkv_proj_deq_scale_final = self._pad_qkv_bias(qkv_proj_deq_scale)
            qkv_weights_dict[f"{self.prefix}.qkv.quant_bias"] = qkv_proj_quant_bias_final
            qkv_weights_dict[f"{self.prefix}.qkv.deq_scale"] = qkv_proj_deq_scale_final

        out_proj_weights_dict = self.proj.linear.get_weights(f"{self.prefix}.proj")
        out_proj_weight = out_proj_weights_dict[f"{self.prefix}.proj.weight"]
        out_proj_weight_final = self._pad_out_weight(out_proj_weight)
        out_proj_weights_dict[f"{self.prefix}.proj.weight"] = out_proj_weight_final
        weights_dict.update(qkv_weights_dict)
        weights_dict.update(out_proj_weights_dict)
        return weights_dict
    
    def build_qkv_graph(self, graph):
        if self.quantize:
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
            op_name="block_qkv"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
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
        return [org_shape[0], self.num_heads_pre_rank, self.padded_head_dim]

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
            "enable_quant_input": self.quantize,
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
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(proj_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, proj_linear_tensor_map)
        
    def build_graph(self, graph):
        self.build_qkv_graph(graph)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph)
        self.build_proj_graph(graph)
        attn_res_add = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="block_attn_res_add",
        )
        graph.operations.append(attn_res_add)
        graph.add_operation(
            attn_res_add, ["hidden_states", "attn_proj_out"], ["hidden_states"]
        )

    def _pad_qkv_weight(self, weight):
        """
        Pad QKV weight tensor for dimension matching (static pad).
        """
        weight = torch.nn.functional.pad(
                weight.reshape(3, self.num_heads_pre_rank * 2, self.original_head_dim // 2, self.embed_dim),
                (0, 0, 0, self.pad_size)
            ).reshape(self.num_heads_pre_rank * self.padded_head_dim * 3, self.embed_dim)
        return weight

    def _pad_qkv_bias(self, bias):
        """
        Pad QKV bias tensor for dimension matching (static pad).
        """
        bias = torch.nn.functional.pad(
                bias.reshape(3, self.num_heads_pre_rank * 2, self.original_head_dim // 2),
                (0, self.pad_size)
            ).reshape(3 * self.num_heads_pre_rank * self.padded_head_dim)
        return bias

    def _pad_out_weight(self, weight):
        """
        Pad attention output weight tensor for dimension matching (static pad).
        """
        weight = torch.nn.functional.pad(
                weight.reshape(self.embed_dim, self.num_heads_pre_rank * 2, self.original_head_dim // 2),
                (0, self.pad_size)
            ).reshape(self.embed_dim, self.num_heads_pre_rank * self.padded_head_dim)
        return weight                 


class Qwen3VLVisionMlp(nn.Module):
    def __init__(self, config, weights, prefix, norm_prefix, comm_backend):
        super().__init__()
        self.norm_prefix = norm_prefix
        self.config = config
        self.weights = weights
        self.dtype = weights.dtype
        self.comm_backend = comm_backend
        self.prefix = prefix
        self.quantize = config.quantize == QuantType.W8A8
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.linear_fc1 = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.linear_fc1",
            weights=weights,
            bias=True
        )
        self.linear_fc2 = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.linear_fc2",
            weights=weights,
            bias=True
        )

    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.linear_fc1.linear.get_weights(f"{self.prefix}.linear_fc1"))
        weights_dict.update(self.linear_fc2.linear.get_weights(f"{self.prefix}.linear_fc2"))
        return weights_dict

    def build_fc1_graph(self, graph: atb.GraphOperation):
        if self.quantize:
            input_key_list = [f'{self.norm_prefix}_out', f"{self.prefix}.linear_fc1.weight",
                              f"{self.prefix}.linear_fc1.quant_bias", f"{self.prefix}.linear_fc1.deq_scale"]
            out_data_type = OUT_DATA_TYPE.get(self.dtype, "ACL_DT_UNDEFINED")
        else:
            input_key_list = [f'{self.norm_prefix}_out', f"{self.prefix}.linear_fc1.weight",
                              f"{self.prefix}.linear_fc1.bias"]
            out_data_type = "ACL_DT_UNDEFINED"
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True,
                "outDataType": out_data_type}),
            op_name="block_fc1_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            ['fc1_out']
        )

    def build_activation_graph(self, graph: atb.GraphOperation):
        act = atb.BaseOperation(
            op_type="Activation",
            op_param=json.dumps({"activationType": "ACTIVATION_GELU"}),
            op_name="block_activation",
        )
        graph.operations.append(act)
        graph.add_operation(
            act,
            ['fc1_out'],
            ['act_out'],
        )

    def build_fc2_graph(self, graph: atb.GraphOperation):
        fc2_param = {
            "op_name": "block_fc2_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_fc2.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        fc2_tensor_map = {
            "input": 'act_out',
            "linear_out": 'fc2_out',
        }
        fc2_parallel_param = {
            "op_name": "block_fc2_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.comm_backend
            ),
            "linear_param": fc2_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            fc2_parallel_param
        )
        graph = linear_parallel_builder.build(graph, fc2_tensor_map)

    def build_graph(self, graph):
        self.build_fc1_graph(graph)
        self.build_activation_graph(graph)
        self.build_fc2_graph(graph)
        mlp_res_add = atb.BaseOperation(op_type="Elewise", op_param=json.dumps({'elewiseType': 'ELEWISE_ADD'}),
                                        op_name='mlp_res_add')
        graph.operations.append(mlp_res_add)
        graph.add_operation(mlp_res_add, ['hidden_states', 'fc2_out'], ['layer_out'])


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, weights, prefix, comm_backend):
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.dtype = weights.dtype
        self.weight_names = None
        self.block_graph = None
        self.attn = Qwen3VLVisionAttention(
            config=config,
            weights=weights,
            prefix=f"{self.prefix}.attn",
            norm_prefix=f"{prefix}.norm1",
            comm_backend=comm_backend,
        )
        self.mlp = Qwen3VLVisionMlp(
            config=config,
            weights=weights,
            prefix=f"{prefix}.mlp",
            norm_prefix=f"{prefix}.norm2",
            comm_backend=comm_backend,
        )
        self.norm1 = LayerNorm(
            weights,
            f"{prefix}.norm1",
            quantize=config.quantize
        )
        self.norm2 = LayerNorm(
            weights,
            f"{prefix}.norm2",
            quantize=config.quantize
        )
        self.input_names = ["hidden_states", "seq_len", "cos_embedding", "sin_embedding"]
        self.output_names = ["hidden_states"]

    def get_weights(self):
        weights_dict = OrderedDict()
        for _, module in self.named_children():
            weights_dict.update(module.get_weights())
        self.weight_names = list(weights_dict.keys())
        return weights_dict
    

    def build_graph(self, graph):
        self.block_graph = AtbGraph("_".join(self.prefix.split(".")) + "_graph")
        self.block_graph.add_input_output(
            input=self.weight_names + self.input_names,
            output=["layer_out"]
        )
        self.norm1.build_graph(self.block_graph, ["hidden_states"], [f"{self.prefix}.norm1_out"])
        self.attn.build_graph(self.block_graph)
        self.norm2.build_graph(self.block_graph, ["hidden_states"], [f"{self.prefix}.norm2_out"])
        self.mlp.build_graph(self.block_graph)
        self.block_graph.build()
        graph.operations.append(self.block_graph)
        graph.add_operation(
            self.block_graph,
            self.weight_names + self.input_names,
            self.output_names,
        )


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config, weights, prefix, comm_backend, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.dtype = weights.dtype
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.comm_backend = comm_backend
        self.use_postshuffle_norm = use_postshuffle_norm
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.norm = LayerNorm(
            weights,
            f"{prefix}.norm",         
        )
        self.linear_fc1 = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.linear_fc1",
            weights=weights,
            bias=True
        )
        self.linear_fc2 = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.linear_fc2",
            weights=weights,
            bias=True
        )
    
    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.norm.get_weights())
        weights_dict.update(self.linear_fc1.linear.get_weights(f"{self.prefix}.linear_fc1"))
        weights_dict.update(self.linear_fc2.linear.get_weights(f"{self.prefix}.linear_fc2"))
        return weights_dict
    
    def build_fc1_graph(self, graph, input_tensor_name, output_tensor_name):
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="merger_fc1_linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            [input_tensor_name, f"{self.prefix}.linear_fc1.weight", f"{self.prefix}.linear_fc1.bias"],
            [output_tensor_name]
        )

    def build_activation_graph(self, graph, input_tensor_name, output_tensor_name):
        act = atb.BaseOperation(
            op_type="Activation",
            op_param=json.dumps({"activationType": "ACTIVATION_GELU"}),
            op_name="merger_activation",
        )
        graph.operations.append(act)
        graph.add_operation(
            act,
            [input_tensor_name],
            [output_tensor_name],
        )

    def build_fc2_graph(self, graph, input_tensor_name, output_tensor_name):
        fc2_param = {
            "op_name": "merger_fc2_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_fc2.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        fc2_tensor_map = {
            "input": input_tensor_name,
            "linear_out": output_tensor_name,
        }

        fc2_parallel_param = {
            "op_name": "merger_fc2_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(
                rank=self.tp_rank, world_size=self.tp_world_size, backend=self.comm_backend
            ),
            "linear_param": fc2_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(
            fc2_parallel_param
        )
        graph = linear_parallel_builder.build(graph, fc2_tensor_map)
    
    def reshape_out(self, org_shape):
        return [org_shape[0] * org_shape[1] // self.hidden_size, self.hidden_size]
    
    def reshape_back(self, org_shape):
        return [org_shape[0] * org_shape[1] // self.config.hidden_size, self.config.hidden_size]
    
    def build_graph(self, graph, input_tensor_name, output_tensor_name):
        if self.use_postshuffle_norm:
            graph.add_reshape(input_tensor_name[0], f"{self.prefix}.input_reshape", self.reshape_out)
            self.norm.build_graph(graph, [f"{self.prefix}.input_reshape"], [f"{self.prefix}.norm_out"])
        else:
            self.norm.build_graph(graph, [input_tensor_name[0]], [f"{self.prefix}.norm_out"])
        graph.add_reshape(f"{self.prefix}.norm_out", f"{self.prefix}.norm_out_reshape", self.reshape_out)
        self.build_fc1_graph(graph, f"{self.prefix}.norm_out_reshape", f"{self.prefix}.fc1_out")
        self.build_activation_graph(graph, f"{self.prefix}.fc1_out", f"{self.prefix}.activation_out")
        self.build_fc2_graph(graph, f"{self.prefix}.activation_out", output_tensor_name[0])
        if self.use_postshuffle_norm:
            graph.add_reshape(f"{self.prefix}.input_reshape", input_tensor_name[0], self.reshape_back)


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        if not hasattr(self.config, "quantize"):
            setattr(self.config, "quantize", None)
        if self.config.quantize == QuantType.W8A8_DYNAMIC:
            self.config.quantize = QuantType.W8A8
        if self.config.quantize == QuantType.W8A8SC:
            self.prefix = "visual"
        else:
            self.prefix = "model.visual"
        self.dtype = weights.dtype
        self.soc_info = NPUSocInfo()
        self.comm_backend = self.soc_info.communication_backend
        self.pos_emb_cache = dict()
        self.spatial_merge_size = self.config.spatial_merge_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(self.config, weights, f"{self.prefix}.patch_embed")
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.pos_embed.weight.data = weights.get_tensor(f"{self.prefix}.pos_embed.weight")
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        head_dim = self.config.hidden_size // self.config.num_heads
        self.rope_pad_size = (ATTN_PADDED_HEAD_DIM - head_dim) // 2
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(
            self.config,
            weights,
            f"{self.prefix}.blocks.{i}",
            self.comm_backend
        ) for i in range(self.config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(self.config, weights, f"{self.prefix}.merger", self.comm_backend)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([Qwen3VLVisionPatchMerger(
            self.config,
            weights,
            f"{self.prefix}.deepstack_merger_list.{i}",
            self.comm_backend,
            use_postshuffle_norm=True,
            
        ) for i in range(len(self.deepstack_visual_indexes))])
        self.graph = None
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()
    
    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.patch_embed.get_weights())
        for i, block in enumerate(self.blocks):
            weights = block.get_weights()
            weights_dict.update(weights)
            if i in self.deepstack_visual_indexes:
                deep_merger_index = self.deepstack_visual_indexes.index(i)
                deep_merger = self.deepstack_merger_list[deep_merger_index]
                weights_dict.update(deep_merger.get_weights())
        weights_dict.update(self.merger.get_weights())
        return weights_dict

    def init_graph(self):
        self.weight = self.get_weights()
        self.graph = AtbGraph("Qwen3VL_VIT_graph")
        self.build_graph()
    
    def get_in_tensor_names(self):
        return ["patch_embed_in_tensor", "patch_pos_embeds", "seq_len", "cos_embedding", "sin_embedding"]
    
    def get_out_tensor_names(self):
        return ["patch_merger_out_tensor", "deepstack_feature_0", "deepstack_feature_1", "deepstack_feature_2"]
    
    def build_graph(self):
        self.graph.add_input_output(
            input=list(self.weight.keys()) + self.get_in_tensor_names(),
            output=self.get_out_tensor_names()
        )
        self.patch_embed.build_graph(self.graph, ["patch_embed_in_tensor"], ["hidden_states"])
        pos_embed_add = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="pos_embed_add",
        )
        self.graph.operations.append(pos_embed_add)
        self.graph.add_operation(
            pos_embed_add, ["hidden_states", "patch_pos_embeds"], ["hidden_states"]
        )
        for layer_num, block in enumerate(self.blocks):
            block.build_graph(self.graph)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_index = self.deepstack_visual_indexes.index(layer_num)
                self.deepstack_merger_list[deepstack_merger_index].build_graph(
                    self.graph, ["hidden_states"], [f"deepstack_feature_{deepstack_merger_index}"])
        self.merger.build_graph(self.graph, ["hidden_states"], ["patch_merger_out_tensor"])
        self.graph.execute_as_single = False
        self.graph.build()
        self.graph.set_weights(self.weight)
    
    def prepare_inputs(self, hidden_states, seq_len, position_embeddings, adapted_pos_embed):
        self.graph_inputs.update({"patch_embed_in_tensor": hidden_states.to(self.dtype)})
        self.graph_inputs.update({"seq_len": seq_len.to(torch.int32).npu()})
        self.graph_inputs.update({"cos_embedding": position_embeddings[0].to(self.dtype).npu()})
        self.graph_inputs.update({"sin_embedding": position_embeddings[1].to(self.dtype).npu()})
        self.graph_inputs.update({"patch_pos_embeds": adapted_pos_embed.to(self.dtype).npu()})
        self.graph_param["seq_len"] = seq_len.to(torch.int32)
        self.graph_outputs["patch_merger_out_tensor"] = torch.ones(
            seq_len.sum().item() // (self.spatial_merge_size ** 2),
            self.config.out_hidden_size,
            dtype=self.dtype,
            device="npu"
        )
        for i in range(len(self.deepstack_visual_indexes)):
            key = f"deepstack_feature_{i}"
            self.graph_outputs[key] = torch.ones(
                seq_len.sum().item() // (self.spatial_merge_size ** 2),
                self.config.out_hidden_size,
                dtype=self.dtype,
                device="npu"
            )
    
    @lru_cache(maxsize=128)
    def get_patch_pos_embed(self, grid_thw):
        grid_thw = torch.tensor(np.array(json.loads(grid_thw)))
        patch_pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        seq_len, _ = patch_pos_embeds.size()
        rotary_pos_emb = self._rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        return seqlens, position_embeddings, patch_pos_embeds

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw = json.dumps(grid_thw.tolist())
        seqlens, position_embeddings, patch_pos_embed = self.get_patch_pos_embed(grid_thw)
        self.prepare_inputs(hidden_states, seqlens, position_embeddings, patch_pos_embed)
        graph_out = self.graph.forward(self.graph_inputs, self.graph_outputs, self.graph_param)
        hidden_states = graph_out["patch_merger_out_tensor"]
        deepstack_feature_0 = graph_out["deepstack_feature_0"]
        deepstack_feature_1 = graph_out["deepstack_feature_1"]
        deepstack_feature_2 = graph_out["deepstack_feature_2"]
        return hidden_states, [deepstack_feature_0, deepstack_feature_1, deepstack_feature_2]

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary positional embeddings for spatiotemporal grid positions.
        
        This method generates rotary positional embeddings for tokens arranged in a 3D grid
        (Time × Height × Width), handling both spatial and temporal dimensions with proper
        coordinate mapping and block merging for efficiency.

        """
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        embeddings = torch.nn.functional.pad(embeddings, (0, self.rope_pad_size))
        return embeddings

    def _fast_pos_embed_interpolate(self, grid_thw):
        """
        Efficiently interpolates positional embeddings for spatiotemporal grids using cached embeddings.
        
        This method provides fast positional embedding generation by leveraging pre-computed embeddings
        from the cache and applying spatial-temporal transformations for batched grid inputs.

        """
        device = self.pos_embed.weight.device
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        all_indices, all_weights = self._batch_compute_grid_interpolation(grid_hs.numpy(), grid_ws.numpy())
        idx_tensor = torch.tensor(all_indices, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(all_weights, dtype=self.pos_embed.weight.dtype, device=device)

        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds.sum(dim=0)

        split_sizes = [h * w for h, w in zip(grid_hs, grid_ws)]
        patch_pos_embeds_list = torch.split(patch_pos_embeds, split_sizes, dim=0)

        patch_pos_embeds_permute = []

        for pos_embed, t, h, w in zip(patch_pos_embeds_list, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t), int(h), int(w)
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // self.spatial_merge_size, self.spatial_merge_size,
                            w // self.spatial_merge_size, self.spatial_merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)
    
    def _batch_compute_grid_interpolation(self, grid_hs, grid_ws):
        """
        Batch computation of bilinear interpolation indices and weights.
        
        For each grid size in the batch, computes the four corner indices and weights
        needed for bilinear interpolation from a base grid to the target resolution.

        Args:
            grid_hs: Iterable of target grid heights
            grid_ws: Iterable of target grid widths
        """
        all_h_coords = []
        all_w_coords = []
        for h, w in zip(grid_hs, grid_ws):
            h_coords = np.linspace(0, self.num_grid_per_side - 1, h, dtype=np.float32)
            w_coords = np.linspace(0, self.num_grid_per_side - 1, w, dtype=np.float32)
            h_grid, w_grid = np.meshgrid(h_coords, w_coords, indexing='ij')
            all_h_coords.append(h_grid.ravel())
            all_w_coords.append(w_grid.ravel())

        h_coords = np.concatenate(all_h_coords)
        w_coords = np.concatenate(all_w_coords)

        h_floor = np.floor(h_coords).astype(np.int32)
        w_floor = np.floor(w_coords).astype(np.int32)
        h_ceil = np.minimum(h_floor + 1, self.num_grid_per_side - 1)
        w_ceil = np.minimum(w_floor + 1, self.num_grid_per_side - 1)
        dh = h_coords - h_floor
        dw = w_coords - w_floor

        weights = np.vstack([
            (1 - dh) * (1 - dw),
            (1 - dh) * dw,
            dh * (1 - dw),
            dh * dw
        ])
        indices = np.vstack([
            h_floor * self.num_grid_per_side + w_floor,
            h_floor * self.num_grid_per_side + w_ceil,
            h_ceil * self.num_grid_per_side + w_floor,
            h_ceil * self.num_grid_per_side + w_ceil
        ])
        return indices, weights
