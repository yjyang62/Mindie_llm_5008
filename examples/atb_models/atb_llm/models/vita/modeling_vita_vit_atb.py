# coding=utf-8
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Implement InternRMSNorm based on InternRMSNorm from OpenGVLab/InternVL2-8B
# Implement InternVisionEmbeddings based on InternVisionEmbeddings from OpenGVLab/InternVL2-8B
# Implement InternAttention based on InternAttention from OpenGVLab/InternVL2-8B
# Implement InternMLP based on InternMLP from OpenGVLab/InternVL2-8B
# Implement InternVisionEncoderLayer based on InternVisionEncoderLayer from OpenGVLab/InternVL2-8B
# Implement InternVisionEncoder based on InternVisionEncoder from OpenGVLab/InternVL2-8B
# Implement InternVisionModel based on InternVisionModel from OpenGVLab/InternVL2-8B
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Optional
from collections import OrderedDict, defaultdict
import json
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from atb_llm.utils.initial import NPUSocInfo
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
import _libatb_torch as atb
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.common_op_builders.data_type import CommonOpBuilderType, NormType
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import ParallelType, \
    TensorParallelInfo, CommunicationBackend
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.utils.layers import TensorParallelRowLinear, TensorParallelColumnLinear
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from .configuration_vita_vit import InternVisionConfig

logger = logging.get_logger(__name__)
GRAPH_KEY = "VIT"


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm.')
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning('Discovered apex but it failed to load, falling back to InternRMSNorm.')
    pass


NORM2FN = {
    'rms_norm': InternRMSNorm,
    'layer_norm': nn.LayerNorm,
}


class LayerNormATB(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.layer_norm_eps = 1e-6
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

    def build_graph(self, graph):
        norm_param = {
            "layerType": "LAYER_NORM_NORM",
            "normParam": {
                "quantType": "QUANT_UNDEFINED",
                "epsilon": self.layer_norm_eps,
                "beginParamsAxis": 2,
                "beginNormAxis": 2
            },
        }
        norm_op = atb.BaseOperation(
            op_type=NormType.LAYERNORM,
            op_param=json.dumps(norm_param),
            op_name="norm",
        )
        graph.operations.append(norm_op)
        graph.add_operation(
            norm_op, ["hidden_states", f"{self.prefix}.weight", f"{self.prefix}.bias"], [f"{self.prefix}_out"]
        )


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings
    
    def _get_pos_embed(self, pos_embed, h, w):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(h, w), mode="bicubic", align_corners=False)
            .reshape(1, -1, h * w)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig, weights, layer_id, backend, prefix):
        super().__init__()
        self.config = config
        self.weights = weights
        self.layer_id = layer_id
        self.backend = backend

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        setattr(config, 'quantize', None)
        self.quantize = config.quantize
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_heads_pre_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        if self.head_dim * self.num_heads != self.embed_dim:
            logger.error(
                f'`embed_dim` must be divisible by `num_heads` (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).',
                ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
            )
            raise ValueError(
                f'`embed_dim` must be divisible by `num_heads` (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.prefix = f"{prefix}.attn"
        self.norm_prefix = f"{prefix}.norm1"


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
                gqa_size=self.head_dim,
        )

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.qkv.linear.get_weights(f"{prefix}.qkv"))

        weights_dict.update(self.proj.linear.get_weights(f"{prefix}.proj"))
        return weights_dict
    
    def build_qkv_graph(self, graph):
        input_key_list = [f"{self.norm_prefix}_out", f"{self.prefix}.qkv.weight", f"{self.prefix}.qkv.bias"]
        linear_out = ["qkv_linear_out"]
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="qkv" + "_Linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            linear_out
        )
        split_op = atb.BaseOperation(
            op_type="Split",
            op_param=json.dumps({
                "splitDim": -1,
                "splitNum": 3
            }),
            op_name="qkv" + "_Split"
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            ["qkv_linear_out"],
            ["q_split", "k_split", "v_split"],
        )

    # (B,S,H) ->(B*S,N,D) ->(B,S,H)
    def reshape_q(self, org_shape):
        self.org_shape_0 = org_shape[0]
        self.org_shape_1 = org_shape[1]
        return [org_shape[0] * org_shape[1], self.num_heads_pre_rank, self.head_dim]
    
    def reshape_kv(self, org_shape):
        return [org_shape[0] * org_shape[1], self.num_heads_pre_rank, self.head_dim]
    
    def reshape_out(self, org_shape):

        return [self.org_shape_0, self.org_shape_1, org_shape[1] * org_shape[2]]

    def build_attention_graph(self, graph):
        attention_op = atb.BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps({
                "headNum": self.num_heads_pre_rank,
                "kvHeadNum": self.num_heads_pre_rank,
                "qkScale": 1.0 / math.sqrt(self.head_dim),
                "calcType": "PA_ENCODER"}),
            op_name="selfattention"
        )
        graph.add_reshape("q_split", "q_split_reshape", self.reshape_q)
        graph.add_reshape("k_split", "k_split_reshape", self.reshape_kv)
        graph.add_reshape("v_split", "v_split_reshape", self.reshape_kv)

        graph.operations.append(attention_op)
        input_key_list = ["q_split_reshape", "k_split_reshape", "v_split_reshape", "seq_len"]
        output_key_list = ["atten_out"]
        graph.add_operation(attention_op, input_key_list, output_key_list)
        graph.add_reshape("atten_out", "atten_out", self.reshape_out)

    def build_dense_graph(self, graph):
        dense_linear_param = {
            "op_name": "dense_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.proj.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 128 if self.quantize == QuantType.W4A16 else 0
        }
        dense_linear_tensor_map = {
            "input": 'atten_out',
            "linear_out": 'dense_out'
        }
        dense_linear_parallel_param = {
            "op_name": "dense_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": dense_linear_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(dense_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, dense_linear_tensor_map)

    def build_graph(self, graph):
        self.build_qkv_graph(graph)
        self.build_attention_graph(graph)
        self.build_dense_graph(graph)


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig, weights, layer_id=0, backend=None, prefix=None):
        super().__init__()
        self.config = config
        self.weights = weights
        self.dtype = weights.dtype
        self.layer_id = layer_id
        self.backend = backend
        
        self.prefix = f"{prefix}.mlp"
        self.norm_prefix = f"{prefix}.norm2"  
        self.act = ACT2FN[config.hidden_act]
        setattr(config, 'quantize', None)
        self.quantize = config.quantize
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
       
        self.fc1 = TensorParallelColumnLinear.load(config,
                    prefix=f"{self.prefix}.fc1",
                    weights=weights,
                    bias=True,)
        self.fc2 = TensorParallelRowLinear.load(config,
                    prefix=f"{self.prefix}.fc2",
                    weights=weights,
                    bias=True,)
    
    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        weights_dict.update(self.fc1.linear.get_weights(f"{self.prefix}.fc1"))
        weights_dict.update(self.fc2.linear.get_weights(f"{self.prefix}.fc2"))
        return weights_dict
    
    def build_activation_graph(self, graph):
        act = atb.BaseOperation(
            op_type="Activation",
            op_param=json.dumps({'activationType': 'ACTIVATION_GELU', "geluMode": "TANH_MODE"}),
            op_name="Activation_gelu",
        )
        swish_input_list = ["fc1_out"]
        swish_output_list = ["activation_out"]
        graph.operations.append(act)
        graph.add_operation(
            act,
            swish_input_list,
            swish_output_list,
        )


    def build_fc1_graph(self, graph):
        input_key_list = [f"{self.norm_prefix}_out", f"{self.prefix}.fc1.weight", f"{self.prefix}.fc1.bias"]
        linear_out = ["fc1_out"]
        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({
                "transposeB": True,
                "hasBias": True}),
            op_name="fc1" + "_Linear"
        )
        graph.operations.append(linear_op)
        graph.add_operation(
            linear_op,
            input_key_list,
            linear_out
        )
        


    def build_fc2_graph(self, graph):
        fc2_linear_param = {
            "op_name": "fc2_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.fc2.linear,
            "enable_quant_input": False,
            "default_dtype": self.dtype,
            "group_size": 128 if self.quantize == QuantType.W4A16 else 0
        }
        fc2_linear_tensor_map = {
            "input": 'activation_out',
            "linear_out": 'fc2_out'
        }
        fc2_linear_parallel_param = {
            "op_name": "fc2_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size,
                                                backend=self.backend),
            "linear_param": fc2_linear_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(fc2_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, fc2_linear_tensor_map) 
    
    def build_graph(self, graph):
        self.build_fc1_graph(graph)
        self.build_activation_graph(graph)
        self.build_fc2_graph(graph)


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float, weights, idx, prefix):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.weights = weights
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type
        self.layer_id = idx
        self.soc_info = NPUSocInfo()
        self.prefix = f"{prefix}.{idx}"
        self.backend = CommunicationBackend.HCCL if self.soc_info.need_nz else CommunicationBackend.LCCL
        self.layer_norm_eps = config.layer_norm_eps

        self.norm1 = LayerNormATB(config, weights, f"{self.prefix}.norm1")
        self.attn = InternAttention(config, weights, self.layer_id, self.backend, self.prefix)
        self.norm2 = LayerNormATB(config, weights, f"{self.prefix}.norm2")
        self.mlp = InternMLP(config, weights, self.layer_id, backend=self.backend, prefix=self.prefix)


    def get_in_tensor_names(self):
        return ["hidden_states", "seq_len"]

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            if name == "drop_path1" or name == "drop_path2":
                continue
            if isinstance(module, nn.ModuleList):
                for i, single_module in enumerate(module):
                    weights_dict.update(single_module.get_weights(f"{prefix}.{name}.{i}"))
            else:
                if name == "mlp" or name == "attn":
                    weights_dict.update(module.get_weights(f"{prefix}.{self.layer_id}.{name}"))
                if name == "norm1" or name == "norm2":
                    weights_dict.update(module.get_weights(f"{prefix}.{self.layer_id}.{name}"))
        weights_dict[f"{prefix}.{self.layer_id}.ls1"] = self.weights.get_tensor(f"{prefix}.{self.layer_id}.ls1").npu()
        weights_dict[f"{prefix}.{self.layer_id}.ls2"] = self.weights.get_tensor(f"{prefix}.{self.layer_id}.ls2").npu()
        self.weight_names = list(weights_dict.keys())
        return weights_dict
    
    def reshape_hidden(self, org_shape):
        return [org_shape[0] * org_shape[1], org_shape[2]]
    
    def reshape_out(self, org_shape):
        return [13, 1025, org_shape[1]]

    def build_add1_graph(self, graph):
        ls1_op = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="add_1",
        )
        graph.operations.append(ls1_op)
        graph.add_operation(
            ls1_op, ["hidden_states", "ls1_out"], ["hidden_states"]
        )

    def build_add2_graph(self, graph):
        ls1_op = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="add_2",
        )
        graph.operations.append(ls1_op)
        graph.add_operation(
            ls1_op, ["hidden_states", "ls2_out"], ["layer_out"]
        )
    
    def build_ls1_graph(self, graph):
        ls1_op = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_MUL"}),
            op_name="ls1",
        )
        graph.operations.append(ls1_op)
        graph.add_operation(
            ls1_op, ["dense_out", f"{self.prefix}.ls1"], ["ls1_out"]
        )

    def build_ls2_graph(self, graph):
        ls1_op = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_MUL"}),
            op_name="ls2",
        )
        graph.operations.append(ls1_op)
        graph.add_operation(
            ls1_op, ["fc2_out", f"{self.prefix}.ls2"], ["ls2_out"]
        )


    
    def build_graph(self, graph):
        self.layer_graph = AtbGraph("encoder" + f"_layer_{self.layer_id}_graph")
        self.layer_graph.add_input_output(input=self.weight_names + self.get_in_tensor_names(), output=["layer_out"])
        self.norm1.build_graph(self.layer_graph)
        self.attn.build_graph(self.layer_graph)
        self.build_ls1_graph(self.layer_graph)
        self.build_add1_graph(self.layer_graph)

        self.norm2.build_graph(self.layer_graph)
        self.mlp.build_graph(self.layer_graph)
        self.build_ls2_graph(self.layer_graph)
        self.build_add2_graph(self.layer_graph)
        self.layer_graph.build()
        graph.operations.append(self.layer_graph)
        graph.add_operation(self.layer_graph, self.weight_names + self.get_in_tensor_names(), ["hidden_states"])


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig, weights):
        super().__init__()
        self.config = config
        self.weights = weights
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.model_prefix = "model.vision_tower.vision_tower.encoder.layers"
        self.layers = nn.ModuleList([InternVisionEncoderLayer(config, dpr[idx], weights, idx, self.model_prefix)
                                     for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True
        self.graph = None
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()

    def prepare_inputs(self, hidden_states):
        context_length = torch.tensor(hidden_states.size(0) * [hidden_states.size(1)], dtype=torch.int32).npu()
        self.graph_inputs[GRAPH_KEY].update({"hidden_states": hidden_states})
        self.graph_inputs[GRAPH_KEY].update({"seq_len": context_length})
        self.graph_param[GRAPH_KEY]["seq_len"] = context_length.cpu().to(torch.int32)
        self.graph_outputs[GRAPH_KEY]["hidden_states"] = hidden_states

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        self.prepare_inputs(hidden_states)
        hidden_states = self.graph.forward(self.graph_inputs[GRAPH_KEY],
                                           self.graph_outputs[GRAPH_KEY], self.graph_param[GRAPH_KEY])
        return hidden_states

    def get_in_tensor_names(self):
        return ["hidden_states", "seq_len"]

    def get_out_tensor_names(self):
        return ['hidden_states']
    
    def get_weights(self):
        weights_dict = OrderedDict()
        for layer in self.layers:
            weights = layer.get_weights(self.model_prefix)
            weights_dict.update(weights)
        return weights_dict
    
    def build_graph(self):
        self.graph.add_input_output(
            input=list(self.weight.keys()) + self.get_in_tensor_names(), output=self.get_out_tensor_names())
        for layer in self.layers:
            layer.build_graph(self.graph)
        self.graph.execute_as_single = False
        self.graph.build()
        self.graph.set_weights(self.weight)

    def init_graph(self):
        self.weight = self.get_weights()
        self.graph = AtbGraph("vita_vit_graph")
        self.build_graph()


class InternVisionModel(PreTrainedModel):
    def __init__(self, config: InternVisionConfig, weights=None):
        super().__init__(config)
        self.config = config
        self.weights = weights
        
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config, self.weights)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}.'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, pixel_values: Optional[torch.FloatTensor] = None):
        if pixel_values is None:
            logger.error('You have to specify pixel_values.',
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError('You have to specify pixel_values.')
        if len(pixel_values.shape) == 4:
            hidden_states = self.embeddings(pixel_values)
        else:
            logger.error(f'Wrong `pixel_values` size: {pixel_values.shape}.',
            ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(f'Wrong `pixel_values` size: {pixel_values.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
        )
        return encoder_outputs