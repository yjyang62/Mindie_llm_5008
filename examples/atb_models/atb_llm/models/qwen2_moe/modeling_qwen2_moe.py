# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import torch
from torch import nn
from transformers.activations import ACT2FN

from atb_llm.models.qwen2.modeling_base import (
    QwenRMSNorm,
    QwenRMSNormBias,
    QwenRMSNormWrapper,
)
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    load_column_multi,
)
from atb_llm.utils.moe_utils import assign
from atb_llm.utils import OpBackend

from atb_llm.utils.layers.linear import FastLinear
from atb_llm.utils.quantize.pack_type import PackType, calc_linear_pack_type
from atb_llm.utils.log import logger
from atb_llm.utils.weights import ProcessGroupType
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.moe_utils import parse_ep_balance_file, EPLBType
from ..qwen2.modeling_qwen2 import FlashQwenAttention
from ..base.model_utils import get_tqdm_iterator


class QwenMLP(nn.Module):
    def __init__(self, prefix, config, weights, intermediate_size):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_up_proj = load_column_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            head_size=1,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",  # down_proj
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
                (intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )


class QwenEp(nn.Module):
    """
    for experts parallel.
    """
    def __init__(self, prefix, config, weights):
        super().__init__()
        expert_gate_proj = weights.get_tensor(f"{prefix}.gate_proj.weight")
        self.expert_gate_proj = nn.Parameter(expert_gate_proj)
        expert_up_proj = weights.get_tensor(f"{prefix}.up_proj.weight")
        self.expert_up_proj = nn.Parameter(expert_up_proj)
        expert_down_proj = weights.get_tensor(f"{prefix}.down_proj.weight")
        self.expert_down_proj = nn.Parameter(expert_down_proj)


class QwenExpertGate(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)


class QwenSharedExpertGate(nn.Module):

    def __init__(self, prefix, config, weights):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)


class QwenMoe(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, prefix, config, weights, layer_id=None, llm_config=None, init_expert_table=None):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        self.ep = weights.mapping.has_moe_ep()
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.tp = True  # default the model is tensor parallel
        self.expert_parallel_degree = 1 if self.tp else self.tp_world_size
        if self.expert_parallel_degree == 0:
            msg = "expert_parallel degree should not be 0!"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
        
        self.gate = FastLinear.load(
                prefix=f"{prefix}.gate",
                weights=weights,
                bias=False,
                )
        
        self.expert_lists = assign(config.num_experts, weights.mapping.moe_ep.group_size)
        self.device_expert = self.expert_lists[weights.mapping.moe_ep.rank]
        if self.ep:
            mapping = weights.mapping.moe_ep
        else:
            mapping = weights.mapping.moe_tp if weights.mapping.has_moe_tp() else weights.mapping.mlp_tp

        self.rank = mapping.rank
        self.world_size = mapping.group_size

        loading_table = []
        self.expert_lists = []
        if llm_config is not None and \
            llm_config.models.qwen_moe.eplb.level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:
            if (llm_config.models.qwen_moe.eplb.expert_map_file is not None and
                    os.path.exists(llm_config.models.qwen_moe.eplb.expert_map_file)):
                loading_table = parse_ep_balance_file(
                    llm_config.models.qwen_moe.eplb.expert_map_file,
                    layer_id=layer_id,
                    n_device=self.world_size,)
            elif llm_config.models.qwen_moe.eplb.level == EPLBType.DYNAMIC_EPLB:
                loading_table = init_expert_table
            if loading_table is None:
                msg = "Invalid EPLB table."
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)
            self.expert_lists = loading_table[layer_id]
        elif self.ep:
            self.expert_lists = assign(config.num_experts, self.world_size)
        else:
            self.expert_lists = [[i for i in range(config.num_experts)] for j in range(self.world_size)]

        self.device_expert = self.expert_lists[self.rank]
        if (config.ep_level == 1) and self.ep:
            temp_list = [j for j in range(config.num_experts)]
            temp_list = temp_list[self.device_expert[0]:] + temp_list[:self.device_expert[0]]
            self.gate.weight.data = self.gate.weight.data[temp_list]

        expert_prefix = f"{prefix}.experts"
        linear_names = [f'{expert_prefix}.{0}.up_proj', f'{expert_prefix}.{0}.gate_proj']
        pack_name = f'{expert_prefix}.{0}.gate_up_proj'
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.pack_type = calc_linear_pack_type(weights, linear_names, norm_name, pack_name)
        if weights.mapping.moe_ep.group_size > 1:
            weights.switch_process_group(ProcessGroupType.MOE_EP)
        elif weights.mapping.moe_tp.group_size > 1:
            weights.switch_process_group(ProcessGroupType.MOE_TP)
        else:
            weights.switch_process_group(ProcessGroupType.MOE_DP) # FakeGroup(0,1)

        pack_prefixes = None
        if self.pack_type in [
            PackType.ALL_FP, PackType.ALL_W8A8, PackType.ALL_W8A8_ANTI, PackType.ALL_W4A16,
            PackType.ALL_W4A16_ANTI, PackType.ALL_W8A16, PackType.ALL_W8A16_ANTI,
            PackType.MIX_W8A8_DYNAMIC, PackType.MIX_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W8A8_DYNAMIC, PackType.ALL_W8A8_DYNAMIC_ANTI,
            PackType.ALL_W4A8, PackType.MIX_W4A8,
            PackType.ALL_W4A8_ANTI, PackType.MIX_W4A8_ANTI,
        ]:
            pack_prefixes = [[f"{expert_prefix}.{i}.gate_proj", f"{expert_prefix}.{i}.up_proj"] \
                            for i in self.device_expert]
            self.gate_up_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefix_list=pack_prefixes,
                    weights=weights,
                    bias=False
                )
        elif self.pack_type in [PackType.ALL_W8A8SC, PackType.ALL_W8A8SC_ANTI]:
            pack_prefixes = [[f"{expert_prefix}.{i}.gate_up_proj"] \
                            for i in self.device_expert]
            self.gate_up_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefix_list=pack_prefixes,
                    weights=weights,
                    bias=False
                )
        else:
            gate_prefixes = [[f"{expert_prefix}.{i}.gate_proj"] \
                            for i in self.device_expert]
            up_prefixes = [[f"{expert_prefix}.{i}.up_proj"] \
                            for i in self.device_expert]
            self.gate_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefix_list=gate_prefixes,
                    weights=weights,
                    bias=False
                )
            self.up_proj = TensorParallelColumnLinear.load_moe(
                    config,
                    prefix_list=up_prefixes,
                    weights=weights,
                    bias=False
                )

        down_prefixes = [f"{expert_prefix}.{i}.down_proj" \
                        for i in self.device_expert]
        self.down_proj = TensorParallelRowLinear.load_moe(
                config,
                prefix_list=down_prefixes,
                process_group=weights.process_group,
                weights=weights,
                bias=False
            )

        if config.has_shared_expert:
            # share experts
            shared_expert_prefix = f"{prefix}.shared_expert"
            self.shared_expert = QwenMLP(prefix=shared_expert_prefix, config=config, weights=weights,
                                        intermediate_size=config.shared_expert_intermediate_size)
            # share experts gate
            shared_expert_gate_prefix = f"{prefix}.shared_expert_gate"
            self.shared_expert_gate = QwenSharedExpertGate(prefix=shared_expert_gate_prefix,
                                                           config=config, weights=weights)
        
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
            )

        if self.ep:
            if llm_config is not None and \
                llm_config.models.qwen_moe.eplb.level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:
                expert_routing_map = {}
                expert_list = self.expert_lists
                for i, v in enumerate(torch.tensor(expert_list).flatten()):
                    v = v.item()
                    if v not in expert_routing_map:
                        expert_routing_map[v] = [i]
                    else:
                        expert_routing_map[v].append(i)

                for key in expert_routing_map.keys():
                    num_of_duplications = len(expert_routing_map[key])
                    expert_routing_map[key] = expert_routing_map[key][self.rank % num_of_duplications]
                
                expert_routing_map = torch.scatter(torch.zeros(len(expert_routing_map.keys()), dtype=torch.int32),
                                                0, 
                                                torch.tensor(list(expert_routing_map.keys()), dtype=torch.int64),
                                                torch.tensor(list(expert_routing_map.values()), dtype=torch.int32)
                                                )
                weights.expert_routing_map[layer_id] = expert_routing_map


class FlashQwenLayer(nn.Module):
    def __init__(self, layer_id, config, weights, llm_config=None, init_expert_table=None):
        super().__init__()
        prefix = f"model.layers.{layer_id}"

        weights.switch_process_group(ProcessGroupType.ATTN)
        self.self_attn = FlashQwenAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights, attn_decode_backend=OpBackend.ATB
        )
        
        if config.is_dense_layer[layer_id]:
            self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights,
                                                  intermediate_size=config.intermediate_size)
        else:
            self.mlp = QwenMoe(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_id=layer_id,
                               llm_config=llm_config, init_expert_table=init_expert_table)
        
        if self.self_attn.pack_type in [PackType.ALL_FP, PackType.ALL_W8A16, PackType.ALL_W4A16]:
            self.input_layernorm = QwenRMSNorm(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
                                     PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
                                     PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
                                     PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.input_layernorm = QwenRMSNormWrapper(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.self_attn.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8_DYNAMIC]:
            self.input_layernorm = QwenRMSNormBias(
                prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        else:
            raise AssertionError(f'self_attn.pack_type: {self.self_attn.pack_type} not supported')
        

        if self.mlp.pack_type in [PackType.ALL_FP, PackType.ALL_W4A16, PackType.ALL_W8A16]:
            self.post_attention_layernorm = QwenRMSNorm(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8_ANTI, PackType.MIX_W8A8_ANTI,
                                     PackType.ALL_W8A16_ANTI, PackType.MIX_W8A16_ANTI,
                                     PackType.ALL_W4A16_ANTI, PackType.MIX_W4A16_ANTI,
                                     PackType.ALL_W8A8_DYNAMIC_ANTI]:
            self.post_attention_layernorm = QwenRMSNormWrapper(
                prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
            )
        elif self.mlp.pack_type in [PackType.ALL_W8A8, PackType.MIX_W8A8, PackType.ALL_W8A8_DYNAMIC]:
            self.post_attention_layernorm = QwenRMSNormBias(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
        else:
            raise AssertionError(f'mlp.pack_type: {self.mlp.pack_type} not supported')


class FlashQwenModel(nn.Module):
    def __init__(self, config, weights, llm_config=None, init_expert_table=None):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        
        layer_list = []
        iterator = get_tqdm_iterator(range(config.num_hidden_layers), 
                                     weights.mapping.rank % weights.mapping.local_world_size)
        for layer_id in iterator:
            layer_list.append(FlashQwenLayer(
                    layer_id,
                    config,
                    weights,
                    llm_config,
                    init_expert_table
            ))
        self.layers = nn.ModuleList(layer_list)

        self.norm = QwenRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )
        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
