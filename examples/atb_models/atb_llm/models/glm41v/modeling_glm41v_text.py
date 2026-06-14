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
from typing import Optional, List, Tuple
import torch
from torch import nn
import torch_npu
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.base.modeling import FlashAttention, FlashLayer, MLP
from atb_llm.models.base.inputs_modifier.qlen_modifier import QLenModifier
from atb_llm.models.base.inputs_modifier.flash_comm_modifier import FlashCommModifier
from atb_llm.models.base.graph_manager import ATBGraphManager, DapGraphWrapper, SpeculateGraphWrapper, \
    FlashCommGraphWrapper
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import TensorParallelRowLinear, RMSNorm, TensorEmbedding, TensorHead, \
    load_column_multi, PositionRotaryEmbedding, AttentionMask
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.data.weight_wrapper import WeightWrapper, get_module, AttnWrapper, MlpWrapper
from atb_llm.utils.quantize.pack_type import calc_linear_pack_type
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log import logger


_800_9000_SOCS = (100, 101, 102, 103, 104)
DUO_SOCS = (200, 201, 202, 203, 204, 205)
A2_SOCS = (220, 221, 222, 223, 224, 225)
A3_SOCS = (250, 251, 252, 253, 254, 255)


class Glm41vTextAttention(FlashAttention):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__(prefix, config, weights)
        if config.quantize == "w8a8sc":
            self.qkv_names = [f"{prefix}.query_key_value"]
        else:
            self.qkv_names = [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        self.dense_name = f"{prefix}.o_proj"
        self.qkv_bias = True
        self.multi_query_group_num = config.num_key_value_heads
        self.kv_head_nums_per_rank = max(self.multi_query_group_num // weights.process_group.size(), 1)
        self.pack_type = calc_linear_pack_type(weights, self.qkv_names, self.norm_name)
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=self.dense_name,
            weights=weights,
            bias=False,
        )
        self.load_qkv_weights()


class Glm41vTextMLP(MLP):
    def __init__(self, prefix, config, weights):
        super().__init__(prefix, config, weights)

        self.gate_up_names = [f'{prefix}.gate_up_proj']
        self.pack_name = f'{prefix}.gate_up_proj'
        self.down_name = f'{prefix}.down_proj'
        self.load_weights()


class Glm41vTextDecoderLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, prefix):
        super().__init__(layer_id, config, weights, prefix)
        self.self_attn = Glm41vTextAttention(
                prefix=f"{self.prefix}.self_attn", config=config, weights=weights
            )
        self.mlp = Glm41vTextMLP(prefix=f"{self.prefix}.mlp", config=config, weights=weights)
        self.load_weights()
        self.post_self_attn_layernorm = RMSNorm(prefix=f"{self.prefix}.post_self_attn_layernorm",
                                                weights=weights, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(prefix=f"{self.prefix}.post_mlp_layernorm",
                                          weights=weights, eps=config.rms_norm_eps)


class Glm41vWeightWrapper(WeightWrapper):
    def register_layer(self, layer: Glm41vTextDecoderLayer, quantize_type):
        self.layer_linear_type.clear()
        self.layer_linear_descs.clear()
        self.layer_linear_transpose_types.clear()
        self.layer_is_anti_outlier.clear()
        self.register_layer_attn(layer, self.attn_wrapper, quantize_type)
        self.register_model_norm(layer.post_self_attn_layernorm)
        self.register_layer_mlp(layer, self.mlp_wrapper, quantize_type)
        self.register_model_norm(layer.post_mlp_layernorm)
        self.linear_type.append(self.layer_linear_type.copy())
        self.linear_descs.append(self.layer_linear_descs.copy())
        self.linear_transpose_types.append(self.layer_linear_transpose_types.copy())
        self.is_anti_outlier.append(self.layer_is_anti_outlier.copy())
        attn_pack_type = get_module(layer, self.attn_wrapper.wrapper_name).pack_type
        mlp_pack_type = get_module(layer, self.mlp_wrapper.wrapper_name).pack_type
        self.pack_quant_type.append([attn_pack_type, mlp_pack_type])


class FlashGlm41vTextModelForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        if self.quantize == "w8a8sc":
            prefix = "language_model"
        else:
            prefix = "model.language_model"
        self.enable_rope_quant_kvcache = self.config.quantization_config.kv_quant_type is not None
        self.hidden_size = config.hidden_size
        self.multi_query_group_num = self.config.num_key_value_heads

        self.embed_tokens = TensorEmbedding(prefix=f"{prefix}.embed_tokens", weights=weights)
        self.layers = nn.ModuleList(
            [
                Glm41vTextDecoderLayer(layer_id, config, weights, prefix)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps)
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix=f"{prefix}.lm_head",
                weights=weights,
                is_norm=False,
            )
        else:
            self.lm_head = load_column_multi(
                self.config,
                prefixes=["lm_head"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )
        self.attn_mask = AttentionMask.static(self.max_base_len)
        rope_theta = getattr(self.config, "rope_theta", 10000)
        partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 1.0)
        self.rotary_pos_emb = PositionRotaryEmbedding.static(
            dim=int(self.head_size * partial_rotary_factor), base=rope_theta, device="cpu")
        mrope_section = self.config.rope_scaling.mrope_section
        self.mrope_section = [x * 2 for x in mrope_section] * 2
        self.rotary_pos_emb.update_cos_sin_cache_total(torch.float32, self.device, self.config.max_position_embeddings)
        self.cos_embed = self.rotary_pos_emb.get_cos_cached_total().repeat_interleave(2, dim=-1)
        self.sin_embed = self.rotary_pos_emb.get_sin_cached_total().repeat_interleave(2, dim=-1)
        self.cos_embed, self.sin_embed = self.cos_embed.to(self.dtype), self.sin_embed.to(self.dtype)
        self.cos_embed_decode = \
            self.cos_embed[..., : self.cos_embed.shape[-1] // 2].unsqueeze(1)  # BS, 1, D
        self.sin_embed_decode = \
            self.sin_embed[..., : self.cos_embed.shape[-1] // 2].unsqueeze(1)  # BS, 1, D
        self.ascend_weight = []
        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.soc_info = NPUSocInfo()
        if self.soc_info.need_nz:
            self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
            self.transdata_param = json.dumps({})
            self.transdata_operation.set_param(self.transdata_param)
        # Multi graph management
        self.graph_manager = ATBGraphManager()
        self.qlen_decorator = QLenModifier()
        self.flash_comm_modifier = FlashCommModifier(weights, self.hidden_size, self._flash_comm_gate())

    def init_ascend_operations(self, config):
        pass

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=None,
            down_name='down_proj'
        )
        weight_wrapper = Glm41vWeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.embed_tokens)
        for i in range(self.config.num_hidden_layers):
            layer = self.layers[i]
            weight_wrapper.register_layer(layer, self.config.quantize)
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
        weight_wrapper.register_model_norm(self.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper
    
    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_quant_type = weight_wrapper.linear_type
        pack_quant_type = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        coder_param = {
            "enableAddNorm": False,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.config.num_attention_heads // self.tp_world_size,
            "hiddenSizePerAttentionHead": self.config.hidden_size // self.config.num_attention_heads,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": max(self.multi_query_group_num // self.tp_world_size, 1),
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_type,
            "weightQuantType": self.config.quantize if self.config.quantize else "",
            "quantGroupSize": self.config.quantization_config.group_size,
            "linearQuantType": linear_quant_type,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "isUnpadInputs": True,
            "skipWordEmbedding": True,
            "isLmHeadParallel": True,
            "enableSwiGLU": True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "positionEmbeddingType": PositionEmbeddingType.ROPE,
            "linearHasBias": [[True, False, False, False]] * self.config.num_hidden_layers
        }
        encoder_param = {
            **coder_param,
            "isPrefill": True,
            "enablePreFetchWeight": self.soc_info.soc_version in DUO_SOCS,  # Negative performance gains in A2
            "enableLcoc": self.lcoc_enable,
        }
        decoder_param = {
            **coder_param,
            "isPrefill": False,
            "enableLcoc": False
        }
        if self.speculate_enable:
            self.graph_manager.register_graph(SpeculateGraphWrapper())
        if self.enable_dap:
            self.graph_manager.register_graph(DapGraphWrapper())
        if self.flash_comm_modifier.enable_flash_comm:
            self.graph_manager.register_graph(FlashCommGraphWrapper())

        specified_params = {"decode": decoder_param}
        self.graph_manager.set_param("glm41v_DecoderModel", encoder_param, specified_params)
        self.graph_manager.set_weight(self.ascend_weight)
    
    def init_kvcache(self, kv_cache):
        kcache_id_exist = not self.ascend_kcache_id or self.ascend_kcache_id != id(
            kv_cache[0][0]
        )
        vcache_id_exist = not self.ascend_vcache_id or self.ascend_vcache_id != id(
            kv_cache[0][1]
        )
        if kcache_id_exist or vcache_id_exist:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            logger.debug(f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.debug(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.graph_manager.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            logger.warning(
                f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}"
            )

    def update_thw_cos_sin(self, position_ids_thw):
        cos = self.cos_embed[position_ids_thw]
        sin = self.sin_embed[position_ids_thw]
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))], dim=-1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))], dim=-1)
        cos = cos[..., : cos.shape[-1] // 2].unsqueeze(1)  # BS, 1, D
        sin = sin[..., : sin.shape[-1] // 2].unsqueeze(1)  # BS, 1, D
        return cos, sin
    
    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        attention_mask = kwargs.get('attn_mask', None)
        if attention_mask is None:
            if is_prefill:
                attention_mask = self.attn_mask.get_rope_prefill_mask(self.max_base_len, self.dtype, self.device)
            else:
                attention_mask = self.attn_mask.get_rope_decode_mask(self.dtype, self.device)
        if self.soc_info.need_nz:
            attention_mask = self.transdata_operation.execute([attention_mask])[0]
        if is_prefill and lm_head_indices is None:  # prefill
            lm_head_indices = torch.tensor(
                range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device
            )
        self.acl_operation_inputs = [
            input_ids,
            position_ids,
            self.cos_embed,
            self.sin_embed,
            attention_mask,
            block_tables.to(torch.int32),
            slots.to(torch.int32),
            self.placeholder,
            self.placeholder,
            self.placeholder,
            input_lengths.to(torch.int32),
            lm_head_indices.to(torch.int64) if is_prefill else self.lm_head_indices_fake
        ]
        acl_param = {"seqLen": input_lengths.tolist()}
        self.qlen_decorator.modify_inputs(
            self.acl_operation_inputs, acl_param, self.device,
            is_prefill=is_prefill,
            enable_prefill_pa=False if self.inference_mode is None else self.inference_mode.enable_prefill_pa,
            enable_splitfuse_pa=not self.soc_info.is_300i(),
            **kwargs
        )
        self.flash_comm_modifier.modify_inputs(
            self.acl_operation_inputs,
            is_prefill,
            acl_param
        )
        self.acl_param = json.dumps(acl_param)
        return self.acl_operation_inputs, self.acl_param

    def execute_ascend_operator(
        self,
        acl_inputs,
        acl_param,
        is_prefill
    ):
        acl_model_out = self.graph_manager.select_and_execute(self, acl_inputs, acl_param, is_prefill=is_prefill)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_kvcache(kv_cache)

        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices, **kwargs)
        position_ids_thw = kwargs.pop("position_ids_thw", None)
        if is_prefill:
            cos, sin = self.update_thw_cos_sin(position_ids_thw)
            acl_inputs[1] = torch.arange(input_lengths.sum(), dtype=position_ids.dtype, device=position_ids.device)
        else:
            cos, sin = self.cos_embed_decode, self.sin_embed_decode
        acl_inputs[2] = cos
        acl_inputs[3] = sin
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits
    
    def dap_forward(
            self,
            input_ids: List[torch.Tensor],
            position_ids: List[torch.Tensor],
            is_prefill: List[bool],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: List[torch.Tensor],
            slots: List[torch.Tensor],
            input_lengths: List[torch.Tensor],
            max_seq_len: List[int],
            lm_head_indices: List[torch.Tensor | None],
            dap_kwargs: List[dict],
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_kvcache(kv_cache)
        all_inputs = []
        preceder_inputs = self.prepare_inputs_for_ascend(input_ids[0], position_ids[0], is_prefill[0], kv_cache,
                                                         block_tables[0], slots[0], input_lengths[0], max_seq_len[0],
                                                         lm_head_indices[0], **dap_kwargs[0])
        acl_inputs, acl_param = preceder_inputs[0], preceder_inputs[1]
        position_ids_thw = dap_kwargs[0].pop("position_ids_thw", None)
        cos, sin = self.update_thw_cos_sin(position_ids_thw)
        acl_inputs[1] = torch.arange(
            input_lengths[0].sum(), dtype=position_ids[0].dtype, device=position_ids[0].device)
        acl_inputs[2] = cos
        acl_inputs[3] = sin
        successor_inputs = self.prepare_inputs_for_ascend(input_ids[1], position_ids[1], is_prefill[1], kv_cache,
                                                          block_tables[1], slots[1], input_lengths[1], max_seq_len[1],
                                                          lm_head_indices[1], **dap_kwargs[1])
        acl_inputs_successor, acl_param_successor = successor_inputs[0], successor_inputs[1]
        position_ids_thw_successor = dap_kwargs[1].pop("position_ids_thw", None)
        cos_successor, sin_successor = self.update_thw_cos_sin(position_ids_thw_successor)
        acl_inputs_successor[1] = torch.arange(
            input_lengths[1].sum(), dtype=position_ids[1].dtype, device=position_ids[1].device)
        acl_inputs_successor[2] = cos_successor
        acl_inputs_successor[3] = sin_successor
        if len(acl_inputs_successor) < len(acl_inputs):
            acl_inputs = acl_inputs[:len(acl_inputs_successor)]
            acl_param = dict(
                (k, acl_param[k])
                for k in acl_param_successor.keys()
            )
        all_inputs.extend(acl_inputs)
        all_inputs.extend(acl_inputs_successor)
        acl_param_dict = json.loads(acl_param)
        for k, v in json.loads(acl_param_successor).items():
            acl_param_dict[f"{k}_successor"] = v
        logits = self.execute_dap_ascend_operator(
            all_inputs, json.dumps(acl_param_dict), is_prefill[0])
        return logits
    
    def execute_dap_ascend_operator(self,
                                    acl_inputs: list,
                                    acl_param: str,
                                    is_prefill: bool) -> torch.Tensor:
        acl_model_out = self.graph_manager.select_and_execute(self, acl_inputs, acl_param, \
            is_prefill=is_prefill, enable_dap=True)
        if len(acl_model_out) != 2:
            err_msg = "Number of output tensors is not equal to the expected value."
            logger.error(err_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise RuntimeError(err_msg)
        return acl_model_out
    
    def _flash_comm_gate(self) -> bool:
        soc_version = self.soc_info.soc_version
        return not any([
            self.enable_dap,
            self.tp_world_size == 1,
            soc_version in _800_9000_SOCS,
            soc_version in DUO_SOCS and self.tp_world_size > 4,
            soc_version in A2_SOCS + A3_SOCS and not self.soc_info.is_support_hccs(),
            self.lcoc_enable
        ])