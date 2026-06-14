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
import math
from typing import Optional, List, Tuple

import torch
import torch_npu
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.glm4_moe.config_glm4_moe import Glm4moeConfig
from atb_llm.models.glm4_moe.modeling_glm4_moe import FlashGlm4moeModel
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper
from atb_llm.utils.env import ENV
from atb_llm.utils.layers import PositionRotaryEmbedding
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.moe_utils import assign, ExpertParallelDegree
from atb_llm.utils.weights import ProcessGroupType

from atb_llm.utils.layers import (
    TensorEmbedding,
    load_column_multi,
)
from atb_llm.utils.layers.norm.fast_layer_norm import NormType

_DECODER_MODEL = "glm_MoeDecoderModel"
_SUPPORT_LCOC = "enableLcoc"
_SUPPORT_SPECULATE = "enableSpeculate"
_IS_PREFILL = "isPrefill"
_PREFILL = "prefill"
_DECODE = "decode"
_HCCL = "hccl"
_W8A8 = "w8a8"


class FlashGlm4moeForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        self.acl_decoder_regression_operation = None

        self.distributed_enable = kwargs.get('distributed_enable', False)
        self.max_batch_size = kwargs.get('max_batch_size', 0)
        self.total_batch_size = self.max_batch_size
        if not self.distributed_enable:
            self.max_batch_size = 0

        super().__init__(config, weights, **kwargs)
        
        if self.llm_config is not None:
            self.model_config = self.llm_config.llm
            self.parallel_config = self.llm_config.llm.parallel_options
        else:
            self.model_config = None
            self.parallel_config = None

        self.ep_level = ExpertParallelDegree.NO_EP
        if self.mapping.has_moe_ep():
            if not hasattr(self.model_config, "ep_level"):
                logger.info("For expert parallel, "
                    "the ep_level variable needs to be defined in the model configuration file."
                    "The available options are 1, 2, or 3.")
                self.ep_level = ExpertParallelDegree.STATIC_EP
            else:
                self.ep_level = self.model_config.ep_level
        config.ep_level = self.ep_level
        if hasattr(self.model_config, 'alltoall_ep_buffer_scale_factors'):
            config.alltoall_ep_buffer_scale_factors = self.model_config.alltoall_ep_buffer_scale_factors
        self.model = FlashGlm4moeModel(config, weights)
        weights.switch_process_group(ProcessGroupType.LM_HEAD)
        self.config = config
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.in_tensor_length = 16
        self.acl_operation_inputs = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None

        self.placeholder = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=self.device)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.padding_idx = config.pad_token_id
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.expert_array = self.placeholder
        self.expert_group = torch.arange(1024, dtype=torch.int32).npu() # 1024: const for groupedTopK
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.num_of_experts = config.n_routed_experts
        self.num_of_selected_experts = [config.num_experts_per_tok]
        self.first_k_dense_replace = config.first_k_dense_replace if config.first_k_dense_replace else 0
        self.n_shared_experts = config.n_shared_experts if config.n_shared_experts else 0
        self.norm_topk_prob = config.norm_topk_prob if config.norm_topk_prob else False
        self.head_size = config.head_dim
        self.rotary_embedding = PositionRotaryEmbedding.static(dim=int(config.head_dim * config.partial_rotary_factor),
                                                               base=self.rope_theta,
                                                               device="cpu", scaling_factor=self.scaling_factor) \
            .to(self.device)

        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.topk_method = config.topk_method
        self.routed_scaling_factor = config.routed_scaling_factor

        self.ascend_weight = None

        self.enable_fused_routing = False if self.soc_info.need_nz else True
        self.communication_backend = self.soc_info.communication_backend        
        self.device_expert = assign(self.config.n_routed_experts, self.mapping.moe_ep.group_size)[
            self.mapping.moe_ep.rank]
        if self.mapping.has_moe_ep():
            if self.ep_level in [ExpertParallelDegree.DYNAMIC_EP, ExpertParallelDegree.MIX_EP]:
                self.dep_communication_backend = {_PREFILL: _HCCL, _DECODE: _HCCL} if self.model_config is None else {
                    _PREFILL: self.model_config.communication_backend.prefill, 
                    _DECODE: self.model_config.communication_backend.decode
                }
                self.p_to_d_weight = self.ep_level == ExpertParallelDegree.MIX_EP
            logger.info(f"Expert parallel level is {self.ep_level}.")
            logger.info(f"Experts of rank {self.mapping.moe_ep.rank} are: {self.device_expert}")

        self.num_of_device_expert = len(self.device_expert)
        self.start_device_expert_id = torch.tensor(self.device_expert[0], dtype=torch.int64).npu().view(-1)
        self.max_device_expert_id = torch.tensor([len(self.device_expert) - 1], dtype=torch.int64).npu().view(-1)
        self.enable_node_based_a2a = False
        
        if self.enable_node_based_a2a:
            logger.info("Node Based A2A only support HCCL as communication backend.")
            self.dep_communication_backend = {'prefill': 'hccl', 'decode': 'hccl'}

    def init_ascend_operations(self, config: Glm4moeConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(_DECODER_MODEL)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(_DECODER_MODEL)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        moe_mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=(self.n_shared_experts > 0)
        )
        weight_wrapper = MoeWeightWrapper(self.soc_info, self.tp_rank,
                                              attn_wrapper, moe_mlp_wrapper,
                                              self.num_of_experts)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            if i < self.first_k_dense_replace:
                weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=True, attn_quantize_type=_W8A8)
            elif self.mapping.has_moe_ep():
                weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=False,
                                                  expert_roster=[i for i, _ in enumerate(self.device_expert)],
                                                  attn_quantize_type=_W8A8)
                del layer.mlp
                torch.npu.empty_cache()
            else:
                weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=False, attn_quantize_type=_W8A8)
                del layer.mlp
                torch.npu.empty_cache()
            if self.config.use_qk_norm:
                weight_wrapper.register_model_norm(layer.self_attn.q_norm)  # q_norm
                weight_wrapper.register_model_norm(layer.self_attn.k_norm)  # k_norm
        
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                torch.npu.empty_cache()
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        pack_quant_configs = weight_wrapper.pack_quant_type

        attn_linear_types = weight_wrapper.attn_linear_types
        mlp_linear_types = weight_wrapper.mlp_linear_types
        moe_linear_types = weight_wrapper.moe_linear_types

        attn_linear_transpose_types = weight_wrapper.attn_linear_transpose_types
        mlp_linear_transpose_types = weight_wrapper.mlp_linear_transpose_types
        moe_linear_transpose_types = weight_wrapper.moe_linear_transpose_types

        # compatible with linearQuantType
        for i in range(self.num_layers):
            attn_linear_types[i].append(attn_linear_types[i][-1])
            attn_linear_transpose_types[i].append(-1)

        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "useQKNorm": self.config.use_qk_norm,
            "linearHasBias": [[self.config.attention_bias, False, False, False]] * self.config.num_hidden_layers,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "linearQuantType": attn_linear_types,
            "mlpLinearQuantType": mlp_linear_types,
            "moeLinearQuantType": moe_linear_types,
            "linearTransposeType": attn_linear_transpose_types,
            "mlpLinearTransposeType": mlp_linear_transpose_types,
            "moeLinearTransposeType": moe_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableInitQuant": True if self.quantize == "w8a8_dynamic" else False,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            'hasSharedExpert': True if self.n_shared_experts > 0 else False,
            'hasSharedExpertGate': False,
            "rank": self.tp_rank,
            "expertParallelDegree": self.ep_level,
            "numOfExperts": self.num_of_experts,
            "numOfDeviceExperts": self.num_of_device_expert,
            "deviceExpert": self.device_expert,
            "numOfGroups": 1,
            "routingMethod": self.get_routing_method_type(),
            "enableATBGateMatmul": False,
            "processLogits": self.get_process_logits_type(),
            "routedScalingFactor": self.routed_scaling_factor,
            "firstKDenseReplace": self.first_k_dense_replace,
            "numOfSharedExperts": self.n_shared_experts,
            "numOfSelectedExperts": self.num_of_selected_experts,
            "numOfSelectedGroups": 3,
            "topkGroups": self.topk_group,
            "enableFusedTopk": True if self.topk_method == "noaux_tc" and self.n_group * 32 >= self.num_of_experts \
                                    else False,
            "mapping": self.mapping.to_dict_v2(),
            "worldSize": self.mapping.world_size,
            "rankTableFile": ENV.rank_table_file,
            "enableAddNorm": False,
            "normHasBias": False,
            "enableFusedRouting": self.enable_fused_routing,
            "hasAttnTp": self.mapping.has_attn_tp(),
            "attnTpRank": self.mapping.attn_tp.rank,
            "attnTpSize": self.mapping.attn_tp.group_size,
            "attnTpDomain": self.mapping.attn_tp.domain if self.mapping.has_dp() else "",
            "hasAttnDp": self.mapping.has_dp(),
            "attnDpRank": self.mapping.attn_dp.rank,
            "attnDpSize": self.mapping.attn_dp.group_size,
            "attnDpDomain": self.mapping.attn_dp.domain,
            "hasMlpTp": self.mapping.has_mlp_tp(),
            "mlpTpRank": self.mapping.mlp_tp.rank,
            "mlpTpSize": self.mapping.mlp_tp.group_size,
            "mlpTpDomain": self.mapping.mlp_tp.domain if self.mapping.has_dp() else "",
            "enableAllToAllMC2": self.ep_level == ExpertParallelDegree.DYNAMIC_EP,
            "enableDispatchCombineV2": True,
            "maxDecodeDpTokenSize": self.max_batch_size,
            "enableDpOut": ENV.enable_dp_partition_up,
        }
        
        if self.mapping is not None:
            if self.ep_level == ExpertParallelDegree.DYNAMIC_EP:
                min_moe_ep_buffer_size, min_moe_tp_buffer_size = self.calc_moe_buffer_size()
                if self.mapping.moe_ep.buffer_size < min_moe_ep_buffer_size:
                    msg = f"`hccl_moe_ep_buffer` = {self.mapping.moe_ep.buffer_size} is not enough for " \
                        f"batch size = {self.total_batch_size} and MoE EP size = {self.mapping.moe_ep.group_size}, " \
                        f"so the buff size will be set to {min_moe_ep_buffer_size}."
                    logger.warning(msg)
                    self.mapping.moe_ep.buffer_size = min_moe_ep_buffer_size
                if self.mapping.moe_tp.buffer_size < min_moe_tp_buffer_size:
                    msg = f"`hccl_moe_tp_buffer`={self.mapping.moe_tp.buffer_size} is not enough for " \
                        f"batch size = {self.total_batch_size} and MoE TP size = {self.mapping.moe_tp.group_size}, " \
                        f"so the buff size will be set to {min_moe_tp_buffer_size}."
                    logger.warning(msg)
                    self.mapping.moe_tp.buffer_size = min_moe_tp_buffer_size
            coder_param.update({"mapping": self.mapping.to_dict_v2()})
        
        if self.ep_level == ExpertParallelDegree.DYNAMIC_EP and self.mapping.moe_tp.group_size > 1:
            msg = f"Dynamic EP only supports moe_tp = 1, but gets moe_tp = {self.mapping.moe_tp.group_size}."
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
        
        encoder_param = {
            **coder_param, _IS_PREFILL: True, _SUPPORT_LCOC: self.lcoc_enable,
            _SUPPORT_SPECULATE: False,
            "backend": self.communication_backend if self.ep_level < 2 \
                            else self.dep_communication_backend[_PREFILL],
            "enableGMMSwigluQuant": False,
            "enableNodeBaseAll2All": self.enable_node_based_a2a,
        }
        decoder_param = {
            **coder_param, _IS_PREFILL: False, _SUPPORT_LCOC: False,
            "backend": self.communication_backend if self.ep_level < 2 \
                            else self.dep_communication_backend[_DECODE],
            "enableGMMSwigluQuant": self.quantize == "w8a8_dynamic" and (not self.soc_info.need_nz),
        }
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    # called by super().forward()
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
        self.init_position_rotary_embedding(position_ids, max_seq_len)
        self.cos_embed = self.cos_embed.view(-1, 2, self.cos_embed.shape[-1] // 2)
        self.sin_embed = self.sin_embed.view(-1, 2, self.sin_embed.shape[-1] // 2)
        if self.mapping.has_dp() and self.mapping.has_mlp_tp():
            if ENV.enable_dp_move_up and not ENV.enable_dp_partition_up:
                token_size_per_dp_group = kwargs.get("token_size_per_dp_group", None)
                token_size = token_size_per_dp_group.sum().item()
            else:
                token_size = len(input_ids)
            shard_effective_token_indices = kwargs.get("shard_effective_token_indices", self.placeholder)
            token_index_with_padding = kwargs.get("token_index_with_padding", self.placeholder)
            skip_padding_token_indices = kwargs.get("skip_padding_token_indices", self.placeholder)
            final_hidden_states = torch.empty([token_size, self.config.hidden_size],
                                              dtype=kv_cache[0][0].dtype,
                                              device=input_ids.device)
        q_lens = kwargs.get('q_lens', [])

        input_length = len(input_ids)
        if not self.enable_fused_routing:
            self.expert_array = torch.arange(self.config.num_experts_per_tok * input_length,
                                             dtype=torch.int32, device=input_ids.device)

        attn_padding_idx = self.placeholder
        attn_unpadding_idx = self.placeholder
        ffn_padding_idx = self.placeholder
        ffn_unpadding_idx = self.placeholder
        lm_head_skip_padding_token_indices = self.placeholder
        gather_prenorm_idx = self.placeholder
        dynamic_ep_idx = self.placeholder
        moe_idx = self.placeholder
        
        dep_inputs = [attn_padding_idx, attn_unpadding_idx, ffn_padding_idx,
            ffn_unpadding_idx, lm_head_skip_padding_token_indices, gather_prenorm_idx,
            self.start_device_expert_id, self.max_device_expert_id,
            dynamic_ep_idx, moe_idx, self.placeholder]

        if self.mapping.has_dp():
            if ENV.enable_dp_move_up:
                has_tp = self.mapping.has_attn_tp() or self.mapping.lm_head_tp.group_size > 1
                if not self.distributed_enable or (has_tp and self.distributed_enable):
                    dep_inputs = kwargs.get("dep_inputs", None)
                    dep_inputs = dep_inputs[:6] + \
                                    [self.start_device_expert_id, self.max_device_expert_id] + dep_inputs[6:]
                moe_idx = dep_inputs[-2] # note that moe_idx is the second last tensor in dep_inputs
                self.expert_array = torch.ones(moe_idx.shape[0], dtype=torch.float16).npu().view(-1, 1)
                if self.ep_level == ExpertParallelDegree.DYNAMIC_EP and is_prefill and self.enable_node_based_a2a:
                    dynamic_ep_idx = dep_inputs[-3]
                    dynamic_ep_idx = torch.arange(dynamic_ep_idx.shape[0] * self.mapping.num_nodes,
                        dtype=torch.int32).npu().view(-1)
                    dep_inputs[-3] = dynamic_ep_idx
            else:
                message = "Please export DP_MOVE_UP_ENABLE=1 when set attn_dp > 1."
                logger.error(message)
                raise RuntimeError(message)
        else:
            if self.ep_level == ExpertParallelDegree.DYNAMIC_EP:
                message = "Currently do not support ep_level=2 when attn_dp=1."
                logger.error(message)
                raise NotImplementedError(message)
            input_length = len(input_ids)
            self.expert_array = torch.tensor(
                            [j for j in range(input_length * self.config.num_experts_per_tok)],
                            dtype=torch.int32
                            ).npu().view(-1)
        
        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                           dtype=torch.int64, device=input_ids.device)
        if is_prefill:
            if self.soc_info.need_nz:
                pad_max = math.ceil(self.max_position_embeddings / 16) * 16
                attn_mask = self.attn_mask.get_attn_mask(pad_max, kv_cache[0][0].dtype,
                                                         kv_cache[0][0].device)
                attn_mask = self.transdata_operation.execute([attn_mask])[0]
            else:
                attn_mask = self.attn_mask.get_attn_mask(self.max_base_len, self.dtype, self.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
                "qLen": q_lens
            })
            self.acl_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                torch.where(attn_mask == -torch.inf, 1, attn_mask) if self.dtype == torch.bfloat16 else attn_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64),
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot
            ]
            if self.mapping.has_dp() and self.mapping.has_mlp_tp():
                self.acl_operation_inputs.insert(12, final_hidden_states)
                self.acl_operation_inputs.insert(13, shard_effective_token_indices.to(self.device))
                self.acl_operation_inputs.insert(14, token_index_with_padding)
                self.acl_operation_inputs.insert(15, skip_padding_token_indices)
            if self.split_fuse_enable:
                self.acl_operation_inputs.append(torch.tensor(q_lens).to(self.device).to(torch.int32))
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist(),
                "qLen": q_lens
            })
            self.acl_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                self.attn_mask_fake,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64) if self.mapping.has_dp() and self.mapping.has_mlp_tp() \
                                                else self.lm_head_indices_fake,
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot
            ]
            if self.mapping.has_dp() and self.mapping.has_mlp_tp():
                self.acl_operation_inputs.insert(12, final_hidden_states)
                self.acl_operation_inputs.insert(13, shard_effective_token_indices)
                self.acl_operation_inputs.insert(14, token_index_with_padding)
                self.acl_operation_inputs.insert(15, skip_padding_token_indices)

        self.acl_operation_inputs.extend(dep_inputs)
        return self.acl_operation_inputs, self.acl_param

    def select_logits(self, logits, **kwargs):
        dp_logits_num = kwargs.get("dp_logits_num")
        if dp_logits_num is None:
            return logits
        dp_rank_id = self.mapping.attn_dp.rank
        if dp_rank_id == 0:
            logits = logits[:dp_logits_num[dp_rank_id]]
        else:
            logits = logits[dp_logits_num[dp_rank_id - 1]: dp_logits_num[dp_rank_id]]
        return logits
    
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
            reset=False,
            **kwargs,
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()
        self.init_kvcache(kv_cache)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices, **kwargs)
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        if is_prefill and self.distributed_enable:
            logits = self.select_logits(logits, **kwargs)
        return logits

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill):
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            model_operation = self.acl_decoder_operation
            acl_model_out = model_operation.execute(acl_inputs, acl_param)
        try:
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            msg = "Runtime Error, please refer to the logs for more info"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise RuntimeError(msg) from e
        return acl_hidden_state

    def init_kvcache(self, kv_cache):
        kcache_id = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])

    def get_process_logits_type(self) -> str:
        return "normScaling" if self.routed_scaling_factor > 1 else "normalization"

    def get_routing_method_type(self) -> str:
        return "noAuxTc"
            
    def calc_moe_buffer_size(self):
        num_redundant_experts = 0
        moe_ep_buffer_size = math.ceil(
            math.ceil(self.total_batch_size / self.mapping.world_size) * self.hidden_size
            * (self.num_of_experts + num_redundant_experts) * 4 / (1024 ** 2)
        ) + 1
        moe_tp_buffer_size = math.ceil(
            math.ceil(self.total_batch_size / self.mapping.world_size) * self.hidden_size
            * self.mapping.moe_tp.group_size * 4 / (1024 ** 2)
        ) + 1
        return moe_ep_buffer_size, moe_tp_buffer_size