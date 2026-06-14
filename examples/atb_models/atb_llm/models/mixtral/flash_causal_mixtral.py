# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.mixtral.config_mixtral import MixtralConfig
from atb_llm.models.mixtral.modeling_mixtral import FlashMixtralModel
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper
from atb_llm.utils.env import ENV
from atb_llm.utils.layers import (
    TensorEmbedding,
    load_column_multi,
)
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.moe_utils import assign, ExpertParallelDegree
from atb_llm.utils.weights import ProcessGroupType


class FlashMixtralForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        # called the:
        # self.init_ascend_operations

        self.distributed_enable = kwargs.get('distributed_enable', False)
        self.max_batch_size = kwargs.get('max_batch_size', 0)
        self.total_batch_size = self.max_batch_size
        if not self.distributed_enable:
            self.max_batch_size = 0

        super().__init__(config, weights, **kwargs)
        
        if self.llm_config is not None:
            self.mx_config = self.llm_config.llm
            self.parallel_config = self.llm_config.llm.parallel_options
        else:
            self.mx_config = None
            self.parallel_config = None

        self.ep = self.mapping.has_moe_ep()
        if self.ep:
            logger.info("For expert parallel, "
                    "the ep_level variable needs to be defined in the model configuration file."
                    "The available options are 1, 2, or 3.")
            if not hasattr(self.mx_config, "ep_level"):
                logger.info("Since you haven't set ep_level, we will set ep_level to 1 by default, " \
                            "which means static ep.")    
                self.ep_level = ExpertParallelDegree.STATIC_EP
            else:
                self.ep_level = self.mx_config.ep_level
        else:
            self.ep_level = ExpertParallelDegree.NO_EP
        config.ep_level = self.ep_level

        self.model = FlashMixtralModel(config, weights)
        self.config = config
        weights.switch_process_group(ProcessGroupType.LM_HEAD)        
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.config = config
        self.in_tensor_length = 17
        self.acl_operation_inputs = []

        self.placeholder = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=self.device)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.prefill_skip_word_embedding = False
        self.decoder_skip_word_embedding = False

        self.padding_idx = config.pad_token_id
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.hidden_dim = config.hidden_size
        self.expert_array = []
        self.expert_group = torch.tensor([1], dtype=torch.int32).npu()
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.final_bias = torch.zeros([self.config.n_routed_experts, self.config.hidden_size], dtype=self.dtype).npu()
        self.tp = config.tp if config.tp else True # False
        self.num_of_experts = config.n_routed_experts
        self.num_of_selected_experts = [config.num_experts_per_tok]
        self.ascend_weight_wrapper = None
        self.ascend_weight = None
        self.enable_fused_routing = False if self.soc_info.need_nz else True

        self.device_expert = assign(self.config.n_routed_experts, self.mapping.moe_ep.group_size)[
            self.mapping.moe_ep.rank]
        if self.ep:
            if self.ep_level in [ExpertParallelDegree.DYNAMIC_EP, ExpertParallelDegree.MIX_EP]:
                self.dep_communication_backend = {'prefill': 'hccl', 'decode': 'hccl'} if self.mx_config is None else {
                    'prefill': self.mx_config.communication_backend.prefill, 
                    'decode': self.mx_config.communication_backend.decode
                }
                self.p_to_d_weight = self.ep_level == ExpertParallelDegree.MIX_EP
            logger.info(f"Expert parallel level is {self.ep_level}.")
            logger.info(f"Experts of rank {self.mapping.moe_ep.rank} are: {self.device_expert}")

        self.num_of_device_expert = len(self.device_expert)
        self.start_device_expert_id = torch.tensor(self.device_expert[0], dtype=torch.int64).npu().view(-1)
        self.max_device_expert_id = torch.tensor([len(self.device_expert) - 1], dtype=torch.int64).npu().view(-1)

    # called by super().prepare_inputs_for_ascend
    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: MixtralConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("mixtral_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("mixtral_DecoderModel")
        pass

    def init_weight_wrapper(self):
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
            wrapper_name='block_sparse_moe',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=False
        )
        weight_wrapper = MoeWeightWrapper(self.soc_info, self.tp_rank,
                                          attn_wrapper, moe_mlp_wrapper,
                                          self.config.n_routed_experts)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        return weight_wrapper

    def get_weights(self):
        weight_wrapper = self.init_weight_wrapper()
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            if self.ep:
                weight_wrapper.register_moe_layer(layer, self.quantize,
                                expert_roster=[i for i, _ in enumerate(self.device_expert)])
                del layer.block_sparse_moe
                torch.npu.empty_cache()
            else:
                weight_wrapper.register_moe_layer(layer, self.quantize)
                del layer.block_sparse_moe
                torch.npu.empty_cache()
            if self.soc_info.need_nz:
                del layer.self_attn
                del layer.post_attention_layernorm
                torch.npu.empty_cache()
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        ascend_weight_wrapper = self.get_weights()
        self.ascend_weight = ascend_weight_wrapper.weights

        pack_quant_types = ascend_weight_wrapper.pack_quant_type
        attn_linear_types = ascend_weight_wrapper.attn_linear_types
        mlp_linear_types = ascend_weight_wrapper.mlp_linear_types
        moe_linear_types = ascend_weight_wrapper.moe_linear_types
        attn_linear_transpose_types = ascend_weight_wrapper.attn_linear_transpose_types
        mlp_linear_transpose_types = ascend_weight_wrapper.mlp_linear_transpose_types
        moe_linear_transpose_types = ascend_weight_wrapper.moe_linear_transpose_types
        
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
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_types,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "linearQuantType": attn_linear_types,
            "mlpLinearQuantType": mlp_linear_types,
            "moeLinearQuantType": moe_linear_types,
            "linearTransposeType": attn_linear_transpose_types,
            "mlpLinearTransposeType": mlp_linear_transpose_types,
            "moeLinearTransposeType": moe_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.mapping.rank,
            "expertParallelDegree": self.ep_level,
            "numOfExperts": self.config.n_routed_experts,
            "numOfDeviceExperts": self.num_of_device_expert,
            "deviceExpert": self.device_expert,
            "routingMethod": 'softMaxTopK' if self.soc_info.need_nz else 'integratedSoftmaxTopK',
            "numOfSelectedExperts": self.config.num_experts_per_tok,
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
            "isMlpFullTP": False,
            "enableDispatchCombineV2": True
        }
        if coder_param["routingMethod"] not in ['softMaxTopK', 'integratedSoftmaxTopK']:
            msg = "The routingMethod chosen is not valid, please choose among the following:\n \
                  'softMaxTopK': regular routing method with softmax and topk-sort operators\n \
                  'integratedSoftmaxTopK': routing method with the integration of softmax and topk-sort operators\n \
                  'deviceLimited': device-limited routing method (e.g. deepseekv2); \
                  invalid for Mixtral MoE and Deepseekv1"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
        
        if self.mapping is not None:
            if self.ep_level == ExpertParallelDegree.DYNAMIC_EP:
                min_moe_ep_buffer_size, min_moe_tp_buffer_size = self.calc_moe_buffer_size()
                if self.mapping.moe_ep.buffer_size < min_moe_ep_buffer_size:
                    self.mapping.moe_ep.buffer_size = min_moe_ep_buffer_size
                if self.mapping.moe_tp.buffer_size < min_moe_tp_buffer_size:
                    self.mapping.moe_tp.buffer_size = min_moe_tp_buffer_size
            coder_param.update({"mapping": self.mapping.to_dict_v2()})
        
        if self.ep_level == ExpertParallelDegree.DYNAMIC_EP and self.mapping.moe_tp.group_size > 1:
            msg = f"Dynamic EP only supports moe_tp = 1, but gets moe_tp = {self.mapping.moe_tp.group_size}."
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
        
        encoder_param = {
            **coder_param, "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
            "skipWordEmbedding": self.prefill_skip_word_embedding,
            "backend": self.soc_info.communication_backend if self.ep_level < 2 \
                            else self.dep_communication_backend['prefill'],
        }
        decoder_param = {
            **coder_param, "isPrefill": False,
            "enableLcoc": False,
            "skipWordEmbedding": self.decoder_skip_word_embedding,
            "backend": self.soc_info.communication_backend if self.ep_level < 2 \
                            else self.dep_communication_backend['decode'],
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
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                            self.device,
                                                            self.max_position_embeddings)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        if self.mapping.has_dp() and self.mapping.has_mlp_tp():
            if ENV.enable_dp_move_up and not ENV.enable_dp_partition_up:
                token_size_per_dp_group = kwargs.get("token_size_per_dp_group", None)
                token_size = token_size_per_dp_group.sum().item()
            else:
                token_size = len(input_ids)
            shard_effective_token_indices = kwargs.get("shard_effective_token_indices", None)
            token_index_with_padding = kwargs.get("token_index_with_padding", None)
            skip_padding_token_indices = kwargs.get("skip_padding_token_indices", None)
            final_hidden_states = torch.empty([token_size, self.config.hidden_size],
                                          dtype=kv_cache[0][0].dtype,
                                          device=input_ids.device)

        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=self.device)

        input_length = len(input_ids)
        self.expert_array = self.placeholder

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
            else:
                message = "Please export ENABLE_DP_MOVE_UP=1 when set attn_dp > 1."
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
            
        if is_prefill:
            if self.soc_info.need_nz:
                atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype,
                                                          kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.attn_mask.get_attn_mask(self.max_base_len, kv_cache[0][0].dtype,
                                                                    kv_cache[0][0].device)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                torch.where(atten_mask == -torch.inf, 1, atten_mask) if self.dtype == torch.bfloat16 else atten_mask,
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
                self.zero_hot,
            ]
            if self.mapping.has_dp() and self.mapping.has_mlp_tp():
                self.acl_operation_inputs.insert(12, final_hidden_states)
                self.acl_operation_inputs.insert(13, shard_effective_token_indices)
                self.acl_operation_inputs.insert(14, token_index_with_padding)
                self.acl_operation_inputs.insert(15, skip_padding_token_indices)
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                torch.zeros(input_lengths.size(0),
                            self.num_attention_heads,
                            1, input_lengths.max().item(),
                            dtype=self.dtype,
                            device=self.device) if self.dtype == torch.bfloat16 else self.attn_mask_fake,
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
                self.zero_hot,
            ]
            if self.mapping.has_dp() and self.mapping.has_mlp_tp():
                self.acl_operation_inputs.insert(12, final_hidden_states)
                self.acl_operation_inputs.insert(13, shard_effective_token_indices)
                self.acl_operation_inputs.insert(14, token_index_with_padding)
                self.acl_operation_inputs.insert(15, skip_padding_token_indices)

        self.acl_operation_inputs.extend(dep_inputs)
        return self.acl_operation_inputs, self.acl_param

    def calc_moe_buffer_size(self):
        if hasattr(self.mx_config, 'eplb') and self.mx_config.eplb is not None and \
            self.mx_config.eplb.num_redundant_experts is not None:
            num_redundant_experts = self.mx_config.eplb.num_redundant_experts
        else:
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

if __name__ == "__main__":
    test_config = MixtralConfig()
    test_weights = None
    model = FlashMixtralForCausalLM(test_config, test_weights)