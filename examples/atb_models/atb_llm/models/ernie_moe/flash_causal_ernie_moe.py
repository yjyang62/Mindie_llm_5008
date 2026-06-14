# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
from atb_llm.models.ernie_moe.config_ernie_moe import ErniemoeConfig
from atb_llm.models.ernie_moe.modeling_ernie_moe import FlashErniemoeModel
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger
from atb_llm.utils.weights import ProcessGroupType
from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.moe_utils import assign, ExpertParallelDegree

CPP_DECODER_MODEL = "ernie_moe_DecoderModel"


class FlashErniemoeForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.distributed_enable = kwargs.get('distributed_enable', False)
        self.max_batch_size = kwargs.get('max_batch_size', 0)
        if not self.distributed_enable:
            self.max_batch_size = 0
        super().__init__(config, weights, **kwargs)
        if self.llm_config is not None:
            self.model_config = self.llm_config.llm
            self.parallel_config = self.llm_config.llm.parallel_options
        else:
            self.model_config = None
            self.parallel_config = None
        if self.mapping.has_moe_ep():
            if not hasattr(self.model_config, "ep_level"):
                logger.warning("`ep_level` is not set in the model configuration file, "
                               "so it will be set to default value 1.")
                self.ep_level = ExpertParallelDegree.STATIC_EP
            else:
                self.ep_level = self.model_config.ep_level
            if self.ep_level == ExpertParallelDegree.DYNAMIC_EP:
                self.dep_communication_backend = {"prefill": "hccl", "decode": "hccl"} if self.model_config is None \
                    else {"prefill": self.model_config.communication_backend.prefill,
                          "decode": self.model_config.communication_backend.decode}
        else:
            self.ep_level = ExpertParallelDegree.NO_EP
        config.ep_level = self.ep_level
        self.model = FlashErniemoeModel(config, weights, model_config=self.model_config)
        weights.switch_process_group(ProcessGroupType.LM_HEAD)
        self.lm_head = load_column_multi(
            config,
            prefixes=["model.embed_tokens" if config.tie_word_embeddings else "lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.acl_param_encoder = None
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []
        self.ascend_weight = None

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.expert_array = self.placeholder
        self.expert_group = torch.tensor([0], dtype=torch.int32).npu()
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.num_experts = config.moe_num_experts
        self.num_shared_experts = config.moe_num_shared_experts
        self.num_experts_per_tok = config.moe_k
        self.moe_layer_start_index = config.moe_layer_start_index
        self.enable_fused_routing = True
        if hasattr(config, "attn_quantize"):
            self.attn_quantize = config.attn_quantize
        elif self.quantize == "w8a8_dynamic":
            self.attn_quantize = "w8a8"
        else:
            self.attn_quantize = self.quantize
        self.device_experts = assign(self.num_experts, self.mapping.moe_ep.group_size)[self.mapping.moe_ep.rank]
        self.num_device_experts = len(self.device_experts)
        self.start_device_expert_id = torch.tensor(self.device_experts[0], dtype=torch.int64).npu().view(-1)
        self.max_device_expert_id = torch.tensor([len(self.device_experts) - 1], dtype=torch.int64).npu().view(-1)
        logger.info(f"Expert parallel level is {self.ep_level}.")
        logger.info(f"Experts of rank {self.mapping.moe_ep.rank} are {self.device_experts}.")

    def init_ascend_operations(self, config: ErniemoeConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_DECODER_MODEL)
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_DECODER_MODEL)

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
            shared_experts=self.num_shared_experts > 0
        )
        weight_wrapper = MoeWeightWrapper(self.soc_info, self.tp_rank,
                                          attn_wrapper, moe_mlp_wrapper,
                                          self.num_experts)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            if i < self.moe_layer_start_index:
                weight_wrapper.register_moe_layer(
                    layer, self.quantize, dense_layer=True, attn_quantize_type=self.attn_quantize)
            else:
                weight_wrapper.register_moe_layer(
                    layer, self.quantize, dense_layer=False, attn_quantize_type=self.attn_quantize)
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

        for i in range(self.num_layers):
            attn_linear_types[i].append(attn_linear_types[i][-1])
            attn_linear_transpose_types[i].append(-1)

        is_w8a8_dynamic = self.quantize == "w8a8_dynamic"
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
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "rank": self.mapping.rank,
            "numOfExperts": self.num_experts,
            "numOfSelectedExperts": self.num_experts_per_tok,
            "numOfDeviceExperts": self.num_device_experts,
            "hasSharedExpert": self.num_shared_experts > 0,
            "routingMethod": "topkFused",
            "processLogits": "normalization",
            "firstKDenseReplace": self.moe_layer_start_index,
            "worldSize": self.mapping.world_size,
            "mapping": self.mapping.to_dict_v2(),
            "backend": self.soc_info.communication_backend,
            "rankTableFile": ENV.rank_table_file,
            "enableAddNorm": False,
            "normHasBias": False,
            "enableFusedRouting": self.enable_fused_routing,
            "expertParallelDegree": self.ep_level,
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
            "enableInitQuant": True if is_w8a8_dynamic else False,
            "maxDecodeDpTokenSize": self.max_batch_size,
            "enableDpOut": ENV.enable_dp_partition_up,
        }

        encoder_param = {
            **coder_param, "isPrefill": True, "enableLcoc": self.lcoc_enable, "enableGMMSwigluQuant": False
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "enableLcoc": False, 
            "enableGMMSwigluQuant": True if (is_w8a8_dynamic and (not self.soc_info.need_nz)) else False
        }
        self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
        self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        self.acl_encoder_operation.set_weight(self.ascend_weight)
        self.acl_decoder_operation.set_weight(self.ascend_weight)

    def init_position_rotary_embedding(self, position_ids, max_seq_len):
        self.rotary_embedding.update_cohere_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

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
        if self.mapping.has_dp() and self.mapping.has_mlp_tp():
            if ENV.enable_dp_move_up and not ENV.enable_dp_partition_up:
                token_size_per_dp_group = kwargs.get("token_size_per_dp_group", None)
                token_size = token_size_per_dp_group.sum().item()
            else:
                token_size = len(input_ids)
            shard_effective_token_indices = kwargs.get("shard_effective_token_indices", self.placeholder)
            token_index_with_padding = kwargs.get("token_index_with_padding", self.placeholder)
            skip_padding_token_indices = kwargs.get("skip_padding_token_indices", self.placeholder)
            final_hidden_states = torch.empty(
                [token_size, self.config.hidden_size], dtype=kv_cache[0][0].dtype, device=input_ids.device)

        q_lens = kwargs.get('q_lens', [])
        input_length = len(input_ids)

        if not self.enable_fused_routing:
            self.expert_array = torch.arange(self.num_experts_per_tok * input_length,
                                             dtype=torch.int32, device=input_ids.device)

        attn_padding_idx = self.placeholder
        attn_unpadding_idx = self.placeholder
        ffn_padding_idx = self.placeholder
        ffn_unpadding_idx = self.placeholder
        lm_head_skip_padding_token_indices = self.placeholder
        gather_prenorm_idx = self.placeholder
        dynamic_ep_idx = self.placeholder
        moe_idx = self.placeholder
        
        dep_inputs = [attn_padding_idx, attn_unpadding_idx, ffn_padding_idx, ffn_unpadding_idx,
                      lm_head_skip_padding_token_indices, gather_prenorm_idx,
                      self.start_device_expert_id, self.max_device_expert_id, dynamic_ep_idx, moe_idx,
                      self.placeholder]

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
                [j for j in range(input_length * self.num_experts_per_tok)], dtype=torch.int32).npu().view(-1)

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        if is_prefill:
            if self.soc_info.need_nz:
                pad_max = math.ceil(self.max_position_embeddings / 16) * 16
                attn_mask = self.attn_mask.get_attn_mask(pad_max, kv_cache[0][0].dtype, kv_cache[0][0].device)
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
                attn_mask,
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
                lm_head_indices.to(torch.int64) if self.mapping.has_dp() and self.mapping.has_mlp_tp()
                                                else self.lm_head_indices_fake,
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
