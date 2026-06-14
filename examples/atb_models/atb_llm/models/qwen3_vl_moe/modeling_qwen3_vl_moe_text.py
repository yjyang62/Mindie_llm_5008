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
from typing import Optional, List, Tuple
import torch
from torch import nn
import torch_npu
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.base.modeling import FlashLayer
from atb_llm.models.base.graph_manager.graph_manager import ATBGraphManager
from atb_llm.models.base.inputs_modifier.qlen_modifier import QLenModifier
from atb_llm.models.base.graph_manager import SpeculateGraphWrapper
from atb_llm.models.qwen3_vl.modeling_qwen3_vl_text import Qwen3VLTextRotaryEmbedding
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import TensorParallelColumnLinear, TensorParallelRowLinear, RMSNorm, \
    TensorEmbedding, get_linear, load_column_multi, AttentionMask
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.layers.linear import FastLinear
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper
from atb_llm.utils.quantize.pack_type import calc_linear_pack_type
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.utils.log import logger

A2_SOCS = (220, 221, 222, 223, 224, 225)
A3_SOCS = (250, 251, 252, 253, 254, 255)


class TensorParallelColumnStackedMOE(TensorParallelColumnLinear):
    """Tensor parallel column linear for stacked MOE, shards expert gate-up weights."""

    @staticmethod
    def get_col_packed_mlp(prefix, weights):
        slice_ = weights.get_tensor(prefix).transpose(1, 2)
        total_size = slice_.shape[1]
        pack_num = 2
        if total_size % pack_num != 0:
            err_msg = "Prepacked mlp is not divisible by gate and up"
            logger.error(err_msg)
            raise AssertionError(err_msg)
        gate_or_up_size = total_size // pack_num
        world_size = weights.process_group.size()
        rank = weights.process_group.rank()
        if gate_or_up_size % world_size != 0:
            err_msg = f"Prepacked mlp cannot be sharded across {world_size} shards"
            logger.error(err_msg)
            raise AssertionError(err_msg)
        gate_layer, up_layer = slice_.split((gate_or_up_size, gate_or_up_size), dim=1)
        gate_list = torch.chunk(gate_layer, world_size, dim=1)
        up_list = torch.chunk(up_layer, world_size, dim=1)
        tensor = torch.cat([gate_list[rank], up_list[rank]], dim=1)
        return tensor

    @classmethod
    def load_moe(cls, config, prefix_list: List[str], weights, bias: bool, **kwargs):
        if bias:
            err_msg = "Bias is not supported in stacked MOE down weights."
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
        if len(prefix_list) > 1:
            err_msg = "Stacked MOE Gate-up weight only support single prefix."
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
        weight = cls.get_col_packed_mlp(prefix_list[0], weights)
        bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)


class TensorParallelRowStackedMOE(TensorParallelRowLinear):
    """Tensor parallel row linear for stacked MOE, shards expert down weights."""

    @classmethod
    def load_moe(cls, config, prefix_list: List[str], process_group, weights, bias: bool, **kwargs):
        if bias:
            err_msg = "Bias is not supported in stacked MOE Down weights."
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
        if len(prefix_list) > 1:
            err_msg = "Stacked MOE Down weight only support single prefix."
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
        weight = weights.get_sharded(prefix_list[0], dim=1).transpose(1, 2).contiguous()
        bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear, process_group=process_group)


class Qwen3VLMOETextAttention(nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.config = config
        config_quantize_change_flag = False
        if config.quantize == QuantType.W8A8_DYNAMIC:
            config_quantize_change_flag = True
            cache_quantize = config.quantize
            config.quantize = QuantType.W8A8
            weights.quantize = QuantType.W8A8
        self.qkv_names = [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        dense_name = f"{prefix}.o_proj"
        self.head_size = self.config.head_dim
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.input_layernorm'
        self.pack_type = calc_linear_pack_type(weights, self.qkv_names, self.norm_name)
        self.query_key_value = load_column_multi(
            config,
            prefixes=self.qkv_names,
            weights=weights,
            head_size=self.head_size,
            bias=False
        )
        self.q_norm = RMSNorm(prefix=f"{prefix}.q_norm", weights=weights, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(prefix=f"{prefix}.k_norm", weights=weights, eps=config.rms_norm_eps)
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=dense_name,
            weights=weights,
            bias=False,
            gqa_size=self.head_size
        )
        if config_quantize_change_flag:
            config.quantize = cache_quantize
            weights.quantize = cache_quantize


class Qwen3VLMOETextMOE(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        layer_prefix = '.'.join(prefix.split('.')[:-1])
        self.norm_name = f'{layer_prefix}.post_attention_layernorm'
        self.gate = FastLinear.load(
            prefix=f"{prefix}.gate",
            weights=weights,
            bias=False,
        )
        expert_prefix = f"{prefix}.experts"
        if config.quantize is None or config.quantize == QuantType.FLOAT:
            self.gate_up_proj = TensorParallelColumnStackedMOE.load_moe(
                config,
                prefix_list=[f"{expert_prefix}.gate_up_proj"],
                weights=weights,
                bias=False
            )
            self.down_proj = TensorParallelRowStackedMOE.load_moe(
                config,
                prefix_list=[f"{expert_prefix}.down_proj"],
                process_group=weights.process_group,
                weights=weights,
                bias=False
            )
            self.pack_type = PackType.ALL_FP
        else:
            self.device_expert = [i for i in range(config.num_experts)]
            pack_prefixes = [[f"{expert_prefix}.{i}.gate_proj", f"{expert_prefix}.{i}.up_proj"] \
                            for i in self.device_expert]
            self.gate_up_proj = TensorParallelColumnLinear.load_moe(
                config,
                prefix_list=pack_prefixes,
                weights=weights,
                bias=False
            )
            self.down_proj = TensorParallelRowLinear.load_moe(
                config,
                prefix_list=[f"{expert_prefix}.{i}.down_proj" for i in self.device_expert],
                process_group=weights.process_group,
                weights=weights,
                bias=False
            )
            linear_names = [f"{prefix}.experts.{0}.up_proj", f"{prefix}.experts.{0}.gate_proj"]
            self.pack_type = calc_linear_pack_type(weights, linear_names, self.norm_name)


class Qwen3VLTextDecoderLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, prefix):
        super().__init__(layer_id, config, weights, prefix)
        self.self_attn = Qwen3VLMOETextAttention(
                prefix=f"{self.prefix}.self_attn", config=config, weights=weights
            )
        self.mlp = Qwen3VLMOETextMOE(prefix=f"{self.prefix}.mlp", config=config, weights=weights)
        self.load_weights()


class FlashQwen3VLMOETextModelForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        prefix = "model.language_model"
        self.multi_query_group_num = self.config.num_key_value_heads
        self.embed_tokens = TensorEmbedding(prefix=f"{prefix}.embed_tokens", weights=weights)
        self.layers = nn.ModuleList(
            [
                Qwen3VLTextDecoderLayer(layer_id, config, weights, prefix)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps)
        self.lm_head = load_column_multi(
            self.config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.attn_mask = AttentionMask.static(self.max_base_len)
        self.rotary_pos_emb = Qwen3VLTextRotaryEmbedding(self.config, device="cpu")
        self.cos_embed, self.sin_embed = self.rotary_pos_emb(self.config.max_position_embeddings)
        self.cos_embed = self.cos_embed.to(self.dtype).to(self.device)
        self.sin_embed = self.sin_embed.to(self.dtype).to(self.device)
        self.cos_embed_decode = torch.concat((self.cos_embed, self.cos_embed), dim=-1).unsqueeze(1)
        self.sin_embed_decode = torch.concat((self.sin_embed, self.sin_embed), dim=-1).unsqueeze(1)
        self.mrope_section = self.config.rope_scaling.mrope_section
        self.ascend_weight = []
        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.expert_array = None
        self.expert_group = torch.tensor([1], dtype=torch.int32).npu()
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.soc_info = NPUSocInfo()
        if self.soc_info.need_nz:
            self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
            self.transdata_param = json.dumps({})
            self.transdata_operation.set_param(self.transdata_param)
        self._update_matmul_params(self.quantize)
        # Multi graph management
        self.graph_manager = ATBGraphManager()
        self.qlen_decorator = QLenModifier()

    def init_ascend_operations(self, config):
        pass

    def register_layer_weights(self, weight_wrapper, layer, is_dense_layer=False):
        attn_quantize_type = QuantType.W8A8 if self.quantize == QuantType.W8A8_DYNAMIC else self.quantize
        weight_wrapper.soc_info.matmul_nd_nz = self.matmul_nd_nz
        weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=is_dense_layer,
                                          attn_quantize_type=attn_quantize_type,
                                          qk_norm=True)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=None,
            down_name='down_proj',
            shared_experts=False,
        )
        weight_wrapper = MoeWeightWrapper(
            self.soc_info,
            self.tp_rank,
            attn_wrapper,
            mlp_wrapper,
            self.config.num_experts
        )
        weight_wrapper.register_embedding(self.embed_tokens)
        for i in range(self.config.num_hidden_layers):
            layer = self.layers[i]
            self.register_layer_weights(weight_wrapper, layer)
        weight_wrapper.register_model_norm(self.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        is_w8a8_dynamic = self.quantize == QuantType.W8A8_DYNAMIC
        weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        pack_quant_types = weight_wrapper.pack_quant_type
        attn_linear_types = weight_wrapper.attn_linear_types
        mlp_linear_types = weight_wrapper.mlp_linear_types
        moe_linear_types = weight_wrapper.moe_linear_types
        attn_linear_transpose_types = weight_wrapper.attn_linear_transpose_types
        mlp_linear_transpose_types = weight_wrapper.mlp_linear_transpose_types
        moe_linear_transpose_types = weight_wrapper.moe_linear_transpose_types
        for i in range(self.num_layers):
            attn_linear_types[i].append(attn_linear_types[i][-1])
            attn_linear_transpose_types[i].append(-1)
        coder_param = {
            "enableAddNorm": False,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.config.num_attention_heads // self.tp_world_size,
            "hiddenSizePerAttentionHead": self.config.head_dim,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": max(self.multi_query_group_num // self.tp_world_size, 1),
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_types,
            "weightQuantType": self.config.quantize if self.config.quantize else "",
            "quantGroupSize": self.config.quantization_config.group_size,
            "linearQuantType": attn_linear_types,
            "mlpLinearQuantType": mlp_linear_types,
            "moeLinearQuantType": moe_linear_types,
            "linearTransposeType": attn_linear_transpose_types,
            "mlpLinearTransposeType": mlp_linear_transpose_types,
            "moeLinearTransposeType": moe_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "isUnpadInputs": True,
            "skipWordEmbedding": True,
            "isLmHeadParallel": True,
            "enableSwiGLU": True,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "numOfExperts": self.config.num_experts,
            "numOfSelectedExperts": self.config.num_experts_per_tok,
            "routingMethod": 'softMaxTopK' if self.soc_info.need_nz else 'integratedSoftmaxTopK',
            "enableFusedRouting": True,
            "isDenseLayer": self.config.is_dense_layer,
            "hasSharedExpert": False,
            "backend": self.soc_info.communication_backend,
            "mapping": self.mapping.to_dict_v2(),
            "positionEmbeddingType": PositionEmbeddingType.ROPE,
            "linearHasBias": [[False, False, False, False]] * self.config.num_hidden_layers,
            "useQKNorm": True,
            "enableInitQuant": True if (is_w8a8_dynamic and (not self.soc_info.need_nz)) else False,
        }
        encoder_param = {
            **coder_param,
            "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
            "enableGMMSwigluQuant": False
        }
        decoder_param = {
            **coder_param,
            "isPrefill": False,
            "enableLcoc": False,
            "enableGMMSwigluQuant": True if (is_w8a8_dynamic and (not self.soc_info.need_nz)) else False
        }
        if self.speculate_enable:
            self.graph_manager.register_graph(SpeculateGraphWrapper())
        specified_params = {"decode": decoder_param}
        self.graph_manager.set_param("qwen3vl_MoeDecoderModel", encoder_param, specified_params)
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

    def apply_interleaved_mrope(self, emb, mrope_section):
        emb_t = emb[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            emb_t[..., idx] = emb[dim, ..., idx]
        return emb_t

    def update_thw_cos_sin(self, position_ids_thw):
        cos = self.cos_embed[position_ids_thw]
        sin = self.sin_embed[position_ids_thw]
        cos = self.apply_interleaved_mrope(cos, self.mrope_section)
        sin = self.apply_interleaved_mrope(sin, self.mrope_section)
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)
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
        self.expert_array = torch.tensor(
                            [j for j in range(len(input_ids) * self.config.num_experts_per_tok)],
                            dtype=torch.int32).npu()
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
        self.acl_operation_inputs.extend([
            self.expert_array,
            self.expert_group,
            self.one_hot,
            self.zero_hot
        ])
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
        deepstack_visual_embeds = kwargs.pop("deepstack_visual_embeds", [])
        if is_prefill:
            cos, sin = self.update_thw_cos_sin(position_ids_thw)
            acl_inputs[1] = torch.arange(input_lengths.sum(), dtype=position_ids.dtype, device=position_ids.device)
            acl_inputs.extend(deepstack_visual_embeds)
        else:
            cos, sin = self.cos_embed_decode, self.sin_embed_decode
        acl_inputs[2] = cos
        acl_inputs[3] = sin
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits

    def _update_matmul_params(self, quantize: QuantType):
        if self.soc_info.soc_version in (A2_SOCS + A3_SOCS):
            is_float = quantize is None or quantize == QuantType.FLOAT
            self.matmul_nd_nz = not is_float
        else:
            self.matmul_nd_nz = False
        logger.info(f"Qwen3_vl_moe: matmul_nd_nz is: {self.matmul_nd_nz}")