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
import math
import torch
from torch import nn
import torch_npu
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.base.modeling import FlashLayer, MLP
from atb_llm.models.base.inputs_modifier.qlen_modifier import QLenModifier
from atb_llm.models.base.inputs_modifier.flash_comm_modifier import FlashCommModifier
from atb_llm.models.base.graph_manager import ATBGraphManager, SpeculateGraphWrapper, FlashCommGraphWrapper
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import TensorParallelRowLinear, RMSNorm, TensorEmbedding, TensorHead, load_column_multi, \
    AttentionMask
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.quantize.pack_type import calc_linear_pack_type
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.log import logger


_800_9000_SOCS = (100, 101, 102, 103, 104)
DUO_SOCS = (200, 201, 202, 203, 204, 205)
A2_SOCS = (220, 221, 222, 223, 224, 225)
A3_SOCS = (250, 251, 252, 253, 254, 255)


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        base = getattr(config, "rope_theta", 5000000)
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.double) / dim)).to(torch.float)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, max_seq_len):
        seq_idx = torch.arange(max_seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        emb = torch.outer(seq_idx, self.inv_freq)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class Qwen3VLTextAttention(nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.config = config
        if config.quantize == QuantType.W8A8SC:
            self.qkv_names = [f"{prefix}.query_key_value"]
        else:
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


class Qwen3VLTextMLP(MLP):
    def __init__(self, prefix, config, weights):
        super().__init__(prefix, config, weights)
        self.gate_up_names = [f'{prefix}.gate_proj', f'{prefix}.up_proj']
        self.pack_name = f'{prefix}.gate_up_proj'
        self.down_name = f'{prefix}.down_proj'
        self.load_weights()


class Qwen3VLTextDecoderLayer(FlashLayer):
    def __init__(self, layer_id, config, weights, prefix):
        super().__init__(layer_id, config, weights, prefix)
        self.self_attn = Qwen3VLTextAttention(
                prefix=f"{self.prefix}.self_attn", config=config, weights=weights
            )
        self.mlp = Qwen3VLTextMLP(prefix=f"{self.prefix}.mlp", config=config, weights=weights)
        self.load_weights()


class FlashQwen3VLTextModelForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        super().__init__(config, weights, **kwargs)
        self.config = config
        if self.quantize == QuantType.W8A8SC:
            prefix = "language_model"
        else:
            prefix = "model.language_model"
        self.multi_query_group_num = self.config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.embed_tokens = TensorEmbedding(prefix=f"{prefix}.embed_tokens", weights=weights)
        self.layers = nn.ModuleList(
            [
                Qwen3VLTextDecoderLayer(layer_id, config, weights, prefix)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps)
        if self.quantize == QuantType.W8A8SC:
            self.lm_head = TensorHead.load_weight(
                config,
                prefix=f"{prefix}.lm_head",
                weights=weights,
                is_norm=False,
            )
        else:
            if self.config.tie_word_embeddings:
                self.lm_head = load_column_multi(
                    self.config,
                    prefixes=[f"{prefix}.embed_tokens"],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
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
        self.rotary_pos_emb = Qwen3VLTextRotaryEmbedding(self.config, device="cpu")
        self.cos_embed, self.sin_embed = self.rotary_pos_emb(self.config.max_position_embeddings)
        self.cos_embed = self.cos_embed.to(self.dtype).to(self.device)
        self.sin_embed = self.sin_embed.to(self.dtype).to(self.device)
        self.cos_embed_decode = torch.concat((self.cos_embed, self.cos_embed), dim=-1).unsqueeze(1)
        self.sin_embed_decode = torch.concat((self.sin_embed, self.sin_embed), dim=-1).unsqueeze(1)
        self.mrope_section = self.config.rope_scaling.mrope_section
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
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.embed_tokens)
        for i in range(self.config.num_hidden_layers):
            layer = self.layers[i]
            weight_wrapper.register_layer(layer, self.config.quantize)
            weight_wrapper.register_model_norm(layer.self_attn.q_norm)  # q_norm
            weight_wrapper.register_model_norm(layer.self_attn.k_norm)  # k_norm
            if self.soc_info.need_nz and self.adapter_manager is None:
                del layer.self_attn
                del layer.post_attention_layernorm
                del layer.mlp
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
            "hiddenSizePerAttentionHead": self.config.head_dim,
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
            "linearHasBias": [[False, False, False, False]] * self.config.num_hidden_layers,
            "useQKNorm": True,
        }
        encoder_param = {
            **coder_param,
            "isPrefill": True,
            "enablePreFetchWeight": self.soc_info.soc_version in DUO_SOCS,  # Negative performance gains in A2
            "enableLcoc": self.lcoc_enable
        }
        decoder_param = {
            **coder_param,
            "isPrefill": False,
            "enableLcoc": False
        }
        if self.speculate_enable:
            self.graph_manager.register_graph(SpeculateGraphWrapper())
        if self.flash_comm_modifier.enable_flash_comm:
            self.graph_manager.register_graph(FlashCommGraphWrapper())
        specified_params = {"decode": decoder_param}
        self.graph_manager.set_param("qwen3vl_DecoderModel", encoder_param, specified_params)
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