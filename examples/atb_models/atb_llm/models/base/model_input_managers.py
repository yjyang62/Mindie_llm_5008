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
import torch
import torch_npu
from atb_llm.utils.layers import AttentionMask
from atb_llm.utils.layers import PositionRotaryEmbedding
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from atb_llm.utils.log.logging import logger, ErrorCode, print_log
from atb_llm.utils.initial import NPUSocInfo


class AttentionMaskGenerator:
    def __init__(self, mindie_llm_config, model_status, torch_device, max_base_len=128) -> None:
        self.max_base_len = max_base_len
        self.hf_config = mindie_llm_config.hf_config
        self.model_status = model_status
        self.torch_dtype = mindie_llm_config.hf_config.torch_dtype
        self.torch_device = torch_device
        self.soc_info = NPUSocInfo()
        self.position_embedding_type = self.model_status.position_embedding_type
        self.rank = mindie_llm_config.mapping.attn_tp.rank

        self.attention_mask_ins = AttentionMask.static(self.max_base_len, dtype=self.torch_dtype)
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)
    
    def generate_mask(self, attn_mask, is_prefill, **kwargs):
        if attn_mask is None:
            if is_prefill:
                attn_mask = self.generate_prefill_mask(**kwargs)
            else:
                attn_mask = self.generate_decode_mask(**kwargs)

        if self.soc_info.need_nz:
            attn_mask = self.transdata_operation.execute([attn_mask])[0]
        
        return attn_mask
    
    def generate_prefill_mask(self, **kwargs) -> None:
        """
        Generates the prefill mask for the model based on the position embedding type.

        Args:
            kwargs: Additional keyword arguments for the mask generation process.
                `max_seq_len` is required when creating alibi mask.
        Returns:
            The generated prefill mask.
        """
        if self.position_embedding_type == PositionEmbeddingType.ROPE:
            return self.attention_mask_ins.get_rope_prefill_mask(
                self.max_base_len, self.torch_dtype, self.torch_device
            )
        elif self.position_embedding_type == PositionEmbeddingType.ALIBI:
            return self.attention_mask_ins.get_alibi_prefill_mask(
                kwargs.get("max_seq_len"), self.hf_config, self.model_status,
                self.torch_dtype, self.rank
            )
        else:
            error_msg = "Error: position_embedding_type is illegal"
            logger.error(error_msg, ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError(error_msg)

    def generate_decode_mask(self, **kwargs) -> None:
        """
        Generates the decode mask for the model based on the position embedding type.

        Args:
            kwargs: Additional keyword arguments for the mask generation process.
                `position_ids` is required when creating alibi mask.
        Returns:
            The generated prefill mask.
        """
        if self.position_embedding_type == PositionEmbeddingType.ROPE:
            return self.attention_mask_ins.get_rope_decode_mask(
                self.torch_dtype, self.torch_device
            )
        elif self.position_embedding_type == PositionEmbeddingType.ALIBI:
            return self.attention_mask_ins.get_alibi_decode_mask(
                kwargs.get("max_seq_len"), kwargs.get("position_ids", []).tolist(),
                self.hf_config, self.model_status,
                self.torch_dtype, self.rank
            )
        else:
            error_msg = "Error: position_embedding_type is illegal"
            logger.error(error_msg, ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError(error_msg)


class PositionEmbeddingGenerator:
    def __init__(self, mindie_llm_config, model_status, torch_device) -> None:
        self.hf_config = mindie_llm_config.hf_config
        self.model_status = model_status
        self.torch_dtype = mindie_llm_config.hf_config.torch_dtype
        self.torch_device = torch_device
        self.position_embedding_type = self.model_status.position_embedding_type
        self.rope_type = self.hf_config.rope_scaling.rope_type

        self.position_embedding_ins: PositionRotaryEmbedding = PositionRotaryEmbedding.static(
            dim=self.model_status.head_dim,
            base=self.hf_config.rope_theta,
            device="cpu",
            scaling_factor=self.hf_config.rope_scaling.factor).to(self.torch_device)
        
        self.cosine_table = None
        self.sine_table = None
        self.placeholder = torch.zeros(1, dtype=self.torch_dtype, device="npu")
        
    def generate_position_embedding(self, max_seq_len):
        if self.position_embedding_type == PositionEmbeddingType.ROPE:
            if self.rope_type == "linear":
                self.position_embedding_ins.update_cos_sin_cache_total(
                    self.torch_dtype, self.torch_device, max_seq_len)
            elif self.rope_type == "llama3":
                self.position_embedding_ins.update_llama3_cos_sin_cache_total(
                    self.hf_config, self.torch_dtype, self.torch_device, max_seq_len)
            elif self.rope_type == "yarn":
                self.position_embedding_ins.yarn_scaling_rotary_embedding(
                    self.hf_config, self.torch_device, max_seq_len)
            elif self.rope_type == "dynamic":
                self.position_embedding_ins.dynamic_ntk_rotary_embedding(
                    self.hf_config, max_seq_len, self.torch_device)
                    
            if self.rope_type == "yarn" or self.rope_type == "dynamic":
                self.cosine_table = self.placeholder
                self.sine_table = self.placeholder
            else:
                self.cosine_table = self.position_embedding_ins.get_cos_cached_total()
                self.sine_table = self.position_embedding_ins.get_sin_cached_total()
        elif self.position_embedding_type == PositionEmbeddingType.ALIBI:
            self.cosine_table = self.placeholder
            self.sine_table = self.placeholder
        else:
            logger.error("Error: position_embedding_type is illegal",
                         ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)


class KVCacheUpdater:
    def __init__(self, mindie_llm_config) -> None:
        self.k_cache_id: int | None = None
        self.k_cache_shape: list | None = None
        self.v_cache_id: int | None = None
        self.v_cache_shape: list | None = None
        self.mapping = mindie_llm_config.mapping
        self.soc_info = NPUSocInfo()

    def update_kv_cache(self, kv_cache, engine_wrappers):
        kcache_id_diff = self.k_cache_id != id(kv_cache[0][0])
        vcache_id_diff = self.v_cache_id != id(kv_cache[0][1])
        kcache_shape_diff = self.k_cache_shape != kv_cache[0][0].shape
        vcache_shape_diff = self.v_cache_shape != kv_cache[0][1].shape
        kcache_diff = not self.k_cache_id or kcache_id_diff or kcache_shape_diff
        vcache_diff = not self.v_cache_id or vcache_id_diff or vcache_shape_diff
        if kcache_diff or vcache_diff:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            print_log(self.mapping.rank, logger.info, f"k cache's shape {k_caches[0].shape=}")
            print_log(self.mapping.rank, logger.info, f"v cache's shape {v_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                print_log(self.mapping.rank, logger.info, 
                          f"kv cache's tensor format changes to {torch_npu.get_npu_format(k_caches[0])}")
            # set engine's kv cache
            caches = {}
            for i, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
                k_cache_name = f"layer_{i}_k_cache"
                v_cache_name = f"layer_{i}_v_cache"
                caches.update({k_cache_name: k_cache, v_cache_name: v_cache})
            
            for engine in engine_wrappers:
                engine.set_kv_caches(caches)

            # update kv cache's id and shape
            self.k_cache_id = id(kv_cache[0][0])
            self.v_cache_id = id(kv_cache[0][1])
            self.k_cache_shape = kv_cache[0][0].shape
            self.v_cache_shape = kv_cache[0][1].shape
        