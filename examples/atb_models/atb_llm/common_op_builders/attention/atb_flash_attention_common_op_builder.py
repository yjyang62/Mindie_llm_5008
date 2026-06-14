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

from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.data_type import OperationBackend
from atb_llm.common_op_builders.attention.base_attention_common_op_builder import \
    BaseAttentionCommonOpBuilder, BaseAttentionCommonOpBuilderInTensor, AttnType


class FlashAttentionCommonOpBuilderInTensor(BaseAttentionCommonOpBuilderInTensor):
    token_offset: str = Field(...)
    layer_id: str = Field(...)


class ATBFlashAttentionCommonOpBuilder(BaseAttentionCommonOpBuilder):

    @property
    def in_tensor_cls(self):
        return FlashAttentionCommonOpBuilderInTensor

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.attn_type != AttnType.FLASH_ATTENTION:
            return False
        if self.param.operation_backend != OperationBackend.ATB:
            return False
        return True

    def reshape_unsqueeze(self, org_shape):
        new_shape = [1]
        for s in org_shape:
            new_shape.append(s)
        return new_shape
    
    def reshape_fa_v(self, org_shape):
        return [org_shape[0], org_shape[1], self.param.atb_attention_param.get("kvHeadNum", 0), self.param.head_size]
        
    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        super().build(graph, tensor_map)

        graph.add_reshape(self.in_tensor_key.v, f"{self.param.op_name}_v", self.reshape_fa_v)
        graph.add_reshape(self.in_tensor_key.k_cache, f"{self.param.op_name}_k_cache", self.reshape_unsqueeze)
        graph.add_reshape(self.in_tensor_key.v_cache, f"{self.param.op_name}_v_cache", self.reshape_unsqueeze)
        # self attention
        attention_op = atb.BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps(self.param.atb_attention_param),
            op_name=f"{self.param.op_name}_SelfAttention"
        )
        graph.operations.append(attention_op)

        input_key_list = [
            self.in_tensor_key.q, self.in_tensor_key.k, f"{self.param.op_name}_v",
            f"{self.param.op_name}_k_cache", f"{self.param.op_name}_v_cache"
        ]
        if self.param.atb_attention_param.get("maskType", "MASK_TYPE_UNDEFINED") != "MASK_TYPE_UNDEFINED":
            input_key_list.append(self.in_tensor_key.attention_mask)
        input_key_list.extend(
            [self.in_tensor_key.token_offset, self.in_tensor_key.seq_len, self.in_tensor_key.layer_id])
        if self.param.atb_attention_param.get("maskType") in ["MASK_TYPE_ALIBI_COMPRESS",
                                                              "MASK_TYPE_ALIBI_COMPRESS_SQRT",
                                                              "MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN"]:
            input_key_list.append(self.in_tensor_key.slopes)
        graph.add_operation(attention_op, input_key_list, [self.out_tensor_key.attention_out])

        return graph