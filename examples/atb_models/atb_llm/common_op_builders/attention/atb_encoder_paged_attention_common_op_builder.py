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

import _libatb_torch as atb

from atb_llm.common_op_builders.data_type import OperationBackend
from atb_llm.common_op_builders.attention.paged_attention_common_op_builder import PagedAttentionCommonOpBuilder


class ATBEncoderPagedAttentionCommonOpBuilder(PagedAttentionCommonOpBuilder):

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if not self.param.is_prefill:
            return False
        if self.param.operation_backend != OperationBackend.ATB:
            return False
        if self.param.atb_attention_param.get("calcType") != "PA_ENCODER":
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        super().build(graph, tensor_map)

        graph.add_reshape(self.in_tensor_key.q, f"{self.param.op_name}_reshape_q", self.reshape_q)
        graph.add_reshape(self.in_tensor_key.k, f"{self.param.op_name}_reshape_k", self.reshape_kv)
        graph.add_reshape(self.in_tensor_key.v, f"{self.param.op_name}_reshape_v", self.reshape_kv)

        # reshape and cache
        graph = self.add_reshape_and_cache(graph, f"{self.param.op_name}_reshape_k", f"{self.param.op_name}_reshape_v")

        # self attention
        attention_op = atb.BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps(self.param.atb_attention_param),
            op_name=f"{self.param.op_name}_SelfAttention"
        )
        graph.operations.append(attention_op)

        input_key_list = [
            f"{self.param.op_name}_reshape_q", f"{self.param.op_name}_reshape_k", f"{self.param.op_name}_reshape_v"
        ]
        if self.param.atb_attention_param.get("maskType", "MASK_TYPE_UNDEFINED") != "MASK_TYPE_UNDEFINED":
            input_key_list.append(self.in_tensor_key.attention_mask)
        input_key_list.append(self.in_tensor_key.seq_len)
        if self.param.atb_attention_param.get("maskType") in ["MASK_TYPE_ALIBI_COMPRESS",
                                                              "MASK_TYPE_ALIBI_COMPRESS_SQRT",
                                                              "MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN"]:
            input_key_list.append(self.in_tensor_key.slopes)
        graph.add_operation(attention_op, input_key_list, [f"{self.param.op_name}_intermediate_attn_out"])

        graph.add_reshape(f"{self.param.op_name}_intermediate_attn_out", self.out_tensor_key.attention_out,
                          self.reshape_0_12)

        return graph