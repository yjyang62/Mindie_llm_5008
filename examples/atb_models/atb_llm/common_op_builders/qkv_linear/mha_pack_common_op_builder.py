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

from atb_llm.common_op_builders.qkv_linear.base_qkv_linear_common_op_builder import BaseQKVLinearCommonOpBuilder
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.utils.singleton import Singleton


class MhaPackCommonOpBuilder(BaseQKVLinearCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        if not self.param.is_pack:
            return False
        if self.param.head_num != self.param.kv_head_num:
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        super().build(graph, tensor_map)

        self.param.linear_param.update({"linear_module": self.param.linear_modules[0]})
        qkv_linear_builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        qkv_linear_tensor_map = {
            "input": self.in_tensor_key.input,
            "linear_out": self.param.op_name + "_intermediate_mixed_qkv"
        }
        graph = qkv_linear_builder.build(graph, qkv_linear_tensor_map)

        split_op = atb.BaseOperation(
            op_type="Split",
            op_param=json.dumps({
                "splitDim": -1,
                "splitNum": 3
            }),
            op_name=self.param.op_name + "_Split"
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            [self.param.op_name + "_intermediate_mixed_qkv"],
            [self.out_tensor_key.q_out, self.out_tensor_key.k_out, self.out_tensor_key.v_out],
        )

        return graph