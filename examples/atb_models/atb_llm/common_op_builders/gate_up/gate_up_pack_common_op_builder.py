# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import _libatb_torch as atb

from atb_llm.common_op_builders.gate_up.base_gate_up_common_op_builder import BaseGateUpCommonOpBuilder
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.utils.singleton import Singleton


class GateUpPackCommonOpBuilder(BaseGateUpCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if not self.param.is_pack:
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        super().build(graph, tensor_map)

        linear_builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        linear_tensor_map = {
            "input": self.in_tensor_key.input,
            "linear_out": self.out_tensor_key.gate_up_out
        }
        graph = linear_builder.build(graph, linear_tensor_map)

        return graph