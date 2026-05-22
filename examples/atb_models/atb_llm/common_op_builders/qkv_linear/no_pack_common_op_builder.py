# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from pydantic import Field

import _libatb_torch as atb

from atb_llm.common_op_builders.qkv_linear.base_qkv_linear_common_op_builder import \
    BaseQKVLinearCommonOpBuilder
from atb_llm.common_op_builders.linear.base_linear_common_op_builder import BaseLinearCommonOpBuilderInTensor
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.utils.singleton import Singleton


class NoPackCommonOpBuilderInTensor(BaseLinearCommonOpBuilderInTensor):
    input_k: str = Field(...)
    input_v: str = Field(...)


class NoPackCommonOpBuilder(BaseQKVLinearCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    @property
    def in_tensor_cls(self):
        return NoPackCommonOpBuilderInTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        if self.param.is_pack:
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        super().build(graph, tensor_map)
        # add q linear
        linear_module_key = "linear_module"
        input_key = "input"
        linear_out_key = "linear_out"
        self.param.linear_param.update({linear_module_key: self.param.linear_modules[0]})
        q_linear_builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        q_linear_tensor_map = {
            input_key: self.in_tensor_key.input,
            linear_out_key: self.out_tensor_key.q_out,
        }
        graph = q_linear_builder.build(graph, q_linear_tensor_map)
        # add k linear
        self.param.linear_param.update({linear_module_key: self.param.linear_modules[1]})
        k_linear_builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        k_linear_tensor_map = {
            input_key: self.in_tensor_key.input_k,
            linear_out_key: self.out_tensor_key.k_out,
        }
        graph = k_linear_builder.build(graph, k_linear_tensor_map)
        # add v linear
        self.param.linear_param.update({linear_module_key: self.param.linear_modules[2]})
        v_linear_builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        v_linear_tensor_map = {
            input_key: self.in_tensor_key.input_v,
            linear_out_key: self.out_tensor_key.v_out,
        }
        graph = v_linear_builder.build(graph, v_linear_tensor_map)
        return graph