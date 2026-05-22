# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from pydantic import BaseModel, Field

import _libatb_torch as atb

from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder, BaseCommonOpBuilderParam
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class BaseGateUpCommonOpBuilderParam(BaseCommonOpBuilderParam):
    is_pack: bool = Field(True)
    linear_param: dict | None = Field(None)


class BaseGateUpCommonOpBuilderInTensor(BaseModel):
    input: str = Field(...)


class BaseGateUpCommonOpBuilderOutTensor(BaseModel):
    gate_up_out: str = Field(...)


class BaseGateUpCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.GATE_UP

    @property
    def param_cls(self):
        return BaseGateUpCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseGateUpCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseGateUpCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)
        return graph