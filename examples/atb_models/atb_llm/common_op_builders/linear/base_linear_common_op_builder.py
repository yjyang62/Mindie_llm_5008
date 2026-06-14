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
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.layers.linear.linear_utils import LinearUtils
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.env import ENV
from atb_llm.common_op_builders.data_type import CommonOpBuilderType


class BaseLinearCommonOpBuilderParam(BaseCommonOpBuilderParam):
    linear_module: LinearUtils = Field(None)


class BaseLinearCommonOpBuilderInTensor(BaseModel):
    input: str = Field(...)


class BaseLinearCommonOpBuilderOutTensor(BaseModel):
    linear_out: str = Field(...)


class BaseLinearCommonOpBuilder(BaseCommonOpBuilder):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.LINEAR

    @property
    def param_cls(self):
        return BaseLinearCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return BaseLinearCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return BaseLinearCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        if self.param.linear_module.linear_desc == LinearTypeV2.INVALID:
            print_log(ENV.rank, logger.info,
                      f"CommonOpBuilder[{self.common_op_builder_name}] linear_type invalid")
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.model_validate(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.model_validate(tensor_map)
        return graph