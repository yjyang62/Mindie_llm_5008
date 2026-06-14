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
from atb_llm.utils.singleton import Singleton


class PositionalEmbeddingCommonOpBuilderInTensor(BaseModel):
    position_ids: str = Field(..., description="Position IDs")
    cos_table: str = Field(..., description="Cos table")
    sin_table: str = Field(..., description="Sin table")


class PositionalEmbeddingCommonOpBuilderOutTensor(BaseModel):
    cos_embedding: str = Field(..., description="Cos embedding")
    sin_embedding: str = Field(..., description="Sin embedding")


class PositionalEmbeddingCommonOpBuilder(BaseCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        self.category = CommonOpBuilderType.POSITIONAL_EMBEDDING

    @property
    def param_cls(self):
        return BaseCommonOpBuilderParam

    @property
    def in_tensor_cls(self):
        return PositionalEmbeddingCommonOpBuilderInTensor

    @property
    def out_tensor_cls(self):
        return PositionalEmbeddingCommonOpBuilderOutTensor

    def is_match(self, param: dict):
        if not super().verify_base_param(param):
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        self.in_tensor_key = self.in_tensor_cls.parse_obj(tensor_map)
        self.out_tensor_key = self.out_tensor_cls.parse_obj(tensor_map)

        cos_table_gather_op = atb.BaseOperation(
            op_type="Gather",
            op_param="{}",
            op_name=f"{self.param.op_name}_Gather_cosine_table"
        )

        sine_table_gather_op = atb.BaseOperation(
            op_type="Gather",
            op_param="{}",
            op_name=f"{self.param.op_name}_Gather_sine_table"
        )

        graph.operations.extend([cos_table_gather_op, sine_table_gather_op])

        graph.add_operation(
            cos_table_gather_op,
            [self.in_tensor_key.cos_table, self.in_tensor_key.position_ids],
            [self.out_tensor_key.cos_embedding]
        )

        graph.add_operation(
            sine_table_gather_op,
            [self.in_tensor_key.sin_table, self.in_tensor_key.position_ids],
            [self.out_tensor_key.sin_embedding]
        )

        return graph