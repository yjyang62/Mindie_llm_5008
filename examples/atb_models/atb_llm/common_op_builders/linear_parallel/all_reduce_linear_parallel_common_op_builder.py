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

from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import \
    BaseLinearParallelCommonOpBuilder, BaseLinearParallelCommonOpBuilderParam, ParallelType, CommunicationBackend
from atb_llm.utils.singleton import Singleton


class AllReduceLinearParallelCommonOpBuilderParam(BaseLinearParallelCommonOpBuilderParam):
    enable_lcoc: bool = Field(False)


class AllReduceLinearParallelCommonOpBuilder(BaseLinearParallelCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    @property
    def param_cls(self):
        return AllReduceLinearParallelCommonOpBuilderParam

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.parallel_type != ParallelType.ALL_REDUCE:
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict) -> atb.GraphOperation:
        super().build(graph, tensor_map)
        builder = CommonOpBuilderManager.get_builder(self.param.linear_param)
        linear_tensor_map = {"input": self.in_tensor_key.input, "linear_out": self.out_tensor_key.linear_out}
        graph = builder.build(graph, linear_tensor_map)

        if self.param.parallel_info.world_size <= 1:
            return graph

        all_reduce_op = atb.BaseOperation(
            op_type="AllReduce",
            op_param=self.param.parallel_info.json(),
            op_name=self.param.op_name + "_AllReduce"
        )
        graph.operations.append(all_reduce_op)
        graph.add_operation(
            all_reduce_op,
            [self.out_tensor_key.linear_out],
            [self.out_tensor_key.linear_out]  # inplace write
        )

        return graph