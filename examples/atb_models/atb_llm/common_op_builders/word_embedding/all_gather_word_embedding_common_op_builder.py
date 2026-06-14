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

from atb_llm.common_op_builders.word_embedding.base_word_embedding_common_op_builder import \
    BaseWordEmbeddingCommonOpBuilder
from atb_llm.utils.singleton import Singleton


class AllGatherWordEmbeddingCommonOpBuilder(BaseWordEmbeddingCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if not self.param.enable_parallel:
            return False
        if self.param.parallel_info.world_size <= 1:
            return False
        return True

    def build(self, graph: atb.GraphOperation, tensor_map: dict = None) -> None:
        super().build(graph, tensor_map)

        # gather算子
        super().add_gather(graph, f"{self.param.op_name}_gather_out")

        # all gather算子
        all_gather_op = atb.BaseOperation(
            op_type="AllGather",
            op_param=self.param.parallel_info.json(),
            op_name=f"{self.param.op_name}_AllGather"
        )
        graph.operations.append(all_gather_op)
        graph.add_operation(
            all_gather_op,
            [f"{self.param.op_name}_gather_out"],
            [f"{self.param.op_name}_all_gather_out"]
        )

        # transpose算子
        transpose_op = atb.BaseOperation(
            op_type="Transpose",
            op_param=json.dumps({"perm": [1, 0, 2] if self.param.unpad_inputs else [1, 2, 0, 3]}),
            op_name=f"{self.param.op_name}_Transpose"
        )
        graph.operations.append(transpose_op)
        graph.add_operation(
            transpose_op,
            [f"{self.param.op_name}_all_gather_out"],
            [self.out_tensor_key.word_embedding_out]
        )