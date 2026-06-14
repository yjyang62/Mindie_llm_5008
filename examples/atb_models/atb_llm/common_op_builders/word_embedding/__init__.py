# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.word_embedding.no_parallel_word_embedding_common_op_builder import \
    NoParallelWordEmbeddingCommonOpBuilder
from atb_llm.common_op_builders.word_embedding.all_gather_word_embedding_common_op_builder import \
    AllGatherWordEmbeddingCommonOpBuilder


CommonOpBuilderManager.register(AllGatherWordEmbeddingCommonOpBuilder)
CommonOpBuilderManager.register(NoParallelWordEmbeddingCommonOpBuilder)