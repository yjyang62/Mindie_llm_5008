#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from typing import List, Optional

from mindie_llm.text_generator.utils.input_metadata import InputMetadata


@dataclass(slots=True)
class InputMetadataComposite:
    input_metadata: InputMetadata = None
    # 对应proto中blocks_to_copy字段，是一个二维数组，来自原来的"IBIS_BLOCK_COPY"字段，
    # 如果没传blocks_to_copy，保持_data["IBIS_BLOCK_COPY"]为None
    block_copy: Optional[List[List]] = None
    # 对应proto中blocks_to_swap_in,blocks_to_swap_out，是一个二维数组，来自原来的"IBIS_BLOCK_OP"字段，
    block_op: Optional[List[List]] = None

    ############################以下字段只有在mix类型时需要设置######################################
    # 这个batch中prefill和decode的请求数量，主要用于mix类型
    prefill_batch_size = 0
    decode_batch_size: int = 0

    ############################以下字段只有在pull_kv类型时需要设置######################################
    # 数组，存放(p_ip_int, p_src_block_tables, d_src_block_tables)这样的三元关系
    pull_kv_items: Optional[List] = None
