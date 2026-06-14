#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional, List
from llm_manager_python_api_demo.data import Data, create_data
from mindie_llm.utils.file_utils import safe_open
from mindie_llm.utils.log.logging import logger, print_log
from mindie_llm.utils.env import ENV

MAX_TOKEN_ALLOWED = 51200000  # token数规格为51200000
MAX_TOKEN_BYTE_ALLOWED = MAX_TOKEN_ALLOWED * 8  # sizeof(int64_t) is 8 bytes


def load_data(filepath) -> list[list[int]]:
    data = []
    with safe_open(filepath, 'r') as infile:
        line_str = infile.readline()
        while line_str:
            s_data = line_str.split(',')
            tmp_data = [int(item) for item in s_data]
            data.append(tmp_data)
            line_str = infile.readline()
    return data


def convert_data_list(src_data: List[List[int]]) -> List[Data]:
    return [convert_data(item) for item in src_data]


def convert_data(src_data: List[int]) -> Optional[Data]:
    size = len(src_data)
    if size > 0 and size <= MAX_TOKEN_ALLOWED:
        data_obj = create_data(src_data, size)
        return data_obj
    print_log(ENV.rank, logger.error, "error: invalid size in convertData")
    raise RuntimeError("Invalid size in convertData")