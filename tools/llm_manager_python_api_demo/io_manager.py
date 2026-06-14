#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
from collections import deque
from threading import Lock
from typing import List, Optional, Dict
from llm_manager_python_api_demo.data import Data
from llm_manager_python_api_demo.data_loader import load_data, convert_data_list
from mindie_llm.utils.log.logging import logger, print_log
from mindie_llm.utils.env import ENV


class IOManager:
    def __init__(self):
        self.__inputs: deque[Data] = deque()
        self.mtx = Lock()
        self.__using_data: Dict[str: Data] = {}

    def set_input_data(self, dataset: str) -> int:
        resolved_path = os.path.realpath(dataset)
        if not resolved_path:
            print_log(ENV.rank, logger.error, "error: The path of the dataset is invalid!")
            return -1
        data = load_data(resolved_path)
        converted_data = convert_data_list(data)
        with self.mtx:
            self.__inputs.extend(converted_data)
        return 0

    def empty(self) -> bool:
        with self.mtx:
            return len(self.__inputs) == 0

    def get_input_data(self, n: int) -> List[Optional[Data]]:
        with self.mtx:
            ret: List[Data] = []
            for _ in range(n):
                if len(self.__inputs) != 0:
                    tmp_data = self.__inputs.popleft()
                    ret.append(tmp_data)
                    self.__using_data[tmp_data.get_name()] = tmp_data
            return ret

    def get_warmup_inputs(self, n: int) -> List[Optional[Data]]:
        with self.mtx:
            ret: List[Data] = []
            for i in range(min(n, len(self.__inputs))):
                if len(self.__inputs) != 0:
                    tmp_data = self.__inputs[i]
                    ret.append(tmp_data)
            return ret

    def get_input_data_by_quotas(self, remain_prefill_slots: int,
                                 remain_prefill_tokens: int, slot_num: int) -> List[Optional[Data]]:
        with self.mtx:
            ret: List[Data] = []
            while len(self.__inputs) != 0:
                tmp_data = self.__inputs[0]
                demand_token_num = tmp_data.get_data_size()
                if remain_prefill_slots > 0 and remain_prefill_tokens >= demand_token_num and slot_num > 0:
                    ret.append(tmp_data)
                    self.__using_data[tmp_data.get_name()] = tmp_data
                    self.__inputs.popleft()
                    remain_prefill_slots -= 1
                    remain_prefill_tokens -= demand_token_num
                    slot_num -= 1
                else:
                    break
            return ret

    def set_output_data(self, id_str: str) -> None:
        with self.mtx:
            self.__using_data.pop(id_str, None)
    
    def get_inputs(self) -> deque[Data]:
        return self.__inputs