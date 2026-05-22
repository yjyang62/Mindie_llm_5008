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

import threading
from enum import Enum
from typing import Union
from llm_manager_python_api_demo.request_id import RequestId
from llm_manager_python_api_demo.dtype import DType, get_data_size_by_type
from llm_manager_python_api_demo.data import Data
from llm_manager_python_api_demo.status import Status, Code


class EndFlag(Enum):
    # 请求继续迭代执行
    RESPONSE_CONTINUE = 0
    # 请求正常结束
    RESPONSE_EOS = 1
    # 请求被主动CANCEL或STOP，用户不感知，丢弃响应
    RESPONSE_CANCEL = 2
    # 请求执行中出错，响应输出为空，err_msg非空
    RESPONSE_EXEC_ERROR = 3
    # 请求输入校验异常，响应输出为空，err_msg非空
    RESPONSE_ILLEGAL_INPUT = 4
    # 请求因达到最大序列长度而结束，响应为最后一轮迭代输出
    RESPONSE_REACH_MAX_SEQ_LEN = 5
    # 请求因达到最大输出长度（包括请求和模型粒度）而结束，响应为最后一轮迭代输出
    RESPONSE_REACH_MAX_OUTPUT_LEN = 6


class Response:
    def __init__(self, request_id: Union[str, int]):
        self.request_id = request_id
        self.eos_flag = False
        self.flags = 0
        self.outputs = {}
        self.mutex = threading.Lock()

    def add_output(self, data: Data):
        with self.mutex:
            if data.name in self.outputs:
                return Status(Code.INVALID_ARG, f"output '{data.name}' already exists in response")
            self.outputs[data.name] = data
            return Status(Code.OK)
    
    def del_output(self, name):
        with self.mutex:
            if name not in self.outputs or self.outputs.pop(name, None) is None:
                return Status(Code.INVALID_ARG, f"output '{name}' does not exist in response")
            return Status(Code.OK)
    
    def parse_eos_attr(self):
        ibis_eos_attr = self.get_outputs().get("IBIS_EOS_ATTR")
        if ibis_eos_attr is None:
            return Status(Code.ERROR, "Failed to get eos info")
        if ibis_eos_attr.get_type() != DType.TYPE_INT64 or ibis_eos_attr.get_data() is None:
            return Status(Code.ERROR, "Failed to get eos info due to dtype is not INT64 or data is null")

        flag = [seq_eos_attr[0] for seq_eos_attr in ibis_eos_attr.get_data()]
        output_len = [seq_eos_attr[1] for seq_eos_attr in ibis_eos_attr.get_data()]
        return flag, output_len

    def get_output_id(self):
        _, req_token_num = self.parse_eos_attr()
        token_num = sum(req_token_num)
        expect_data_size = token_num * get_data_size_by_type(DType.TYPE_INT64)

        output_ids = self.get_outputs().get("OUTPUT_IDS")
        if output_ids is None:
            return Status(Code.ERROR, "Failed to get output_ids")

        if output_ids.get_data_size() < expect_data_size or output_ids.get_data() is None:
            return Status(Code.ERROR, "Failed to get output_ids due to data_size is wrong or data is null")

        output_ids = output_ids.get_data()
        return output_ids
    
    def get_top_logprobs(self):
        top_logprobs = self.get_outputs().get("TOP_LOGPROBS")
        if top_logprobs is None:
            return Status(Code.ERROR, "Failed to get top logprobs")
        return top_logprobs.get_data()
    
    def get_logprobs(self):
        output_logprobs = self.get_outputs().get("OUTPUT_LOGPROBS")
        if output_logprobs is None:
            return Status(Code.ERROR, "Failed to get output logprobs")
        return output_logprobs.get_data()
    
    def get_seq_id(self):
        seqs_id = self.get_outputs().get("IBIS_SEQS_ID")
        if seqs_id is None:
            return Status(Code.ERROR, "Failed to get sequence id")
        return seqs_id.get_data()
    
    def get_parent_seq_id(self):
        parent_seqs_id = self.get_outputs().get("PARENT_SEQS_ID")
        if parent_seqs_id is None:
            return Status(Code.ERROR, "Failed to get parent sequence id")
        return parent_seqs_id.get_data()
    
    def get_cumulative_logprobs(self):
        cumulative_logprobs = self.get_outputs().get("CUMULATIVE_LOGPROBS")
        if cumulative_logprobs is None:
            return Status(Code.ERROR, "Failed to get cumulative logprobs id")
        return cumulative_logprobs.get_data()
        
    def set_flags(self, flags):
        self.flags = flags

    def set_eos(self, is_final):
        self.eos_flag = is_final

    def is_eos(self):
        return self.eos_flag

    def get_flag(self):
        return self.flags
    
    def get_outputs(self) -> dict:
        return self.outputs

    def get_request_id(self) -> RequestId:
        return self.request_id