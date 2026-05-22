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

from llm_manager_python_api_demo import llm_manager_python
from llm_manager_python_api_demo.data import Data
from llm_manager_python_api_demo.status import Status, Code
from llm_manager_python_api_demo.request_id import RequestId
from llm_manager_python_api_demo.sampling import SamplingParams
from llm_manager_python_api_demo.dtype import DType, get_numpy_dtype_by_type
import numpy as np


class Request:
    def __init__(self, request_id: RequestId):
        self.request_id = request_id
        req_id = llm_manager_python.InferRequestId(request_id.id)
        self.request = llm_manager_python.InferRequest(req_id)
        self.inputs = {}
        self.max_output_len = 1024
        self.response_callback = None

    @staticmethod
    def set_sampling_data(name: str, sampling_type: DType, sampling_data):
        shape = np.array([1, 1], dtype=np.int64)
        data = Data()
        data.set_sampling(name, sampling_type, shape,
            np.array([sampling_data], dtype=get_numpy_dtype_by_type(sampling_type)))
        return data

    def get_immutable_inputs(self):
        return self.inputs

    def set_data_to_request(self, input_data: Data) -> Status:
        """
        在request中设置传入自定义的data数据
        :param input_data: 输入数据
        :return: status
        """
        self.inputs[input_data.get_name()] = input_data
        return Status(Code.OK)

    def set_input_token_num(self, token_num):
        shape = np.array([1], dtype=np.int64)
        data = Data()
        data.set_token_num(DType.TYPE_INT64, shape,
            np.array([token_num], dtype=np.int64))
        self.inputs[data.get_name()] = data
        return Status(Code.OK)

    def set_sampling_params(self, params: SamplingParams):
        """
        为request设置sampling参数
        :param params: sampling参数
        :return:
        """
        temperature = self.set_sampling_data("TEMPERATURE", DType.TYPE_FP32, params.temperature)
        self.inputs["TEMPERATURE"] = temperature
        top_k = self.set_sampling_data("TOP_K", DType.TYPE_INT32, params.top_k)
        self.inputs["TOP_K"] = top_k
        top_p = self.set_sampling_data("TOP_P", DType.TYPE_FP32, params.top_p)
        self.inputs["TOP_P"] = top_p
        typical_p = self.set_sampling_data("TYPICAL_P", DType.TYPE_FP32, params.typical_p)
        self.inputs["TYPICAL_P"] = typical_p
        do_sample = self.set_sampling_data("DO_SAMPLE", DType.TYPE_BOOL, params.do_sample)
        self.inputs["DO_SAMPLE"] = do_sample
        seed = self.set_sampling_data("SEED", DType.TYPE_UINT64, params.seed)
        self.inputs["SEED"] = seed
        repetition_penalty = self.set_sampling_data("REPETITION_PENALTY", DType.TYPE_FP32, params.repetition_penalty)
        self.inputs["REPETITION_PENALTY"] = repetition_penalty
        frequency_penalty = self.set_sampling_data("FREQUENCY_PENALTY", DType.TYPE_FP32, params.frequency_penalty)
        self.inputs["FREQUENCY_PENALTY"] = frequency_penalty
        presence_penalty = self.set_sampling_data("PRESENCE_PENALTY", DType.TYPE_FP32, params.presence_penalty)
        self.inputs["PRESENCE_PENALTY"] = presence_penalty
        watermark = self.set_sampling_data("WATERMARK", DType.TYPE_BOOL, params.watermark)
        self.inputs["WATERMARK"] = watermark
        if params.logprobs:
            logprobs = self.set_sampling_data("LOGPROBS", DType.TYPE_BOOL, params.logprobs)
            self.inputs["LOGPROBS"] = logprobs
        if params.top_logprobs:
            top_logprobs = self.set_sampling_data("TOPLOGPROBS", DType.TYPE_INT32, params.top_logprobs)
            self.inputs["TOPLOGPROBS"] = top_logprobs
        if params.n:
            n = self.set_sampling_data("N", DType.TYPE_INT64, params.n)
            self.inputs["N"] = n
        if params.best_of is None and params.n:
            params.best_of = params.n
        elif params.best_of < params.n:
            raise RuntimeError("Sampling parameters: 'best_of' should not be less than 'n'")
        if params.best_of:
            best_of = self.set_sampling_data("BEST_OF", DType.TYPE_INT64, params.best_of)
            self.inputs["BEST_OF"] = best_of
        if params.use_beam_search:
            use_beam_search = self.set_sampling_data("USE_BEAM_SEARCH", DType.TYPE_BOOL, params.use_beam_search)
            self.inputs["USE_BEAM_SEARCH"] = use_beam_search
        return Status(Code.OK)

    def set_max_output_len(self, max_output_len):
        """
        设置请求最大输出长度
        :param max_output_len: 最大输出长度
        :return:
        """
        self.max_output_len = max_output_len

    def set_ignore_eos(self, ignore_eos: bool):
        """
        为request设置ignore_eos参数
        :param params: ignore_eos参数
        :return:
        """
        ignore_eos = self.set_sampling_data("IGNORE_EOS", DType.TYPE_BOOL, ignore_eos)
        self.inputs["IGNORE_EOS"] = ignore_eos
        return Status(Code.OK)

    def get_max_output_len(self):
        if self.max_output_len == -1:
            return Status((Code.ERROR), "max_output_len has not been initialized")
        return self.max_output_len

    def get_request_id(self) -> RequestId:
        """
        获取请求标识id
        :return:
        """
        return self.request_id

    def get_send_response_callback(self):
        return self.response_callback

    def set_send_response_callback(self, response_callback):
        self.response_callback = response_callback