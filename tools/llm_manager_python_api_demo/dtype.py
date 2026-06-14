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


from enum import Enum
import numpy as np
from llm_manager_python_api_demo import llm_manager_python


class DType(Enum):
    TYPE_INVALID = 0
    TYPE_BOOL = 1
    TYPE_UINT8 = 2
    TYPE_UINT16 = 3
    TYPE_UINT32 = 4
    TYPE_UINT64 = 5
    TYPE_INT8 = 6
    TYPE_INT16 = 7
    TYPE_INT32 = 8
    TYPE_INT64 = 9
    TYPE_FP16 = 10
    TYPE_FP32 = 11
    TYPE_FP64 = 12
    TYPE_STRING = 13
    TYPE_BF16 = 14
    TYPE_BUTT = 15


def get_dtype_by_value(n):
    for key, val in DType.__members__.items():
        if val.value == n:
            return key
    return DType.TYPE_INVALID


def get_data_size_by_type(in_type: DType):
    switcher = {
        DType.TYPE_INVALID: 0,
        DType.TYPE_BOOL: np.dtype(np.bool_).itemsize,
        DType.TYPE_UINT8: np.dtype(np.uint8).itemsize,
        DType.TYPE_UINT16: np.dtype(np.uint16).itemsize,
        DType.TYPE_UINT32: np.dtype(np.uint32).itemsize,
        DType.TYPE_UINT64: np.dtype(np.uint64).itemsize,
        DType.TYPE_INT8: np.dtype(np.int8).itemsize,
        DType.TYPE_INT16: np.dtype(np.int16).itemsize,
        DType.TYPE_INT32: np.dtype(np.int32).itemsize,
        DType.TYPE_INT64: np.dtype(np.int64).itemsize,
        DType.TYPE_FP16: 2,
        DType.TYPE_FP32: 4,
        DType.TYPE_FP64: 8,
        DType.TYPE_BF16: 2,
    }
    return switcher.get(in_type)


def get_numpy_dtype_by_type(in_type):
    switcher = {
        DType.TYPE_BOOL: np.bool_,
        DType.TYPE_UINT8: np.uint8,
        DType.TYPE_UINT16: np.uint16,
        DType.TYPE_UINT32: np.uint32,
        DType.TYPE_UINT64: np.uint64,
        DType.TYPE_INT8: np.int8,
        DType.TYPE_INT16: np.int16,
        DType.TYPE_INT32: np.int32,
        DType.TYPE_INT64: np.int64,
        DType.TYPE_FP16: np.float16,
        DType.TYPE_FP32: np.float32,
        DType.TYPE_FP64: np.float64,
    }
    return switcher.get(in_type, "invalid")


def get_infer_datatype_by_dtype(in_type: DType):
    switcher = {
        DType.TYPE_INVALID: llm_manager_python.InferDataType.TYPE_INVALID,
        DType.TYPE_BOOL: llm_manager_python.InferDataType.TYPE_BOOL,
        DType.TYPE_UINT8: llm_manager_python.InferDataType.TYPE_UINT8,
        DType.TYPE_UINT16: llm_manager_python.InferDataType.TYPE_UINT16,
        DType.TYPE_UINT32: llm_manager_python.InferDataType.TYPE_UINT32,
        DType.TYPE_UINT64: llm_manager_python.InferDataType.TYPE_UINT64,
        DType.TYPE_INT8: llm_manager_python.InferDataType.TYPE_INT8,
        DType.TYPE_INT16: llm_manager_python.InferDataType.TYPE_INT16,
        DType.TYPE_INT32: llm_manager_python.InferDataType.TYPE_INT32,
        DType.TYPE_INT64: llm_manager_python.InferDataType.TYPE_INT64,
        DType.TYPE_FP16: llm_manager_python.InferDataType.TYPE_FP16,
        DType.TYPE_FP32: llm_manager_python.InferDataType.TYPE_FP32,
        DType.TYPE_FP64: llm_manager_python.InferDataType.TYPE_FP64,
        DType.TYPE_STRING: llm_manager_python.InferDataType.TYPE_STRING,
        DType.TYPE_BF16: llm_manager_python.InferDataType.TYPE_BF16,
        DType.TYPE_BUTT: llm_manager_python.InferDataType.TYPE_BUTT
    }
    return switcher.get(in_type)


def get_dtype_by_infer_datatype(in_type: llm_manager_python.InferDataType):
    switcher = {
        llm_manager_python.InferDataType.TYPE_INVALID: DType.TYPE_INVALID,
        llm_manager_python.InferDataType.TYPE_BOOL: DType.TYPE_BOOL,
        llm_manager_python.InferDataType.TYPE_UINT8: DType.TYPE_UINT8,
        llm_manager_python.InferDataType.TYPE_UINT16: DType.TYPE_UINT16,
        llm_manager_python.InferDataType.TYPE_UINT32: DType.TYPE_UINT32,
        llm_manager_python.InferDataType.TYPE_UINT64: DType.TYPE_UINT64,
        llm_manager_python.InferDataType.TYPE_INT8: DType.TYPE_INT8,
        llm_manager_python.InferDataType.TYPE_INT16: DType.TYPE_INT16,
        llm_manager_python.InferDataType.TYPE_INT32: DType.TYPE_INT32,
        llm_manager_python.InferDataType.TYPE_INT64: DType.TYPE_INT64,
        llm_manager_python.InferDataType.TYPE_FP16: DType.TYPE_FP16,
        llm_manager_python.InferDataType.TYPE_FP32: DType.TYPE_FP32,
        llm_manager_python.InferDataType.TYPE_FP64: DType.TYPE_FP64,
        llm_manager_python.InferDataType.TYPE_STRING: DType.TYPE_STRING,
        llm_manager_python.InferDataType.TYPE_BF16: DType.TYPE_BF16,
        llm_manager_python.InferDataType.TYPE_BUTT: DType.TYPE_BUTT
    }
    return switcher.get(in_type)