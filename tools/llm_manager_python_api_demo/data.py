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

from llm_manager_python_api_demo.dtype import DType, get_data_size_by_type, get_dtype_by_infer_datatype
from llm_manager_python_api_demo.counter import id_counter
import numpy as np
from llm_manager_python_api_demo import llm_manager_python


class Data:

    def __init__(self, data_id='0'):
        self.id = data_id
        self.name = ""
        self.type = DType.TYPE_INVALID
        self.shape = []
        self.data = []
        self.item_size = 0
        self.data_size = 0

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, item):
        return self.data[item]
    
    def set_token_id(self, data_type: DType, shape: np.ndarray, data: np.ndarray):
        """
        设置token_id
        :param data_type: 数据类型
        :param shape: 数据shape
        :param data: 具体数据，tokenId
        """
        self.name = "INPUT_IDS"
        self.type = data_type
        self.shape = shape
        self.data = data
        self.item_size = get_data_size_by_type(data_type)
        self.data_size = len(data)

    def set_token_num(self, data_type: DType, shape: np.ndarray, data: np.ndarray):
        self.name = "INPUT_TOKEN_NUM"
        self.type = data_type
        self.shape = shape
        self.data = data
        self.item_size = get_data_size_by_type(data_type)
        self.data_size = len(data)

    def set_sampling(self, name: str, data_type: DType, shape: np.ndarray, data: np.ndarray):
        """
        设置 sampling params
        :param data_type: 数据类型
        :param shape: 数据shape
        :param data: 具体数据，sampling
        """
        self.name = name
        self.type = data_type
        self.shape = shape
        self.data = data
        self.item_size = get_data_size_by_type(data_type)
        self.data_size = len(data)

    def get_id(self):
        return self.id
    
    def get_name(self):
        return self.name

    def get_type(self):
        return self.type

    def get_data(self):
        return self.data

    def get_shape(self):
        return self.shape
    
    def get_data_size(self):
        return self.data_size


def infer_tensor_to_data(tensor: llm_manager_python.InferTensor) -> Data:
    data = Data()
    data.name = tensor.get_name()
    data.type = get_dtype_by_infer_datatype(tensor.get_data_type())
    data.shape = tensor.get_shape()
    data.data = llm_manager_python.tensor_to_numpy(tensor)
    data.item_size = get_data_size_by_type(data.type)
    data.data_size = tensor.get_size()
    return data


def create_data(data_val, size) -> Data:
    new_id = id_counter.increment()
    id_str = str(new_id)
    shape = np.array([1, size], dtype=np.int64)
    data = Data(id_str)
    data.set_token_id(DType.TYPE_INT64, shape, np.array(data_val, dtype=np.int64))
    return data