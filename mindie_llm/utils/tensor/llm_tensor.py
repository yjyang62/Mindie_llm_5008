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

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def get_backend(self):
        pass

    @abstractmethod
    def get_op(self):
        pass

    @abstractmethod
    def get_npu(self):
        pass

    @abstractmethod
    def get_hal(self):
        pass

    @abstractmethod
    def ones(self, *args, **kwargs):
        pass

    @abstractmethod
    def equal(self, *args, **kwargs):
        pass

    @abstractmethod
    def repeat(self, value, size):
        pass

    @abstractmethod
    def softmax(self, *args):
        pass

    @abstractmethod
    def shape(self, value, dim=None):
        pass

    @abstractmethod
    def cumsum(self, *args):
        pass

    @abstractmethod
    def gather(self, input_params, index, dim):
        pass

    @abstractmethod
    def numpy(self, value):
        pass

    @abstractmethod
    def where(self, condition, input_param=None, other=None):
        """
        from mindie_llm.utils.tensor import tensor_backend, tensor
        import numpy as np
        a = tensor.tensor(np.arange(4).reshape((2, 2)), tensor.float32)
        b = tensor.tensor(np.ones((2, 2)), tensor.float32)
        output = tensor_backend.where(a < 3, a, b)
        print(output)
         [[0. 1.]
         [2. 1.]]

        output = tensor_backend.where(a < 3)
        print(output)
         [[0. 0. 1]
         [[0. 1. 0]]
        :param condition: (Tensor[bool]): If True, yield `input`, otherwise yield `other`.
        :param input_param: (Union[Tensor, Scalar]): When `condition` is True, values to select from.
        :param other: (Union[Tensor, Scalar]): When `condition` is False, values to select from.
        :return:
        """
        pass

    @abstractmethod
    def full(self, *args, **kwargs):
        pass

    @abstractmethod
    def tensor(self, *args, **kwargs):
        pass

    @abstractmethod
    def zeros(self, *args, **kwargs):
        pass

    @abstractmethod
    def fill_diagonal(self, mask, fill_value):
        pass

    @abstractmethod
    def to(self, value, device):
        pass

    @abstractmethod
    def get_device(self, value):
        pass

    @abstractmethod
    def cpu(self, value):
        pass

    @abstractmethod
    def scatter(self, input_params, axis, index, src):
        pass

    @abstractmethod
    def masked_fill(self, input_params, mask, value):
        pass

    @abstractmethod
    def pad(self, input_tensor, pad, mode, value):
        pass
