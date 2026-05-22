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

import atb_llm.nn.functional as F
from ..linear import TransposeType
from ..module import Module
from ...tensor import Tensor
from ...parameter import Parameter
from ...network_manager import get_default_net


BIAS = "bias"


class Linear(Module):
    """A linear transformation module that inherits from Module."""
    def __init__(self, prefix: str, bias=False):
        """
        Initialize a linear module.

        Parameters:
            prefix (str): The prefix for parameter names.
            bias (bool, optional): Whether to include a bias term. Defaults to False.
        """
        super().__init__()
        weight_parameter = Parameter(
            prefix=prefix, suffix="weight", enable_auto_transpose=True, enable_nd_nz=True)
        self.register_parameter("weight", weight_parameter)
        if bias:
            bias_parameter = Parameter(prefix=prefix, suffix=BIAS)
            self.register_parameter(BIAS, bias_parameter)
        else:
            self.register_parameter(BIAS, None)

    def __call__(
            self,
            input_: Tensor
        ) -> Tensor:
        """
        Overload the call method to enable instance invocation like a function.

        Parameters:
            input_ (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after linear transformation.
        """
        return self._forward(input_)

    def _forward(
            self,
            input_: Tensor,
        ) -> Tensor:
        """
        Forward propagation method to perform linear transformation.

        Parameters:
            input_ (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after linear transformation.
        """
        args = {
            "input_": input_,
            "weight": self.weight.get_tensor(),
            "bias": self.bias.get_tensor() if self.bias is not None else None,
            "transpose_b": self.weight.trans_flag == TransposeType.TRANSPOSE,
        }
        out = F.linear(**args)
        return out
