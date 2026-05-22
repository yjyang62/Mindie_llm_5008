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

import random
import unittest

import torch
import torch_npu
from ddt import ddt, data, unpack
from operations_test import operation_test


@ddt
class TestScatterOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Scatter"
        self.op_name = "ScatterOperation"
        self.dim = 1
        self.m = random.randint(15, 20)
        self.n = random.randint(150000, 200000)
        self.k = random.randint(10, 20)
        self.op_param = {}
        self.op_set = None

    def get_golden(self, inputs):
        self_tensor = inputs['in0']
        index = inputs['in1']
        src = inputs['in2']
        golden_out = self_tensor.clone()
        reduce_op = self.op_param.get("reduce")
        if reduce_op == 0:  # replace
            golden_out = torch.scatter(self_tensor.clone(), self.dim, index, src).npu()
        elif reduce_op == 1:  # add
            golden_out = torch.scatter_reduce(self_tensor.clone(), self.dim, index, src, reduce="sum").npu()
        elif reduce_op == 2:  # multiply
            golden_out = torch.scatter_reduce(self_tensor.clone(), self.dim, index, src, reduce="prod").npu()
        return [golden_out]

    @data(
        (torch.float16, 0, "scatter_replace"),
        (torch.float16, 1, "scatter_add"),
    )
    @unpack
    def test_scatter_float16_2d(self, dtype, reduce_op, scatter_identifier):
        self.op_param = {
            "dim": self.dim,
            "reduce": reduce_op,
        }
        self.op_set = (self.op_type, self.op_param, self.op_name)

        # self, index, src
        self_tensor = torch.randn([self.m, self.n], dtype=dtype).npu()
        index = torch.randint(0, self.n, [self.m, random.randint(1, self.n)], dtype=torch.int64).npu()
        src = torch.randn([self.m, index.size(1)], dtype=dtype).npu()

        # npu
        scatter_output = torch.zeros([self.m, self.n], dtype=dtype).npu()

        inputs = {'in0': self_tensor, 'in1': index, 'in2': src}
        outputs = {'out0': scatter_output}
        self.out_tensor = outputs["out0"]

        self.run_compare(self.op_set, inputs, outputs)

    @data(
        (torch.float32, 0, "scatter_replace"),
        (torch.float32, 1, "scatter_add"),
        (torch.float32, 2, "scatter_multiply"),
    )
    @unpack
    def test_scatter_float32_2d(self, dtype, reduce_op, scatter_identifier):
        self.op_param = {
            "dim": self.dim,
            "reduce": reduce_op,
        }
        self.op_set = (self.op_type, self.op_param, self.op_name)

        # self, index, src
        self_tensor = torch.randn([self.m, self.n], dtype=dtype).npu()
        index = torch.randint(0, self.n, [self.m, random.randint(1, self.n)], dtype=torch.int64).npu()
        src = torch.randn([self.m, index.size(1)], dtype=dtype).npu()

        # npu
        scatter_output = torch.zeros([self.m, self.n], dtype=dtype).npu()

        inputs = {'in0': self_tensor, 'in1': index, 'in2': src}
        outputs = {'out0': scatter_output}
        self.out_tensor = outputs["out0"]

        self.run_compare(self.op_set, inputs, outputs)

    @data(
        (torch.float16, 0, "scatter_replace"),
        (torch.float16, 1, "scatter_add"),
    )
    @unpack
    def test_scatter_float16_3d(self, dtype, reduce_op, scatter_identifier):
        self.op_param = {
            "dim": self.dim,
            "reduce": reduce_op,
        }
        self.op_set = (self.op_type, self.op_param, self.op_name)

        # self, index, src
        self_tensor = torch.randn([self.m, self.n, self.k], dtype=dtype).npu()
        index = torch.randint(0, self.n, [self.m, random.randint(1, self.n), self.k], dtype=torch.int64).npu()
        src = torch.randn([self.m, index.size(1), self.k], dtype=dtype).npu()

        # npu
        scatter_output = torch.zeros([self.m, self.n, self.k], dtype=dtype).npu()

        inputs = {'in0': self_tensor, 'in1': index, 'in2': src}
        outputs = {'out0': scatter_output}
        self.out_tensor = outputs["out0"]

        self.run_compare(self.op_set, inputs, outputs)

    @data(
        (torch.float32, 0, "scatter_replace"),
        (torch.float32, 1, "scatter_add"),
        (torch.float32, 2, "scatter_multiply"),
    )
    @unpack
    def test_scatter_float32_3d(self, dtype, reduce_op, scatter_identifier):
        self.op_param = {
            "dim": self.dim,
            "reduce": reduce_op,
        }
        self.op_set = (self.op_type, self.op_param, self.op_name)

        # self, index, src
        self_tensor = torch.randn([self.m, self.n, self.k], dtype=dtype).npu()
        index = torch.randint(0, self.n, [self.m, random.randint(1, self.n), self.k], dtype=torch.int64).npu()
        src = torch.randn([self.m, index.size(1), self.k], dtype=dtype).npu()

        # npu
        scatter_output = torch.zeros([self.m, self.n, self.k], dtype=dtype).npu()

        inputs = {'in0': self_tensor, 'in1': index, 'in2': src}
        outputs = {'out0': scatter_output}
        self.out_tensor = outputs["out0"]

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == "__main__":
    unittest.main()