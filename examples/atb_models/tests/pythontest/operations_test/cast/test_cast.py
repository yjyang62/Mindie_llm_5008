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


ACL_INT8 = "ACL_INT8"
ACL_FLOAT16 = "ACL_FLOAT16"
ACL_FLOAT = "ACL_FLOAT"
ACL_INT32 = "ACL_INT32"


@ddt
class TestCastOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Cast"
        self.op_name = "CastOperation"
        self.op_param = {}
        self.m = random.randint(10, 20)
        self.n = random.randint(150000, 200000)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, inputs):
        in_tensor = inputs['in0']
        golden_out_tensor = in_tensor.to(self.output_dtype)
        return [golden_out_tensor]

    @data(
        (ACL_INT8, torch.float16, torch.int8),
        (ACL_FLOAT16, torch.int8, torch.float16),
        (ACL_FLOAT16, torch.float32, torch.float16),
        (ACL_FLOAT, torch.float16, torch.float32),
        (ACL_INT32, torch.float32, torch.int32)
    )
    @unpack
    def test_cast_operation(self, dtype_str, input_dtype, output_dtype):
        self.op_param = {"dtype": dtype_str}
        self.op_set = (self.op_type, self.op_param, self.op_name)
        self.output_dtype = output_dtype

        if input_dtype == torch.int8:
            in_tensor = torch.randint(-128, 127, [self.m, self.n], dtype=input_dtype).npu()
        else:
            in_tensor = torch.randn([self.m, self.n], dtype=input_dtype).npu()

        out_tensor_0 = torch.zeros([self.m, self.n], dtype=output_dtype).npu()

        inputs = {'in0': in_tensor}
        outputs = {'out0': out_tensor_0}
        self.out_tensor = outputs["out0"]

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == "__main__":
    unittest.main()