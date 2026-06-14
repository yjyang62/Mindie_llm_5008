# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
import logging

import torch
import torch.nn as nn
import torch_npu

from operations_test import operation_test


class TestGeluOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Activation"
        self.activation_type_gelu = 'ACTIVATION_GELU'
        self.op_param_gelu = {'activationType': self.activation_type_gelu}
        self.op_name = "Activation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.op_set = (self.op_type, self.op_param_gelu, self.op_name)

    def get_golden(self, in_tensors):
        m = nn.GELU()
        golden_out_tensor = m(in_tensors['in0'])
        return [golden_out_tensor]

    def test_gelu(self):
        in_tensor_0 = torch.rand(self.row, self.col, dtype=torch.float16).npu()
        out_tensor_0 = torch.zeros(self.row, self.col, dtype=torch.float16).npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


class TestSwishOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Activation"
        self.activation_type_swish = 'ACTIVATION_SWISH'
        self.op_param_swish = {'activationType': self.activation_type_swish}
        self.op_name = "Activation"
        self.row = random.randint(1, 100)
        self.col = random.randint(1, 100)
        self.op_set = (self.op_type, self.op_param_swish, self.op_name)

    def get_golden(self, in_tensors):
        golden_out_tensor = in_tensors['in0'] * nn.functional.sigmoid(in_tensors['in0'])
        return [golden_out_tensor]

    def test_swish(self):
        logger = logging.getLogger()

        in_tensor_0 = torch.rand(self.row, self.col, dtype=torch.float16).npu()
        out_tensor_0 = torch.zeros(self.row, self.col, dtype=torch.float16).npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()