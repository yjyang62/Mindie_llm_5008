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

import torch
import torch_npu

from operations_test import operation_test


class TestConcatOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Concat"
        self.concatdim = 1
        self.op_param = {"concatDim": self.concatdim}
        self.op_name = "ConcatOperation"
        self.row = random.randint(1, 1000)
        self.col = random.randint(1, 1000)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):
        golden_out_tensor = torch.cat((in_tensor['in0'], in_tensor['in1']), dim=self.concatdim)
        return [golden_out_tensor]

    def test_float16(self):
        in_tensor_0 = torch.rand(self.row, self.col, dtype=torch.float16).npu()
        in_tensor_1 = torch.rand(self.row, self.col, dtype=torch.float16).npu()
        out_tensor_0 = torch.zeros(
            self.row, 2 * self.col, dtype=torch.float16).npu()

        inputs = {'in0': in_tensor_0, 'in1': in_tensor_1}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()