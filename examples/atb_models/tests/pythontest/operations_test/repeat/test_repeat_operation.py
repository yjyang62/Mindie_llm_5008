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


class TestRepeatOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "Repeat"
        self.col = random.randint(1, 1000)
        self.multiples = random.randint(1, 10)
        self.op_param = {
            "multiples": [self.multiples, self.col],
        }
        self.op_name = "RepeatOperation"
        self.op_set = (self.op_type, self.op_param, self.op_name)


    def get_golden(self, in_tensor):
        golden_out_tensor = in_tensor['in0'].repeat(self.multiples, 1).npu()
        return [golden_out_tensor]


    def test_float16(self):

        in_tensor_0 = torch.rand(1, self.col, dtype=torch.float16).npu()
        out_tensor_0 = torch.zeros(self.multiples, self.col, dtype=torch.float16).npu()

        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()