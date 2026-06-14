# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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


class TestSliceOperation(operation_test.OperationTest):
    def setUp(self):       
        self.op_type = "Slice"
        self.offsets0 = random.randint(1, 100)
        self.offsets1 = random.randint(1, 100)
        self.offsets = [self.offsets0, self.offsets1]
        self.size0 = random.randint(1, 100)
        self.size1 = random.randint(1, 100)
        self.size = [self.size0, self.size1]
        self.op_param = {"offsets": self.offsets, "size": self.size}
        self.op_name = "SliceOperation"
        self.row = random.randint(200, 300)
        self.col = random.randint(200, 300)    
        self.op_set = (self.op_type, self.op_param, self.op_name)
    
    def get_golden(self, in_tensors):
        return [in_tensors['in0'][self.offsets0: self.offsets0 + self.size0,
            self.offsets1: self.offsets1 + self.size1]]

    def test_2d_float(self):
        in_tensor_0 = torch.rand(self.row, self.col, dtype=torch.float16).npu()
        out_tensor_0 = torch.zeros(self.size0, self.size1, dtype=torch.float16).npu()
        
        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}
        
        self.run_compare(self.op_set, inputs, outputs)

    def test_2d_bfloat16(self):
        in_tensor_0 = torch.rand(self.row, self.col, dtype=torch.bfloat16).npu()
        out_tensor_0 = torch.zeros(self.size0, self.size1, dtype=torch.bfloat16).npu()
        
        inputs = {'in0': in_tensor_0}
        outputs = {'out0': out_tensor_0}
        
        self.run_compare(self.op_set, inputs, outputs)

if __name__ == '__main__':
    unittest.main()
    