# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest

import torch
import torch_npu

from operations_test import operation_test


class TestInplaceNanToNumOperation(operation_test.OperationTest):
    def setUp(self):
        self.op_type = "InplaceNanToNum"
        self.nan_value = 1.0
        self.pos_inf_value = 1000.0
        self.neg_inf_value = -1000.0
        self.op_param = {"nanValue": self.nan_value,
                         "posInfValue": self.pos_inf_value,
                         "negInfValue": self.neg_inf_value}
        self.op_name = "InplaceNanToNumOperation"
        self.shape = (4, 10)
        self.op_set = (self.op_type, self.op_param, self.op_name)

    def get_golden(self, in_tensor):
        golden_out_tensor = in_tensor['in0']
        golden_out_tensor[torch.isnan(golden_out_tensor)] = self.nan_value
        golden_out_tensor[torch.isinf(golden_out_tensor) & (golden_out_tensor > 0)] = self.pos_inf_value
        golden_out_tensor[torch.isinf(golden_out_tensor) & (golden_out_tensor < 0)] = self.neg_inf_value
        return [golden_out_tensor]

    def test_float16(self):
        torch_npu.npu.set_device(0)
        in_tensor_0 = torch.randn(4, 10, dtype=torch.float16).npu()
        num_nan = 3
        num_inf = 3
        nan_indices = torch.randint(0, in_tensor_0.numel(), (num_nan,))
        in_tensor_0.view(-1)[nan_indices] = float('nan')
        inf_indices = torch.randint(0, in_tensor_0.numel(), (num_inf,))
        in_tensor_0.view(-1)[inf_indices] = float('inf')
        neginf_indices = torch.randint(0, in_tensor_0.numel(), (num_inf,))
        in_tensor_0.view(-1)[neginf_indices] = float('-inf')


        inputs = {'in0': in_tensor_0}
        outputs = {'out0': in_tensor_0}

        self.run_compare(self.op_set, inputs, outputs)


if __name__ == '__main__':
    unittest.main()
