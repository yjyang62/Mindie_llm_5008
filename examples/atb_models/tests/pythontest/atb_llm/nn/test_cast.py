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
from ddt import ddt, data, unpack

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.utils.quantize.pack_type import DataType
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


@ddt
class TestCastFunction(unittest.TestCase):
    @data((DataType.ACL_FLOAT, torch.float16, torch.float), (DataType.ACL_FLOAT16, torch.float, torch.float16),
          (DataType.ACL_FLOAT, torch.bfloat16, torch.float), (DataType.ACL_BF16, torch.float, torch.bfloat16),
          (DataType.ACL_INT64, torch.int32, torch.int64), (DataType.ACL_INT32, torch.int64, torch.int32),
          (DataType.ACL_FLOAT16, torch.int8, torch.float16), (DataType.ACL_BF16, torch.int8, torch.bfloat16))
    @unpack
    def test_cast(self, acl_dtype, in_dtype, out_dtype):
        if (torch.bfloat16 in [acl_dtype, in_dtype, out_dtype]) and (torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION):
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")

        def golden(x, dtype):
            return x.to(dtype)

        cast = Tensor("x").to(acl_dtype)
        get_default_net().mark_output(cast, "x_cast")
        cast_engine = get_default_net().build_engine()

        shape = (1024, 1024)
        x = torch.rand(shape).to(in_dtype).npu()
        x_cast = torch.empty(shape).to(out_dtype).npu()

        inputs = {"x": x}
        outputs = {"x_cast": x_cast}
        cast_engine.forward(inputs, outputs)

        golden_out = golden(x, out_dtype)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(x_cast, golden_out, rtol=5e-03, atol=5e-03))


if __name__ == '__main__':
    unittest.main()