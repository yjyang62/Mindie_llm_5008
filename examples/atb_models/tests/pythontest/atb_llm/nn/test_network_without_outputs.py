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
import numpy as np
import torch
import torch_npu

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
import atb_llm.nn as nn
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


class TestNegFunction(unittest.TestCase):
    def test(self):
        device = torch.ones(1).npu().device
        num_layers = 24
        for i in range(num_layers):
            nn.functional.copy_blocks(Tensor(f"layer_{i}_0"),
                                    Tensor(f"layer_{i}_1"),
                                    Tensor("in2"),
                                    Tensor("in3"),
                                    Tensor("in4"))
        engine = get_default_net().build_engine()
        block_copy_op = engine
        src_indices = np.ones(1, dtype=np.int32)
        dst_indices = np.ones(1, dtype=np.int32) * 2

        src_block_indices = torch.tensor(
            np.array(src_indices, dtype=np.int32)
            ).to(device=device, non_blocking=False)
        dst_block_indices = torch.tensor(
            np.array(dst_indices, dtype=np.int32)
            ).to(device=device, non_blocking=False)
        cum_sum = torch.tensor(
            np.arange(1, src_block_indices.size(0) + 1, dtype=np.int32)
            ).to(device=device, non_blocking=False)
        dtype = torch.bfloat16

        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")

        kv_cache = [
            [
                torch.ones([8, 2, 2, 2], dtype=dtype).to(device=device, non_blocking=False) * torch.arange(0, 8*2*2*2).view(8, 2, 2, 2).npu(),
                torch.ones([8, 2, 2, 2], dtype=dtype).to(device=device, non_blocking=False) * torch.arange(0, 8*2*2*2).view(8, 2, 2, 2).npu(),
            ] for _ in range(num_layers)
        ]

        mapTensors = {
            "in2": src_block_indices,
            "in3": dst_block_indices,
            "in4": cum_sum,
        }
        for i,x in enumerate(kv_cache):
            mapTensors[f"layer_{i}_0"] = x[0]
            mapTensors[f"layer_{i}_1"] = x[1]

        self.assertTrue(not torch.allclose(kv_cache[0][0][1], kv_cache[0][0][2]))
        for _ in range(10):
            block_copy_op.forward(mapTensors, {})

            torch.npu.synchronize()

        self.assertTrue(torch.allclose(kv_cache[0][0][1], kv_cache[0][0][2]))

if __name__ == '__main__':
    unittest.main()