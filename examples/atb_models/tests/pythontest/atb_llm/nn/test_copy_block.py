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
import numpy as np

import atb_llm.nn as nn
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


class TestCopyBlockFunction(unittest.TestCase):

    @staticmethod
    def process_mapping(mapping):
        src_block_indices = []
        dst_block_indices = []
        for pair in mapping:
            src, dst = pair.tolist()
            src_block_indices.append(src)
            dst_block_indices.append(dst)
        return src_block_indices, dst_block_indices

    def setUp(self):
        device = 'npu'
        num_block = 20
        block_size = 128
        num_head = 8
        head_size = 8
        dtype = torch.bfloat16

        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION and dtype == torch.bfloat16:
            self.skipTest("Atlas 300I DUO doesn't support bfloat16.")
        self.k_cache = torch.randn(num_block, block_size, num_head, head_size).to(dtype).to(device)
        self.v_cache = torch.randn(num_block, block_size, num_head, head_size).to(dtype).to(device)

        src_dst_map = np.array([[1, 2]], dtype=np.int64)
        src_idx, dst_idx = self.process_mapping(src_dst_map)

        self.src_block_indices = torch.tensor(np.array(src_idx, dtype=np.int32)).to(device=device, non_blocking=False)
        self.dst_block_indices = torch.tensor(np.array(dst_idx, dtype=np.int32)).to(device=device, non_blocking=False)
        self.cum_sum = torch.tensor(
            np.arange(1, self.src_block_indices.size(0) + 1, dtype=np.int32)
            ).to(device=device, non_blocking=False)

    def test_copy_blocks(self):
        nn.functional.copy_blocks(Tensor("in0"), 
                                  Tensor("in1"),
                                  Tensor("in2"),
                                  Tensor("in3"),
                                  Tensor("in4"))
        engine = get_default_net().build_engine()

        inputs = {
                    "in0": self.k_cache,
                    "in1": self.v_cache,
                    "in2": self.src_block_indices,
                    "in3": self.dst_block_indices,
                    "in4": self.cum_sum,
                }
        outputs = {}
        engine.forward(inputs, outputs)
        self.assertTrue(torch.equal(self.k_cache[1, ...], self.k_cache[2, ...]))
        self.assertTrue(torch.equal(self.v_cache[1, ...], self.v_cache[2, ...]))
    

if __name__ == '__main__':
    unittest.main()