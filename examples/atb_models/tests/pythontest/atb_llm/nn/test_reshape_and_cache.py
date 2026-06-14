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

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional import reshape_and_cache
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


class TestReshapeAndCacheFunction(unittest.TestCase):
    def test_slot_99(self):
        reshape_and_cache(k=Tensor("k"), v=Tensor("v"),
                          k_cache=Tensor("k_cache"), v_cache=Tensor("v_cache"), slot_mapping=Tensor("slot_mapping"))
        reshape_and_cache_engine = get_default_net().build_engine()

        k = torch.rand(100, 8, 128).half().npu()
        v = torch.rand(100, 8, 128).half().npu()
        k_cache = torch.rand(100, 128, 8, 128).half().npu()
        v_cache = torch.rand(100, 128, 8, 128).half().npu()
        slot_mapping = torch.arange(100, dtype=torch.int32).npu()

        inputs = {}
        inputs["k"] = k
        inputs["v"] = v
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            k_cache = k_cache.reshape(100, 128, 8 * 128 // 16, 16).permute(0, 2, 1, 3).contiguous()
            torch_npu.npu_format_cast_(k_cache, 29)
            v_cache = v_cache.reshape(100, 128, 8 * 128 // 16, 16).permute(0, 2, 1, 3).contiguous()
            torch_npu.npu_format_cast_(v_cache, 29)
        inputs["k_cache"] = k_cache
        inputs["v_cache"] = v_cache
        inputs["slot_mapping"] = slot_mapping
        outputs = {}
        reshape_and_cache_engine.forward(inputs, outputs)

        torch.npu.synchronize()
        
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.assertTrue(torch.allclose(k, k_cache.permute(0, 2, 1, 3).reshape(12800, 8, 128)[:100, :, :], rtol=1e-02, atol=1e-02))
            self.assertTrue(torch.allclose(v, v_cache.permute(0, 2, 1, 3).reshape(12800, 8, 128)[:100, :, :], rtol=1e-02, atol=1e-02))
        else:
            self.assertTrue(torch.allclose(k, k_cache.view(12800, 8, 128)[:100, :, :], rtol=1e-02, atol=1e-02))
            self.assertTrue(torch.allclose(v, v_cache.view(12800, 8, 128)[:100, :, :], rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()