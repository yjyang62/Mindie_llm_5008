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
import atb_llm.nn as nn


class TestRopeFunction(unittest.TestCase):
    def test_rotary_coeff_2(self):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        def golden_rope(q, k, cos, sin):
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, 1, seq_len, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, 1, seq_len, dim]
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        q_emb, k_emb = nn.functional.rope(q=Tensor('q'), k=Tensor('k'), cos_table=Tensor('cos_t'),
            sin_table=Tensor('sin_t'), seqlen=Tensor('seq_len'))
        get_default_net().mark_output(q_emb, "q_emb")
        get_default_net().mark_output(k_emb, "k_emb")
        rope_engine = get_default_net().build_engine()

        q = torch.rand(512, 1024).half().npu()
        k = torch.rand(512, 1024).half().npu()
        cos_t = torch.rand(1024, 128).half().npu()
        sin_t = torch.rand(1024, 128).half().npu()
        seq_len = torch.ones(1, dtype=torch.int).npu() * 512

        q_emb = torch.empty(512, 1024).half().npu()
        k_emb = torch.empty(512, 1024).half().npu()

        inputs = {}
        inputs['q'] = q
        inputs['k'] = k
        inputs['cos_t'] = cos_t
        inputs['sin_t'] = sin_t
        inputs['seq_len'] = seq_len
        outputs = {"q_emb": q_emb, "k_emb": k_emb}
        rope_engine.forward(inputs, outputs)

        q_emb_golden, k_emb_golden = golden_rope(q.view(1, 512, 8, 128),
            k.view(1, 512, 8, 128), cos_t[:512, :], sin_t[:512, :])


        torch.npu.synchronize()
        self.assertTrue(torch.allclose(outputs["q_emb"], q_emb_golden.view(512, 1024), rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()