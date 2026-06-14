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

from atb_llm.nn.functional import multi_latent_attention
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional.attention.multi_latent_attention import CacheMode, CalcType, MaskType
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


class TestMultiLatentAttentionFunction(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.head_num = 8
        self.kv_head_num = 1
        self.block_size = 128
        self.block_num = 8
        self.qk_scale = 1.0
        self.max_context_len = 128
        self.max_block_nums_per_query = (self.max_context_len + self.block_size - 1) // self.block_size
    

    def test_multi_latent_attention(self):
        def golden(q_nope, q_rope, k_cache, v_cache, **kwargs):
            block_tables = kwargs.get("block_tables")
            context_len = kwargs.get("context_len")
            context = []
            for i, context_len in enumerate(context_len):
                block_table = block_tables[i]
                block_num = context_len // self.block_size
                context_len = context_len % self.block_size
                blocks = block_table[:block_num]

                cur_q_nope = q_nope[i:i + 1]
                cur_q_rope = q_rope[i:i + 1]
                cur_q = torch.cat((cur_q_nope, cur_q_rope), dim=-1) 
                cur_k = k_cache[blocks]
                cur_v = v_cache[blocks]

                cur_k = cur_k.view(-1, 1, 512)
                cur_v = cur_v.view(-1, 1, 64)

                if context_len != 0:
                    cur_k = torch.concat((cur_k, k_cache[block_table[block_num], :context_len]), dim=0)
                    cur_v = torch.concat((cur_v, v_cache[block_table[block_num], :context_len]), dim=0)

                cur_q = cur_q.permute(1, 0, 2)
                act_v = cur_k
                act_k = torch.cat((cur_k, cur_v), dim=-1) 
                act_k = act_k.permute(1, 2, 0)

                cur_qk = torch.bmm(cur_q, act_k) * self.qk_scale
                cur_qk = torch.nn.functional.softmax(cur_qk, dim=-1)

                act_v = act_v.permute(1, 0, 2)
                cur_context = torch.bmm(cur_qk, act_v)
                cur_context = cur_context.permute(1, 0, 2).contiguous()

                context.append(cur_context)
            context = torch.concat(context, dim=0).type(torch.float16)
            return context

        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("This operation doesn't support Atlas 300I DUO.")

        q_nope = Tensor("q_nope")
        q_rope = Tensor("q_rope")
        k_cache = Tensor("k_cache")
        v_cache = Tensor("v_cache")
        block_tables = Tensor("block_tables")
        seq_lens = Tensor("seq_lens")
        args = {
            "q_nope": q_nope,
            "q_rope": q_rope,
            "ct_kv": k_cache,
            "k_rope": v_cache, 
            "block_tables": block_tables,
            "context_lens": seq_lens,
            "head_num": self.head_num,
            "qk_scale": self.qk_scale,
            "kv_head_num": self.kv_head_num,
            "mask_type": MaskType.UNDEFINED,
            "calc_type": CalcType.CALC_TYPE_UNDEFINED,
            "cache_mode": CacheMode.KROPE_CTKV
        }
        attn_out = multi_latent_attention(**args)
        get_default_net().mark_output(attn_out, "attn_out")
        mla_engine = get_default_net().build_engine()

        seq_lens = torch.ones((self.batch_size,), dtype=torch.int32)
        num_tokens = int(seq_lens.sum())
        q_nope = torch.rand(num_tokens, self.head_num, 512, dtype=torch.float16).npu()
        q_rope = torch.rand(num_tokens, self.head_num, 64, dtype=torch.float16).npu()
        k_cache = torch.rand(self.block_num, self.block_size, 1, 512, dtype=torch.float16).npu()
        v_cache = torch.rand(self.block_num, self.block_size, 1, 64, dtype=torch.float16).npu()
        block_tables = torch.randint(0, self.block_num, size=(self.batch_size, self.max_block_nums_per_query), dtype=torch.int32).npu()
        attn_out = torch.empty(num_tokens, self.head_num, 512).half().npu()
        inputs = {"q_nope": q_nope, "q_rope": q_rope, "k_cache": k_cache, "v_cache": v_cache, 
                  "block_tables": block_tables, "seq_lens": seq_lens.npu()}
        outputs = {"attn_out": attn_out}
        bind = {"seq_lens": seq_lens}
        mla_engine.forward(inputs, outputs, bind)
        attn_out_golden = golden(q_nope, q_rope, k_cache, v_cache, block_tables=block_tables, context_len=seq_lens)
        self.assertTrue(torch.allclose(attn_out_golden, attn_out, rtol=1e-05, atol=1e-05))


if __name__ == '__main__':
    unittest.main()