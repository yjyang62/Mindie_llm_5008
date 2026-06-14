# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
from ddt import ddt, data
import torch
import torch.nn as nn
import torch_npu

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional import paged_attention
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION


SEQ_LENS = "seq_lens"       # tensor name of seq_lens
CONTEXT_LEN = "context_len" # tensor name of context_len
NPU = 'npu'                     # npu device type


@ddt
class TestAttentionFunction(unittest.TestCase):
    def setUp(self):
        self.batch = random.randint(2, 8)
        self.prefill_len = 100
        self.head_num = 8
        self.head_size = 128
        self.block_size = 128
        self.block_num = 64
        self.qk_scale = 1 / self.head_size ** 0.5
        self.max_context_len = 1024
        self.max_block_nums_per_query = (self.max_context_len + self.block_size - 1) // self.block_size

    @data(torch.float16, torch.bfloat16)
    def test_pa_prefill_mask_zero(self, dtype):
        def golden(q, k, v):
            seqlen = q.shape[0]
            q = q.view(1, q.shape[0], self.head_num, self.head_size).permute(0, 2, 1, 3) * self.qk_scale
            k = k.view(1, k.shape[0], self.head_num, self.head_size).permute(0, 2, 1, 3)
            v = v.view(1, v.shape[0], self.head_num, self.head_size).permute(0, 2, 1, 3)
            # Tensor shape: [b, hn, s, s]
            attn_weights = torch.matmul(q, k.transpose(2, 3))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
            # Tensor shape: [b, hn, s, hd]
            attn_output = torch.matmul(attn_weights, v)
            # Tensor shape: [b*s, h]
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(seqlen, self.head_num, self.head_size)
            return attn_output
        q = Tensor('q')
        k = Tensor('k')
        v = Tensor('v')
        mask = None
        kv_lens = Tensor('seq_lens')
        atten_score = paged_attention(q, k, v, mask=mask, head_num=self.head_num,
                                      qk_scale=self.qk_scale, kv_lens=kv_lens)
        get_default_net().mark_output(atten_score, "atten_score")
        attention_engine = get_default_net().build_engine()

        q = torch.rand(self.prefill_len, self.head_num, self.head_size, dtype=dtype).to(NPU)
        k = torch.rand(self.prefill_len, self.head_num, self.head_size, dtype=dtype).to(NPU)
        v = torch.rand(self.prefill_len, self.head_num, self.head_size, dtype=dtype).to(NPU)
        seq_lens = torch.ones(1, dtype=torch.int32) * self.prefill_len
        atten_score = torch.empty(self.prefill_len, self.head_num, self.head_size, dtype=dtype).to(NPU)

        inputs = {}
        inputs["q"] = q
        inputs["k"] = k
        inputs["v"] = v
        inputs[SEQ_LENS] = seq_lens.to(NPU)
        outputs = {"atten_score": atten_score}
        attention_engine.forward(inputs, outputs, {SEQ_LENS: seq_lens})

        golden_score = golden(q, k, v)

        torch.npu.synchronize()
        self.assertTrue(torch.allclose(atten_score, golden_score, rtol=1e-02, atol=1e-02))
    
    def test_pa_decoder(self):
        def golden(q, k_cache, v_cache, block_tables, context_len):
            q = q.type(torch.float32)
            k_cache = k_cache.type(torch.float32)
            v_cache = v_cache.type(torch.float32)

            context = []
            for i, context_len in enumerate(context_len):
                block_table = block_tables[i]
                block_num = context_len // self.block_size
                context_len = context_len % self.block_size
                blocks = block_table[:block_num]

                cur_q = q[i:i + 1]    # [q_len, head_num, head_size]
                cur_k = k_cache[blocks]  # [kv_len, head_num, head_size]
                cur_v = v_cache[blocks]  # [kv_len, head_num, head_size_v]

                cur_k = cur_k.view(-1, self.head_num, self.head_size)
                cur_v = cur_v.view(-1, self.head_num, self.head_size)

                if context_len != 0:
                    cur_k = torch.concat((cur_k, k_cache[block_table[block_num], :context_len]), dim=0)
                    cur_v = torch.concat((cur_v, v_cache[block_table[block_num], :context_len]), dim=0)

                cur_q = cur_q.permute(1, 0, 2)  # [head_num, q_len, head_size]
                cur_k = cur_k.permute(1, 2, 0)  # [head_num, head_size, kv_len]
                cur_qk = torch.bmm(cur_q, cur_k) * self.qk_scale  # [head_num, q_len, kv_len]
                cur_qk = torch.nn.functional.softmax(cur_qk, dim=-1)

                cur_v = cur_v.permute(1, 0, 2)  # [head_num, kv_len, head_size_v]
                cur_context = torch.bmm(cur_qk, cur_v)  # [head_num, q_len, head_size_v]
                cur_context = cur_context.permute(1, 0, 2).contiguous() # [q_len, head_num, head_size_v]

                context.append(cur_context)
            context = torch.concat(context, dim=0).type(torch.float16)
            return context

        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("Currently, this UT doesn't support Atlas 300I DUO.")

        seq_lens = torch.ones((self.batch,), dtype=torch.int32)
        ntokens = int(seq_lens.sum())
        q = Tensor('q')
        k_cache = Tensor('k_cache')
        v_cache = Tensor('v_cache')
        qk_scale = self.qk_scale
        block_table = Tensor('block_table')
        context_len = Tensor('context_len')
        atten_score = paged_attention(q, k_cache=k_cache, v_cache=v_cache, head_num=self.head_num, qk_scale=qk_scale,
                                      block_table=block_table, kv_lens=context_len, high_precision=True)
        get_default_net().mark_output(atten_score, "atten_score")
        attention_engine = get_default_net().build_engine()

        q = torch.rand(ntokens, self.head_num, self.head_size, dtype=torch.float16)
        k_cache = torch.rand(self.block_num, self.block_size, self.head_num, self.head_size, dtype=torch.float16)
        v_cache = torch.rand(self.block_num, self.block_size, self.head_num, self.head_size, dtype=torch.float16)
        block_table = torch.randint(0, self.block_num, 
                                    size=(ntokens, self.max_block_nums_per_query), dtype=torch.int32)
        context_len = torch.randint(1, self.max_context_len, size=(ntokens, ), dtype=torch.int32)

        atten_score = torch.empty(ntokens, self.head_num, self.head_size).half().to(NPU)
        
        golden_score = golden(q, k_cache, v_cache, block_table, context_len)

        inputs = {}
        inputs["q"] = q.to(NPU)
        k_cache = k_cache.to(NPU)
        v_cache = v_cache.to(NPU)
        if torch_npu._C._npu_get_soc_version() in NO_BF16_SUPPORT_SOC_VERSION:
            k_cache = k_cache.reshape(self.block_num, self.block_size, self.head_num * self.head_size // 16, 16).permute(0, 2, 1, 3).contiguous()
            torch_npu.npu_format_cast_(k_cache, 29)
            v_cache = v_cache.reshape(self.block_num, self.block_size, self.head_num * self.head_size // 16, 16).permute(0, 2, 1, 3).contiguous()
            torch_npu.npu_format_cast_(v_cache, 29)
        inputs["k_cache"] = k_cache
        inputs["v_cache"] = v_cache
        inputs["block_table"] = block_table.to(NPU)
        inputs[CONTEXT_LEN] = context_len.to(NPU)
        outputs = {"atten_score": atten_score}
        bind = {CONTEXT_LEN: context_len}

        attention_engine.forward(inputs, outputs, bind)

        self.assertTrue(torch.allclose(atten_score, golden_score.to(NPU), rtol=1e-02, atol=1e-02))


if __name__ == '__main__':
    unittest.main()