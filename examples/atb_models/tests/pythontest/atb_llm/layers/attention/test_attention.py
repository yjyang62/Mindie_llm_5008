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
from unittest.mock import MagicMock

import torch
import torch_npu
from ddt import ddt, data, unpack

from atb_llm.utils.layers.attention.attention_mask import AttentionMask
from atb_llm.models.base.config import BaseConfig
from atb_llm.layers.attention.attention import Attention 
from atb_llm.nn.tensor import Tensor


INF = 'inf'
ATTENTION = "atb_llm.layers.attention"


@ddt
class TestAttentionMask(unittest.TestCase):
    def setUp(self):
        self.max_seq_len = 3
        self.dtype = torch.float16
        self.mini_type = torch.float16
        self.attention_mask = AttentionMask.static(self.max_seq_len, self.dtype, self.mini_type)
        self.attention_mask._seq_len_cached = self.max_seq_len
    
    @data((torch.float16, torch.float16, torch.finfo(torch.float16).min),
          (torch.float16, torch.float32, torch.finfo(torch.float32).min),
          (torch.float32, torch.float16, 1))
    @unpack
    def test_static(self, dtype, mini_type, mask_value):
        max_seq_len = 3
        attn_mask = AttentionMask.static(max_seq_len, dtype, mini_type)
        expected_atten_mask_cache = torch.tensor([[0., mask_value, mask_value], [0., 0., mask_value], [0., 0., 0.]])
        self.assertTrue(torch.allclose(attn_mask.atten_mask_cache, expected_atten_mask_cache))
        self.assertEqual(attn_mask._seq_len_cached, 0)
    
    def test_get_decode_attn_mask(self):
        max_s = 2
        input_lengths = torch.tensor([1, 2, 3], dtype=int).npu()
        expected_return = torch.zeros((3, 1, 2), dtype=bool).npu()
        expected_return[0, 0, 1] = True
        attn_mask = self.attention_mask.get_decode_attn_mask(input_lengths, max_s)
        self.assertTrue(torch.equal(attn_mask, expected_return))
    
    def test_update_attn_cache_given_large_seqlen(self):
        seqlen = 4
        self.attention_mask.update_attn_cache(self.dtype, 'cpu', seqlen)
        true_seq_len_cached = self.attention_mask._seq_len_cached
        self.assertEqual(true_seq_len_cached, seqlen)
        mask_value = torch.finfo(torch.float32).min
        expected_atten_mask_cache = torch.tensor(
            [[0., mask_value, mask_value, mask_value], [0., 0., mask_value, mask_value],
             [0., 0., 0., mask_value], [0., 0., 0., 0.]], dtype=torch.float16)
        self.assertTrue(torch.allclose(self.attention_mask.atten_mask_cache, expected_atten_mask_cache))
    
    def test_update_attn_cache_given_different_device(self):
        device = 'npu'
        self.attention_mask.update_attn_cache(self.dtype, device, self.max_seq_len)
        mask_value = torch.finfo(torch.float32).min
        expected_atten_mask_cache = torch.tensor([[0., mask_value, mask_value],
                                                  [0., 0., mask_value], [0., 0., 0.]], dtype=torch.float16)
        expected_atten_mask_cache = expected_atten_mask_cache.npu()
        self.assertTrue(torch.allclose(self.attention_mask.atten_mask_cache, expected_atten_mask_cache))

    def test_get_attn_mask_default_input(self):
        true_return = self.attention_mask.get_attn_mask(self.max_seq_len, self.dtype, "cpu")
        mask_value = torch.finfo(torch.float32).min
        expected_atten_mask_cache = torch.tensor(
            [[0., mask_value, mask_value], [0., 0., mask_value], [0., 0., 0.]], dtype=torch.float16)
        expected_return = expected_atten_mask_cache[:self.max_seq_len, :self.max_seq_len]
        self.assertTrue(torch.allclose(true_return, expected_return))
    
    def test_get_rope_prefill_mask(self):
        self.attention_mask.get_attn_mask = MagicMock()
        _ = self.attention_mask.get_rope_prefill_mask(self.max_seq_len, self.dtype, "cpu")
        self.attention_mask.get_attn_mask.assert_called_once_with(
            self.max_seq_len, self.dtype, "cpu", torch.float32)

    def test_get_rope_decode_mask_intial(self):
        self.attention_mask.get_attn_mask = MagicMock()
        _ = self.attention_mask.get_rope_decode_mask(self.dtype, "cpu")
        self.attention_mask.get_attn_mask.assert_called_once_with(
            1, dtype=self.dtype, device="cpu")
        self.assertIsNotNone(self.attention_mask._rope_decode_mask)

    def test_get_alibi_decode_mask(self):
        config = BaseConfig(num_attention_heads=8)
        setattr(config, "alibi_bias_max", 8.0)
        config_metadata = MagicMock()
        config_metadata.num_attention_heads = 2
        mask = self.attention_mask.get_alibi_decode_mask(
            5, [2, 0, 3], config, config_metadata, torch.float16, rank=0)
        golden_mask = torch.tensor([
        [[[-1.0000, -0.5000, 0.0, -float(INF), -float(INF)]],
         [[-0.5000, -0.2500, 0.0, -float(INF), -float(INF)]]],
        [[[0.0, -float(INF), -float(INF), -float(INF), -float(INF)]],
         [[0.0, -float(INF), -float(INF), -float(INF), -float(INF)]]],
        [[[-1.5, -1.0, -0.5, 0.0, -float(INF)]],
         [[-0.75, -0.5, -0.25, 0.0, -float(INF)]]]], dtype=torch.float16)
        self.assertTrue(torch.allclose(mask.cpu(), golden_mask, rtol=1e-04, atol=1e-04))


@ddt
class TestAttention(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()

        self.weight_tool = MagicMock()
        self.weight_tool.mapping = MagicMock()
        self.weight_tool.mapping.rank = 0
        self.weight_tool.mapping.attn_tp = MagicMock()
        self.weight_tool.mapping.attn_tp.group_size = 1
        self.weight_tool.mapping.attn_tp.process_group = '-1'

        self.config_metadata = MagicMock()
        self.config_metadata.head_dim = 128
        self.config_metadata.num_attention_heads = 8
        self.config_metadata.num_key_value_heads = 2

        self.self_attn = Attention(self.config, self.weight_tool, "test", self.config_metadata)
        self.self_attn.q_norm = MagicMock(return_value=Tensor("test_q_norm"))
        self.self_attn.k_norm = MagicMock(return_value=Tensor("test_k_norm"))
        self.self_attn.dense = MagicMock(return_value=Tensor("testd"))

    @data(([Tensor("test1"), Tensor("test2"), Tensor("test3")], True),([Tensor("test1")], True),
          ([Tensor("test1")], False))
    @unpack
    def test_forward_qkv_pa(self, qkv, is_prefill):
        self.self_attn.qkv = MagicMock(return_value=qkv)
        self.self_attn.qkv.__len__.return_value = len(qkv)
        out = self.self_attn(
            Tensor("inputs"), Tensor("cos_table"), Tensor("sin_table"),
            Tensor("k_cache"), Tensor("v_cache"), Tensor("slots"), Tensor("mask"), Tensor("seq_lens"),
            Tensor("block_table"), is_prefill=is_prefill)
        self.assertIsInstance(out, Tensor)

        out = self.self_attn(
            Tensor("inputs"), Tensor("cos_table"), Tensor("sin_table"),
            Tensor("k_cache"), Tensor("v_cache"), Tensor("slots"), Tensor("mask"), Tensor("seq_lens"),
            Tensor("block_table"), is_prefill=False)
        self.assertIsInstance(out, Tensor)

        self.self_attn.qkv = MagicMock(return_value=[Tensor("test1")])
        self.self_attn.qkv.__len__.return_value = 1
        out = self.self_attn(
            Tensor("inputs"), Tensor("cos_table"), Tensor("sin_table"),
            Tensor("k_cache"), Tensor("v_cache"), Tensor("slots"), Tensor("mask"), Tensor("seq_lens"),
            Tensor("block_table"))
        self.assertIsInstance(out, Tensor)


if __name__ == '__main__':
    unittest.main()