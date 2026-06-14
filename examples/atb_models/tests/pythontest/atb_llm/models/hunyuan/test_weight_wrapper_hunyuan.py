#!/usr/bin/env python
# coding=utf-8
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
from unittest import TestCase
import torch

from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.models.hunyuan.weight_wrapper_hunyuan import HunyuanWeightWrapper, ClaWrapper


class TestHuanyuanWeightWrapper(TestCase):
    def setUp(self):
        mock_linear, mock_layer = MagicMock(), MagicMock()
        mock_layer.mlp.pack_type = PackType.ALL_W8A8_DYNAMIC
        mock_linear.linear.weight.data = torch.empty(100, 100, dtype=torch.int8)
        mock_linear.linear.weight.dtype = torch.int8
        mock_linear.linear.bias = None
        mock_linear.linear.quant_bias.data = torch.empty(100, 100, dtype=torch.int32)
        mock_linear.linear.deq_scale.data = torch.empty(100, 100, dtype=torch.float32)
        mock_linear.linear.input_offset.data = torch.empty(100, 100, dtype=torch.bfloat16)
        mock_linear.linear.input_scale.data = torch.empty(100, 100, dtype=torch.bfloat16)
        mock_linear.linear.index.data = 1
        mock_layer.attn.pack_type = PackType.ALL_W8A8_DYNAMIC
        mock_layer.attn.qkv = mock_linear
        mock_layer.mlp.gate_up_proj = mock_linear
        mock_layer.mlp.down_proj = mock_linear
        self.mock_layer = mock_layer
        soc_info = NPUSocInfo()
        self.attn_wrapper = ClaWrapper(
                    input_norm_name='input_layernorm',
                    wrapper_name='self_attn',
                    pack_name='qkv_proj',
                    o_name='o_proj',
                    num_attention_heads=80,
                    num_key_value_heads=8,
                    cla_share_factor=2,
                    q_norm_name='query_layernorm',
                    k_norm_name='key_layernorm',
                )
        moe_mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=True
        )
        self.weight_wrapper = HunyuanWeightWrapper(soc_info, 0,
                                                self.attn_wrapper, moe_mlp_wrapper,
                                                16, 18304 // 8)

    def test_register_moe_layer_w8a8(self):
        self.weight_wrapper.gmm_quant_nd_nz = False
        self.weight_wrapper.set_gmm_nd_nz = MagicMock()
        self.weight_wrapper.register_moe_layer(self.mock_layer, QuantType.W8A8_DYNAMIC)


if __name__ == '__main__':
    unittest.main()