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
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from atb_llm.models.hunyuan.config_hunyuan import HunyuanConfig
from atb_llm.models.hunyuan.modeling_hunyuan import (
    HunyuanRMSNorm,
    HunyuanRMSNormBias,
    HunyuanRMSNormWrapper,
    HunyuanRMSNormAntiOutlierWrapper,
    HunyuanMLP,
    HunyuanMoE,
    FlashHunyuanAttention,
    FlashHunyuanDecoderLayer,
    FlashHunyuanModel
)


class TestFlashHunyuanModel(TestCase):
    def setUp(self):
        config_dict = {
            "q_lora_rank": None,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "v_head_dim": 128,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "use_cla": True,
            "cla_share_factor": 2,
            "use_qk_norm": False,
            "num_hidden_layers": 27,
            "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
            }
        }

        self.config = HunyuanConfig.from_dict(config_dict)
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.quantize = None
        self.weights.get_tensor.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_multi_weights_col.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_replicated_weights.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_multi_weights_row.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_whole_tensor.return_value = torch.empty(100, 100, dtype=torch.bfloat16)
        self.weights.get_shape.return_value = [100, 100]
        self.weights.switch_process_group.return_value = None

    def test_hunyuanrmsnorm(self):
        HunyuanRMSNorm("attn", self.weights)
        self.weights.get_tensor.assert_called_once_with("attn.weight")

    def test_hunyuanrmsnormbias(self):
        HunyuanRMSNormBias("attn", self.weights)
        self.weights.get_tensor.assert_called_with("attn.bias")

    def test_hunyuanrmsnormwrapper(self):
        HunyuanRMSNormWrapper("attn", self.weights)
        self.weights.get_tensor.assert_called_with("attn.module.bias")

    def test_hunyuanrmsnormantioutlierwrapper(self):
        HunyuanRMSNormAntiOutlierWrapper("attn", self.weights)
        self.weights.get_tensor.assert_called_with("attn.anti.bias")

    def test_flashhunyuanattention(self):
        FlashHunyuanAttention("attn", 0, self.config, self.weights)
        FlashHunyuanAttention("attn", 1, self.config, self.weights)

    def test_hunyuanmlp(self):
        HunyuanMLP("mlp", self.config, self.weights)

    def test_hunyuanmoe(self):
        HunyuanMoE("moe", self.config, self.weights, HunyuanMLP)

    def test_flashhunyuandecoderlayer(self):
        FlashHunyuanDecoderLayer(0, self.config, self.weights)
        FlashHunyuanDecoderLayer(1, self.config, self.weights)

    def test_flashhunyuanmodel(self):
        FlashHunyuanModel(self.config, self.weights)


if __name__ == '__main__':
    unittest.main()