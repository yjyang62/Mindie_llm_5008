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
from unittest.mock import patch, MagicMock
import torch
from torch import multiprocessing as mp
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader_process import EplbLoaderProcess
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.layers import get_linear
from atb_llm.utils.layers import TensorParallelColumnLinear


FAKE_CONFIG_DICT = {
    'max_position_embeddings': 16384,
    'vocab_size': 10240,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "tie_word_embeddings": False,
    "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
            "parallel_embedding": True
            }
}


class MyClass:
    @staticmethod
    def my_static_method():
        return "Original static response"


class TestEplbLoaderProcess(unittest.TestCase):
    def setUp(self):
        weights = None
        self.process = EplbLoaderProcess(weights)
        self.config = DeepseekV2Config(**FAKE_CONFIG_DICT)


    def tearDown(self):
        self.process.shutdown()
        self.process = None

    @patch('atb_llm.utils.layers.TensorParallelColumnLinear.load_moe')
    def test__do_load_weight_column_linear_from_ssd_(self, mock_load_moe):
        quantize = QuantType.W8A8_DYNAMIC
        weight = torch.ones(256, 256, dtype=torch.int8)
        scale = torch.ones(256, 1, dtype=torch.float32)
        offset = torch.ones(256, 1, dtype=torch.float16)
        weights = MagicMock()
        prefixes = [[f"moe.{i}.gate_up_proj"] for i in range(64)]
        bias = None

        linear = get_linear((weight, scale, offset), bias, quantize, False)
        result = TensorParallelColumnLinear(linear)
        mock_load_moe.return_value = result

        result, tensor_weight, tensor_weight_scale, tensor_weight_offset = \
            self.process._do_load_weight_column_linear_from_ssd_(
                weights, self.config, prefixes, bias
            )
        self.assertEqual(linear, result.linear)

    @patch('atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader_process.EplbLoaderProcess._do_load_weight_column_linear_from_ssd_')
    def test__do_load_weight_column_linear_from_ssd(self, mock_load_weight_column_linear_from_ssd):
        weights = MagicMock()
        mock_load_weight_column_linear_from_ssd.return_value = None, None, None, None
        spawn_ctx = mp.get_context("spawn")
        in_q = spawn_ctx.Queue()
        out_q = spawn_ctx.Queue()
        in_q.put((None, [[f"moe.{i}.gate_up_proj"] for i in range(64)], None))
        in_q.put("exist")
        self.process._do_load_weight_column_linear_from_ssd(weights, in_q, out_q)

    def test_load_weight_column_linear_from_ssd(self):
        print("test_load_weight_column_linear_from_ssd")
        quantize = QuantType.W8A8_DYNAMIC
        weight = torch.ones(256, 256, dtype=torch.int8)
        scale = torch.ones(256, 1, dtype=torch.float32)
        offset = torch.ones(256, 1, dtype=torch.float16)
        prefixes = [[f"moe.{i}.gate_up_proj"] for i in range(64)]
        bias = None
        linear = get_linear((weight, scale, offset), bias, quantize, False)
        result = TensorParallelColumnLinear(linear)

        print("mock_load_moe.return_value = linear")
        self.process.weight_column_linear_out.put((result, weight, scale, offset))
        result = self.process.load_weight_column_linear_from_ssd(self.config, prefixes, bias)
        self.assertTrue(torch.allclose(linear.weight, result.linear.weight))

    @patch('atb_llm.utils.layers.TensorParallelRowLinear.load_moe')
    def test__do_load_weight_row_linear_from_ssd_(self, mock_load_moe):
        print("test_load_weight_column_linear_from_ssd")
        quantize = QuantType.W8A8_DYNAMIC
        weight = torch.ones(256, 256, dtype=torch.int8)
        scale = torch.ones(256, 1, dtype=torch.float32)
        offset = torch.ones(256, 1, dtype=torch.float16)
        weights = MagicMock()
        prefixes = [[f"moe.{i}.gate_up_proj"] for i in range(64)]
        bias = None

        linear = get_linear((weight, scale, offset), bias, quantize, False)
        result = TensorParallelColumnLinear(linear)
        mock_load_moe.return_value = result

        process_group = "process_group"
        result, tensor_weight, tensor_weight_scale, tensor_weight_offset = \
            self.process._do_load_weight_row_linear_from_ssd_(
                weights, self.config, prefixes, process_group, bias
            )
        self.assertEqual(linear, result.linear)

    @patch('atb_llm.models.deepseekv2.eplb.eplb_loader.eplb_loader_process.EplbLoaderProcess._do_load_weight_row_linear_from_ssd_')
    def test__do_load_weight_row_linear_from_ssd(self, _do_load_weight_row_linear_from_ssd_):
        weights = MagicMock()
        _do_load_weight_row_linear_from_ssd_.return_value = None, None, None, None
        spawn_ctx = mp.get_context("spawn")
        in_q = spawn_ctx.Queue()
        out_q = spawn_ctx.Queue()
        in_q.put((None, [[f"moe.{i}.gate_up_proj"] for i in range(64)], "process_group", None))
        in_q.put("exist")
        self.process._do_load_weight_row_linear_from_ssd(weights, in_q, out_q)

    def test_load_weight_row_linear_from_ssd(self):
        print("test_load_weight_column_linear_from_ssd")
        quantize = QuantType.W8A8_DYNAMIC
        weight = torch.ones(256, 256, dtype=torch.int8)
        scale = torch.ones(256, 1, dtype=torch.float32)
        offset = torch.ones(256, 1, dtype=torch.float16)
        prefixes = [[f"moe.{i}.gate_up_proj"] for i in range(64)]
        bias = None
        linear = get_linear((weight, scale, offset), bias, quantize, False)
        result = TensorParallelColumnLinear(linear)

        print("mock_load_moe.return_value = linear")
        self.process.weight_row_linear_out.put((result, weight, scale, offset))
        process_group = "process_group"
        result = self.process.load_weight_row_linear_from_ssd(self.config, prefixes, process_group, bias)
        self.assertTrue(torch.allclose(linear.weight, result.linear.weight))

    def test_shundown(self):
        self.process.shutdown()
        result = self.process.load_weight_column_linear_from_ssd(None, None, None)
        self.assertIsNone(result)
        result = self.process.load_weight_row_linear_from_ssd(None, None, None, None)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
