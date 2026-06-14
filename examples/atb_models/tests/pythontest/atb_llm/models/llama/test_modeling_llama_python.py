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
from unittest.mock import MagicMock, patch

from atb_llm.models.llama.modeling_llama_python import LlamaAttention, LlamaMlp, LlamaLayer, LlamaModel
from atb_llm.nn.tensor import Tensor


class TestPythonLlamaModel(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.vocab_size = 130000

        self.weight_tool = MagicMock()
        self.weight_tool.mapping = MagicMock()

        self.config_metadata = MagicMock()
        self.config_metadata.head_dim = 128
        self.config_metadata.num_attention_heads = 32
        self.config_metadata.num_key_value_heads = 8
    
    @patch("atb_llm.models.llama.modeling_llama_python.MergedColumnParallelLinear")
    @patch("atb_llm.models.llama.modeling_llama_python.RowParallelLinear")
    def test_llama_attention(self, mock_row_linear, mock_column_linear):
        mock_column_linear.load = MagicMock()
        mock_row_linear.load = MagicMock()
        LlamaAttention(self.config, self.weight_tool, "attn", self.config_metadata)

    @patch("atb_llm.models.llama.modeling_llama_python.MergedColumnParallelLinear")
    @patch("atb_llm.models.llama.modeling_llama_python.RowParallelLinear")
    def test_llama_mlp(self, mock_row_linear, mock_column_linear):
        mock_column_linear.load = MagicMock()
        mock_row_linear.load = MagicMock()
        LlamaMlp(self.config, self.weight_tool, "mlp", self.config_metadata)
    
    @patch("atb_llm.models.llama.modeling_llama_python.LlamaAttention", MagicMock())
    @patch("atb_llm.models.llama.modeling_llama_python.LlamaMlp", MagicMock())
    @patch("atb_llm.models.llama.modeling_llama_python.RmsNorm", MagicMock())
    def test_llama_layer(self):
        layer = LlamaLayer(self.config, self.weight_tool, "layer", 0)
        
        inputs = Tensor("inputs")
        cos_emb = Tensor("cos_emb")
        sin_emb = Tensor("sin_emb")
        k_cache = Tensor("k_cache")
        v_cache = Tensor("v_cache")
        slots = Tensor("slots")
        _ = layer(inputs, cos_emb, sin_emb, k_cache, v_cache, slots)
        layer.self_attn.assert_called_once()
        layer.mlp.assert_called_once()
    
    @patch("atb_llm.models.llama.modeling_llama_python.LlamaLayer", MagicMock())
    @patch("atb_llm.models.llama.modeling_llama_python.ParallelEmbedding", MagicMock())
    @patch("atb_llm.models.llama.modeling_llama_python.RmsNorm", MagicMock())
    @patch("atb_llm.models.llama.modeling_llama_python.gather", MagicMock())
    @patch("atb_llm.models.llama.modeling_llama_python.nn.ModuleList")
    def test_llama_model(self, fake_module_list):
        mapping = MagicMock()
        fake_module_list = MagicMock()
        fake_module_list.return_value = [MagicMock for _ in range(28)]
        model = LlamaModel(self.config, self.weight_tool, mapping, "model")

        input_ids = Tensor("input_ids")
        position_ids = Tensor("position_ids")
        cosine_table = Tensor("cosine_table")
        sine_table = Tensor("sine_table")
        k_caches = [Tensor(f"k_caches_{i}" for i in range(28))]
        v_caches = [Tensor(f"v_caches_{i}" for i in range(28))]
        
        _ = model.forward(input_ids, position_ids, cosine_table, sine_table, k_caches, v_caches)
        model.embed_tokens.assert_called_once()
        for i in range(28):
            model.layers[i].assert_called_once()
        model.norm.assert_called_once()


if __name__ == "__main__":
    unittest.main()