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

from atb_llm.models.qwen2.modeling_qwen2_python import Qwen2Attention, Qwen2Mlp, Qwen2Layer, Qwen2Model
from atb_llm.nn.tensor import Tensor
from atb_llm.layers.linear.linear import MergedColumnParallelLinear, RowParallelLinear
from atb_llm.layers.norm.normalization import RmsNorm


MODELING_QWEN2 = "atb_llm.models.qwen2.modeling_qwen2_python"


class TestPythonQwen2Model(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MagicMock()
        self.config.vocab_size = 160000

        self.file_loader = MagicMock()
        self.file_loader.mapping = MagicMock()

        self.config_metadata = MagicMock()
        self.config_metadata.head_dim = 128
        self.config_metadata.num_attention_heads = 40
        self.config_metadata.num_key_value_heads = 8

    @patch(f"{MODELING_QWEN2}.MergedColumnParallelLinear")
    @patch(f"{MODELING_QWEN2}.RowParallelLinear")
    @patch(f"{MODELING_QWEN2}.RmsNorm")
    def test_qwen2_attention(
            self,
            mock_norm: RmsNorm,
            mock_row_linear: MergedColumnParallelLinear,
            mock_column_linear: RowParallelLinear
    ) -> None:
        mock_norm.load.return_value = MagicMock("norm_out")
        mock_row_linear.load = MagicMock()
        mock_column_linear.load = MagicMock()
        Qwen2Attention(self.config, self.file_loader, "attn", self.config_metadata)

    @patch(f"{MODELING_QWEN2}.MergedColumnParallelLinear")
    @patch(f"{MODELING_QWEN2}.RowParallelLinear")
    def test_qwen2_mlp(
            self,
            mock_row_linear: MergedColumnParallelLinear,
            mock_column_linear: RowParallelLinear
    ) -> None:
        mock_row_linear.load = MagicMock()
        mock_column_linear.load = MagicMock()
        Qwen2Mlp(self.config, self.file_loader, "mlp", self.config_metadata)

    @patch(f"{MODELING_QWEN2}.Qwen2Attention", MagicMock())
    @patch(f"{MODELING_QWEN2}.Qwen2Mlp", MagicMock())
    @patch(f"{MODELING_QWEN2}.RmsNorm", MagicMock())
    def test_qwen2_layer(self) -> None:
        layer = Qwen2Layer(self.config, self.file_loader, "layer", 0)
        inputs = Tensor("inputs")
        cos_emb = Tensor("cos_emb")
        sin_emb = Tensor("sin_emb")
        k_cache = Tensor("k_cache")
        v_cache = Tensor("v_cache")
        slots = Tensor("slots")
        _ = layer(inputs, cos_emb, sin_emb, k_cache, v_cache, slots)(inputs, cos_emb, sin_emb, k_cache, v_cache)
        layer.self_attn.assert_called_once()
        layer.mlp.assert_called_once()

    @patch(f"{MODELING_QWEN2}.Qwen2Layer", MagicMock())
    @patch(f"{MODELING_QWEN2}.ParallelEmbedding", MagicMock())
    @patch(f"{MODELING_QWEN2}.RmsNorm", MagicMock())
    @patch(f"{MODELING_QWEN2}.gather", MagicMock())
    @patch(f"{MODELING_QWEN2}.nn.ModuleList")
    def test_qwen2_model(self, fake_module_list) -> None:
        mapping = MagicMock()
        fake_module_list = MagicMock()
        fake_module_list.return_value = [MagicMock for _ in range(28)]
        model = Qwen2Model(self.config, self.file_loader, mapping, "model")
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
