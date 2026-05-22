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

import torch
from ddt import ddt

from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.models.qwen2.modeling_qwen2 import (
    QwenMLP, FlashQwenLayer, FlashQwenModel, FlashQwenAttention
)
from atb_llm.utils.quantize.pack_type import PackType

FAKE_CONFIG_DICT = {
    'model_type': 'qwen2',
    'num_hidden_layers': 32,
    'max_position_embeddings': 2048,
    'vocab_size': 65536
}


@ddt
class TestFlashQwen2Model(unittest.TestCase):
    def setUp(self):
        self.config = Qwen2Config.from_dict(FAKE_CONFIG_DICT)
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.dtype = torch.float16
        self.prefix = ""
        self.head_size = 128
        self.hidden_size = 1024
        self.layer_id = 0
        self.attn_decode_backend = 0
        self.weights.dtype = self.dtype
        self.weights.quantize = None
        self.weights.get_tensor.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_multi_weights_col.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_replicated_weights.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_multi_weights_row.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_whole_tensor.return_value = torch.empty(100, 100, dtype=self.dtype)
        self.weights.get_shape.return_value = [100, 100]
        self.weights.switch_process_group.return_value = None
        self.weights.process_group.size = MagicMock(return_value=4)
        self.weights.process_group.rank = MagicMock(return_value=3)

    @patch('atb_llm.models.qwen2.modeling_qwen2.load_column_multi')
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelRowLinear.load')
    def test_qwen_mlp(self, mock_get_tensor, mock_load_column_multi):
        qwen_mlp_instance = QwenMLP(self.prefix, self.config, self.weights)
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=[f"{self.prefix}.gate_proj", f"{self.prefix}.up_proj"],
            weights=self.weights,
            head_size=1,
        )
        mock_get_tensor.assert_called_once_with(
            self.config,
            prefix=f"{self.prefix}.down_proj",
            weights=self.weights,
            bias=False,
        )
        self.assertIsNotNone(qwen_mlp_instance.intermediate_size)

    @patch('atb_llm.models.qwen2.modeling_qwen2.calc_linear_pack_type', return_value=PackType.ALL_FP)
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelColumnLinear.load')
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelRowLinear.load')
    def test_qwen_mlp_case_w8a8sc(self, mock_row_load, mock_col_load, _):
        self.config = Qwen2Config.from_dict({"quantize": "w8a8sc", **FAKE_CONFIG_DICT})
        qwen_mlp_instance = QwenMLP(self.prefix, self.config, self.weights)
        mock_col_load.assert_called_once_with(
            self.config,
            prefix=f"{self.prefix}.w2_w1",
            weights=self.weights,
            bias=False,
        )
        mock_row_load.assert_called_once_with(
            self.config,
            prefix=f"{self.prefix}.c_proj",
            weights=self.weights,
            bias=False,
        )
        self.assertIsNotNone(qwen_mlp_instance.intermediate_size)

    @patch('atb_llm.models.qwen2.modeling_qwen2.load_column_multi')
    def test_flash_qwen_layer(self, mock_load_column_multi):
        flash_qwen_layer_instance = FlashQwenLayer(self.layer_id, self.config, self.weights, self.prefix,
        self.attn_decode_backend)
        self.assertIsNotNone(flash_qwen_layer_instance.mlp)
        self.assertIsNotNone(flash_qwen_layer_instance.attn)

    @patch('atb_llm.models.qwen2.modeling_qwen2.load_column_multi')
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelEmbedding')
    def test_flash_qwen_model(self, mock_tensor_parallel_embedding, mock_load_column_multi):
        flash_qwen_model_instance = FlashQwenModel(self.config, self.weights)
        self.assertIsNotNone(flash_qwen_model_instance.head_size)
        self.assertIsNotNone(flash_qwen_model_instance.num_heads)

    @patch('atb_llm.models.qwen2.modeling_qwen2.load_column_multi')
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelRowLinear.load')
    def test_flash_qwen_attention(self, mock_load, mock_load_column_multi):
        _ = FlashQwenAttention(self.prefix, self.config, self.weights, self.attn_decode_backend)
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=[f"{self.prefix}.q_proj", f"{self.prefix}.k_proj", f"{self.prefix}.v_proj"],
            weights=self.weights,
            head_size=self.head_size,
            bias=True,
        )
        mock_load.assert_called_once_with(
            self.config,
            prefix=f"{self.prefix}.o_proj",
            weights=self.weights,
            gqa_size=self.head_size,
            bias=False,
        )

        # 模拟量化类型为 "w8a8sc"
        self.config.quantize = "w8a8sc"
        flash_qwen_attention_instance = FlashQwenAttention(self.prefix, self.config, self.weights,
        self.attn_decode_backend)

        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=[f"{self.prefix}.q_proj", f"{self.prefix}.k_proj", f"{self.prefix}.v_proj"],
            weights=self.weights,
            head_size=self.head_size,
            bias=True,
        )
        self.assertIsNotNone(flash_qwen_attention_instance.c_proj)

    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelColumnLinear.load')
    @patch('atb_llm.models.qwen2.modeling_qwen2.load_column_multi')
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelRowLinear.load')
    def test_flash_qwen_attention_init_else(self, mock_load_row_linear, mock_load_column_multi,
    mock_load_column_linear):
        mock_config = MagicMock()
        mock_config.num_attention_heads = 32
        mock_config.hidden_size = 4096
        mock_config.quantize = "w8a8"
        mock_config.quantization_config = MagicMock()
        mock_config.quantization_config.kv_quant_type = None
        mock_config.quantization_config.fa_quant_type = None

        mock_config.pack_type = "w8a8"

        mock_weights = MagicMock()
        mock_weights.device = torch.device("cpu")
        mock_weights.dtype = torch.float16
        mock_weights.get_tensor.return_value = torch.empty(100, 100, dtype=torch.float16)
        mock_weights.get_shape.return_value = [100, 100]

        self.prefix = "test_prefix"
        attn_decode_backend = 0

        flash_qwen_attention_instance = FlashQwenAttention(
            self.prefix, mock_config, mock_weights, attn_decode_backend
        )
        self.assertIsNotNone(flash_qwen_attention_instance.prefix)
    
    def test_flash_qwen_attention_case_moe(self):
        self.config.quantize = "w8a8_dynamic"
        self.config.model_type = "qwen3_moe"
        self.config.hidden_size = 256
        self.config.num_attention_heads = 8

        flash_qwen_attention_instance = FlashQwenAttention(self.prefix, self.config, self.weights, self.attn_decode_backend)
        self.assertEqual(flash_qwen_attention_instance.head_size, 32)

    @patch('atb_llm.models.qwen2.modeling_qwen2.calc_linear_pack_type', return_value=PackType.ALL_W16A16SC)
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelColumnLinear.load')
    def test_flash_qwen_attention_case_pack_w16a16sc(self, mock_load_column, mock_calc_linear):
        flash_qwen_attention_instance = FlashQwenAttention(self.prefix, self.config, self.weights,self.attn_decode_backend)
        mock_load_column.assert_called_once_with(
            self.config,
            prefix=f"{self.prefix}.c_attn",
            weights=self.weights,
            bias=self.config.attention_bias,
        )
        mock_calc_linear.assert_called_once()
        self.assertEqual(flash_qwen_attention_instance.c_attn.linear.num_linear_before_pack, 3)

    @patch('atb_llm.models.qwen2.modeling_qwen2.calc_linear_pack_type', return_value=PackType.PACK_QUANT_UNDEFINED)
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelColumnLinear.load')
    def test_flash_qwen_attention_case_else(self, mock_load_column, mock_calc_linear):
        _ = FlashQwenAttention(self.prefix, self.config, self.weights,self.attn_decode_backend)
        mock_load_column.assert_any_call(
            self.config,
            prefix=f"{self.prefix}.q_proj",
            weights=self.weights,
            bias=self.config.attention_bias,
        )
        mock_load_column.assert_any_call(
            self.config,
            prefix=f"{self.prefix}.k_proj",
            weights=self.weights,
            bias=self.config.attention_bias,
        )
        mock_load_column.assert_any_call(
            self.config,
            prefix=f"{self.prefix}.v_proj",
            weights=self.weights,
            bias=self.config.attention_bias,
        )

    @patch('atb_llm.models.qwen2.modeling_qwen2.calc_linear_pack_type', return_value=PackType.ALL_W16A16SC)
    def test_flash_qwen_layer_case_quantize_w16a16sc(self, mock_calc_linear):
        self.config.quantize = "w16a16sc"
        flash_qwen_layer_instance = FlashQwenLayer(self.layer_id, self.config, self.weights, self.prefix,
        self.attn_decode_backend)
        self.assertIsNotNone(flash_qwen_layer_instance.mlp)
        self.assertIsNotNone(flash_qwen_layer_instance.attn)

    @patch('atb_llm.models.qwen2.modeling_qwen2.calc_linear_pack_type', return_value=PackType.PACK_QUANT_UNDEFINED)
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelColumnLinear.load')
    @patch('atb_llm.models.qwen2.modeling_qwen2.TensorParallelRowLinear.load')
    def test_qwen_mlp_case_else(self, mock_row_load, mock_col_load, _):
        qwen_mlp_instance = QwenMLP(self.prefix, self.config, self.weights)
        mock_col_load.assert_any_call(
            self.config,
            prefix=f"{self.prefix}.gate_proj",
            weights=self.weights,
            bias=False,
        )
        mock_col_load.assert_any_call(
            self.config,
            prefix=f"{self.prefix}.up_proj",
            weights=self.weights,
            bias=False,
        )
        mock_row_load.assert_called_once()
        self.assertIsNotNone(qwen_mlp_instance.intermediate_size)


if __name__ == '__main__':
    unittest.main()