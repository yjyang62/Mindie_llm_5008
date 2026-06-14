# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn

from atb_llm.models.qwen3_vl_moe.modeling_qwen3_vl_moe_text import (
    TensorParallelColumnStackedMOE,
    TensorParallelRowStackedMOE,
    Qwen3VLMOETextAttention,
    Qwen3VLMOETextMOE,
    Qwen3VLTextDecoderLayer,
    FlashQwen3VLMOETextModelForCausalLM
)
from atb_llm.utils.quantize.quant_type import QuantType

MODELING_QWEN3VL_MOE = "atb_llm.models.qwen3_vl_moe.modeling_qwen3_vl_moe_text"
INITIAL = "atb_llm.utils.initial"


class TestQwen3VLMOETextComponents(unittest.TestCase):

    def setUp(self):
        self.prefix = ""
        self.config = Mock()
        self.config.quantize = None
        self.config.num_experts = 4
        self.config.num_experts_per_tok = 2
        self.config.head_dim = 128
        self.config.rms_norm_eps = 1e-6
        self.config.num_attention_heads = 32
        self.config.num_key_value_heads = 8
        self.config.num_hidden_layers = 1
        self.config.max_position_embeddings = 2048
        self.config.rope_theta = 10000.0
        self.config.rope_scaling = Mock()
        self.config.rope_scaling.mrope_section = [1, 1, 1]
        self.config.is_dense_layer = False
        self.config.quantization_config = Mock()
        self.config.quantization_config.group_size = 128
        self.config.model_type = "qwen3"
        self.config.hidden_size = 4096

        self.weights = Mock()
        self.weights.process_group = Mock()
        self.weights.process_group.size = Mock(return_value=1)
        self.weights.process_group.rank = Mock(return_value=0)
        self.weights.device = "cpu"
        self.weights.dtype = torch.float16

        self.mock_weights_data = {}
        self.weights.get_tensor = Mock(side_effect=lambda x: self.mock_weights_data.get(x, torch.randn(10, 10)))

        with patch(f"{MODELING_QWEN3VL_MOE}.TensorEmbedding") as mock_tensor_embedding, \
             patch(f"{MODELING_QWEN3VL_MOE}.Qwen3VLTextDecoderLayer") as mock_decoder_layer, \
             patch(f"{MODELING_QWEN3VL_MOE}.load_column_multi") as mock_load, \
             patch(f"{MODELING_QWEN3VL_MOE}.RMSNorm") as mock_rmsnorm, \
             patch(f"{MODELING_QWEN3VL_MOE}.Qwen3VLTextRotaryEmbedding") as mock_rotary_embedding, \
             patch(f"{INITIAL}.NPUSocInfo") as mock_soc_info:

            self.mock_soc_info_instance = Mock()
            self.mock_soc_info_instance.need_nz = False
            self.mock_soc_info_instance.soc_version = 220
            self.mock_soc_info_instance.communication_backend = "nccl"
            mock_soc_info.return_value = self.mock_soc_info_instance

            mock_rotary_embedding.return_value = Mock()
            mock_rotary_embedding.return_value.return_value = (torch.randn(2048, 128), torch.randn(2048, 128))
            mock_rmsnorm.return_value = MagicMock()
            mock_load.return_value = MagicMock()
            mock_tensor_embedding.return_value = MagicMock()
            mock_decoder_layer.return_value = None

            self.model = FlashQwen3VLMOETextModelForCausalLM(self.config, self.weights)

    def test_parallel_column_moe_load_moe(self):
        prefix_list = ["test.experts.gate_up_proj"]
        self.mock_weights_data["test.experts.gate_up_proj"] = torch.randn(1, 128, 256)

        with self.assertRaises(NotImplementedError):
            TensorParallelColumnStackedMOE.load_moe(self.config, prefix_list, self.weights, bias=True)

        with self.assertRaises(NotImplementedError):
            TensorParallelColumnStackedMOE.load_moe(self.config, ["prefix1", "prefix2"], self.weights, bias=False)

    def test_parallel_column_moe_get_col_packed_mlp(self):
        prefix = "test.experts.gate_up_proj"
        tensor = torch.randn(1, 128, 256)
        self.mock_weights_data[prefix] = tensor

        result = TensorParallelColumnStackedMOE.get_col_packed_mlp(prefix, self.weights)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[2], 128)

    def test_parallel_row_moe_load_moe(self):
        prefix_list = ["test.experts.down_proj"]
        self.mock_weights_data["test.experts.down_proj"] = torch.randn(128, 256)

        with self.assertRaises(NotImplementedError):
            TensorParallelRowStackedMOE.load_moe(self.config, prefix_list, self.weights.process_group, self.weights, bias=True)

        with self.assertRaises(NotImplementedError):
            TensorParallelRowStackedMOE.load_moe(self.config, ["prefix1", "prefix2"], self.weights.process_group, self.weights, bias=False)

    @patch(f"{MODELING_QWEN3VL_MOE}.load_column_multi")
    @patch(f"{MODELING_QWEN3VL_MOE}.RMSNorm")
    @patch(f"{MODELING_QWEN3VL_MOE}.TensorParallelRowLinear.load")
    def test_attention_init(self, mock_load, mock_rmsnorm, mock_load_column_multi):
        attention = Qwen3VLMOETextAttention(self.prefix, self.config, self.weights)
        mock_load.assert_called_once_with(
            self.config,
            prefix=f"{self.prefix}.o_proj",
            weights=self.weights,
            bias=False,
            gqa_size=self.config.head_dim,
        )
        mock_rmsnorm.assert_any_call(
            prefix=f"{self.prefix}.q_norm",
            weights=self.weights,
            eps=self.config.rms_norm_eps
        )
        mock_rmsnorm.assert_any_call(
            prefix=f"{self.prefix}.k_norm",
            weights=self.weights,
            eps=self.config.rms_norm_eps
        )
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=[f"{self.prefix}.q_proj", f"{self.prefix}.k_proj", f"{self.prefix}.v_proj"],
            weights=self.weights,
            head_size=self.config.head_dim,
            bias=False,
        )
        self.assertIsInstance(attention, nn.Module)
        self.assertEqual(attention.head_size, self.config.head_dim)

    @patch(f"{MODELING_QWEN3VL_MOE}.FastLinear.load")
    @patch(f"{MODELING_QWEN3VL_MOE}.TensorParallelColumnStackedMOE.load_moe")
    @patch(f"{MODELING_QWEN3VL_MOE}.TensorParallelRowStackedMOE.load_moe")
    def test_moe_init(self, mock_row_load_moe, mock_column_load_moe, mock_fastlinear_load):

        mock_column_load_moe.return_value = Mock()
        mock_row_load_moe.return_value = Mock()
        mock_fastlinear_load.return_value = Mock()
        moe = Qwen3VLMOETextMOE(self.prefix, self.config, self.weights)

        mock_fastlinear_load.assert_called_once_with(
            prefix=f"{self.prefix}.gate",
            weights=self.weights,
            bias=False
        )
        mock_column_load_moe.assert_called_once_with(
            self.config,
            prefix_list=[f"{self.prefix}.experts.gate_up_proj"],
            weights=self.weights,
            bias=False
        )
        mock_row_load_moe.assert_called_once_with(
            self.config,
            prefix_list=[f"{self.prefix}.experts.down_proj"],
            process_group=self.weights.process_group,
            weights=self.weights,
            bias=False
        )

        self.assertIsInstance(moe, nn.Module)

    @patch(f"{MODELING_QWEN3VL_MOE}.FastLinear.load")
    @patch(f"{MODELING_QWEN3VL_MOE}.TensorParallelColumnLinear.load_moe")
    @patch(f"{MODELING_QWEN3VL_MOE}.TensorParallelRowLinear.load_moe")
    def test_moe_init_w8a8_case(self, mock_row_load_moe, mock_column_load_moe, mock_fastlinear_load):

        mock_column_load_moe.return_value = Mock()
        mock_row_load_moe.return_value = Mock()
        mock_fastlinear_load.return_value = Mock()
        self.config.quantize = "w8a8"

        moe = Qwen3VLMOETextMOE(self.prefix, self.config, self.weights)

        device_expert = [i for i in range(self.config.num_experts)]
        pack_prefixes = [[f"{self.prefix}.experts.{i}.gate_proj", f"{self.prefix}.experts.{i}.up_proj"] \
                            for i in device_expert]

        mock_fastlinear_load.assert_called_once_with(
            prefix=f"{self.prefix}.gate",
            weights=self.weights,
            bias=False
        )
        mock_column_load_moe.assert_called_once_with(
            self.config,
            prefix_list=pack_prefixes,
            weights=self.weights,
            bias=False
        )
        mock_row_load_moe.assert_called_once_with(
            self.config,
            prefix_list=[f"{self.prefix}.experts.{i}.down_proj" for i in device_expert],
            process_group=self.weights.process_group,
            weights=self.weights,
            bias=False
        )

        self.assertIsInstance(moe, nn.Module)

    @patch(f"{MODELING_QWEN3VL_MOE}.Qwen3VLMOETextAttention")
    @patch(f"{MODELING_QWEN3VL_MOE}.Qwen3VLMOETextMOE")
    def test_decoder_layer_init(self, mock_text_moe, mock_text_attn):

        mock_text_attn.return_value = Mock()
        mock_text_moe.return_value = Mock()

        def mock_load_weights():
            pass
        Qwen3VLTextDecoderLayer.load_weights = Mock(side_effect=mock_load_weights)

        layer = Qwen3VLTextDecoderLayer(0, self.config, self.weights, self.prefix)
        mock_text_attn.assert_called_once_with(
            prefix=f"{self.prefix}.layers.0.self_attn",
            config=self.config,
            weights=self.weights
        )
        mock_text_moe.assert_called_once_with(
            prefix=f"{self.prefix}.layers.0.mlp",
            config=self.config,
            weights=self.weights
        )

        self.assertIsInstance(layer, nn.Module)
        self.assertIsNotNone(layer.self_attn)
        self.assertIsNotNone(layer.mlp)

    def test_causal_lm_init(self):

        self.assertIsInstance(self.model, nn.Module)
        self.assertIsNotNone(self.model.embed_tokens)
        self.assertIsNotNone(self.model.layers)
        self.assertIsNotNone(self.model.norm)
        self.assertIsNotNone(self.model.lm_head)
        self.assertIsNotNone(self.model.graph_manager)

    @patch(f"{MODELING_QWEN3VL_MOE}.MoeWeightWrapper")
    def test_init_ascend_weight(self, mock_weight_wrapper):

        mock_weight_wrapper_ins = mock_weight_wrapper.return_value
        mock_weight_wrapper_ins.register_embedding = MagicMock()
        mock_weight_wrapper_ins.register_layer_weights = MagicMock()
        mock_weight_wrapper_ins.register_model_norm = MagicMock()
        mock_weight_wrapper_ins.register_model_lmhead = MagicMock()
        mock_weight_wrapper_ins.attn_linear_types = [[1, 2, 3, 4]]
        mock_weight_wrapper_ins.mlp_linear_types = [[1, 2, 3, 4]]
        mock_weight_wrapper_ins.moe_linear_types = [[1, 2, 3, 4]]
        mock_weight_wrapper_ins.attn_linear_transpose_types = [[1, 2, 3, 4]]
        mock_weight_wrapper_ins.mlp_linear_transpose_types = [[1, 2, 3, 4]]
        mock_weight_wrapper_ins.moe_linear_transpose_types = [[1, 2, 3, 4]]

        self.model.graph_manager = MagicMock()
        self.model.graph_manager.set_param.return_value = True
        self.model.init_ascend_weight()
        self.model.graph_manager.set_param.assert_called()

    def test_prepare_inputs(self):

        self.model.cos_embed = torch.randn(2048, 128)
        self.model.sin_embed = torch.randn(2048, 128)
        self.model.placeholder = torch.zeros(1, dtype=torch.float32, device="npu")
        self.model.expert_array = torch.tensor([1, 2, 3, 4], dtype=torch.int32).npu()
        self.model.expert_group = torch.tensor([1], dtype=torch.int32).npu()
        self.model.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.model.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.model.qlen_decorator = Mock()
        self.model.qlen_decorator.modify_inputs.return_value = None
        self.model.soc_info = self.mock_soc_info_instance

        input_ids = torch.randint(0, 1000, (2, 10))
        position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        is_prefill = True
        kv_cache = [(torch.randn(2, 8, 10, 128), torch.randn(2, 8, 10, 128)) for _ in range(2)]
        block_tables = torch.randint(0, 100, (2, 5))
        slots = torch.randint(0, 1000, (10,))
        input_lengths = torch.tensor([5, 5])
        max_seq_len = 10
        self.model.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache, block_tables, slots,
                                        input_lengths, max_seq_len)
        self.assertEqual(len(self.model.acl_operation_inputs), 16)

    def test_init_kvcache(self):

        self.model.acl_encoder_operation = MagicMock()
        self.model.acl_decoder_operation = MagicMock()
        cache = torch.zeros([1, 128, 1, 128], dtype=torch.float16).npu()
        self.model.init_kvcache([(cache, cache)])
        self.assertEqual(self.model.ascend_kcache_id, id(cache))

    def test_forward(self):

        self.model.graph_manager = MagicMock()
        self.model.graph_manager.select_and_execute.return_value = torch.zeros([1024], dtype=torch.float16).npu()
        self.model.execute_ascend_operator([], {}, True)
        self.model.graph_manager.select_and_execute.assert_called_once()

    def test_update_matmul_params(self):

        self.model.soc_info.soc_version = 223
        self.model.dtype = torch.float16
        self.model._update_matmul_params(quantize=QuantType.FLOAT)
        self.assertFalse(self.model.matmul_nd_nz)
        self.model._update_matmul_params(quantize=QuantType.W8A8)
        self.assertTrue(self.model.matmul_nd_nz)
        self.model.soc_info.soc_version = 0
        self.model._update_matmul_params(quantize=QuantType.FLOAT)
        self.assertFalse(self.model.matmul_nd_nz)

    def test_update_thw_cos_sin(self):

        self.model.cos_embed = torch.randn(2048, 128)
        self.model.sin_embed = torch.randn(2048, 128)
        self.model.mrope_section = [1, 1, 1]
        self.model.apply_interleaved_mrope = Mock()
        self.model.apply_interleaved_mrope.side_effect = lambda x, section: x

        position_ids_thw = torch.tensor([0, 1, 2, 3, 4])

        cos, sin = self.model.update_thw_cos_sin(position_ids_thw)

        self.assertIsInstance(cos, torch.Tensor)
        self.assertIsInstance(sin, torch.Tensor)
        self.assertEqual(cos.shape, (5, 1, 256))
        self.assertEqual(sin.shape, (5, 1, 256))

if __name__ == '__main__':
    unittest.main()