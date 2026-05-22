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

from atb_llm.models.internlm2.config_internlm2 import Internlm2Config
from atb_llm.models.internlm2.flash_causal_internlm2 import FlashInternlm2ForCausalLM
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


class TestFlashInternlm2ForCausalLM(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.config = Internlm2Config(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=2,
            rope_theta=1000000,
            rope_scaling={"type": "dynamic", "factor": 2.0},
            max_position_embeddings=1024,
            vocab_size=1000,
        )
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.process_group = MagicMock()
        self.weights.process_group.rank.return_value = 0
        self.weights.process_group.size.return_value = 2

        # 把公共的patch放到setUp中
        self.patcher_init_so = patch('atb_llm.models.base.flash_causal_lm.load_atb_speed')
        self.mock_init_so = self.patcher_init_so.start()

        self.patcher_model = patch('atb_llm.models.internlm2.flash_causal_internlm2.FlashInternlm2Model')
        self.mock_internlm2_model = self.patcher_model.start()

        self.patcher_load_column_multi = patch('atb_llm.models.internlm2.flash_causal_internlm2.load_column_multi')
        self.mock_load_column_multi = self.patcher_load_column_multi.start()

        self.addCleanup(self.patcher_init_so.stop)
        self.addCleanup(self.patcher_model.stop)
        self.addCleanup(self.patcher_load_column_multi.stop)

    def test_init(self):
        lmhead_prefix = "output"
        model_prefix = "model"
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights, lmhead_prefix, model_prefix)
        self.mock_internlm2_model.assert_called_once_with(
            self.config,
            self.weights,
            model_prefix,
        )
        self.mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=[lmhead_prefix],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )
        self.assertIsInstance(flash_model.model, MagicMock)
        self.assertEqual(flash_model.config.num_hidden_layers, self.config.num_hidden_layers)
        self.assertEqual(flash_model.scaling_type, "dynamic")

    @patch('atb_llm.models.internlm2.flash_causal_internlm2.TensorHead')
    def test_init_with_quantize(self, mock_tensor_head):
        self.config.quantize = "w8a8sc"
        mock_tensor_head.load_weight = MagicMock()
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        mock_tensor_head.load_weight.assert_called_once()
        self.assertEqual(flash_model.quantize, self.config.quantize)

    def test_init_with_diff_rope_scaling(self):
        # 默认是"dynamic", test_init 已经测过了
        self.config.rope_scaling.type = "linear"
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        self.assertEqual(flash_model.scaling_type, self.config.rope_scaling.type)
        self.assertIsNotNone(flash_model.rotary_embedding)

        self.config.rope_scaling.type = "other"
        with self.assertRaises(ValueError) as _:
            flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)

        self.config.rope_scaling = None
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        self.assertEqual(flash_model.rope_scaling, self.config.rope_scaling)
        self.assertIsNone(flash_model.rope_scaling)

    def test_init_position_rotary_embedding(self):
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        max_seq_len = 1024
        position_ids = torch.tensor(range(max_seq_len)).to(flash_model.device)
        flash_model.init_position_rotary_embedding(position_ids, max_seq_len)
        self.assertIsNotNone(flash_model.cos_embed)
        self.assertIsNotNone(flash_model.sin_embed)

    def test_get_weights(self):
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        weight_wrapper = flash_model.get_weights()
        self.assertIsNotNone(weight_wrapper)

    def test_get_coder_param(self):
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        encoder_param, decoder_param = flash_model.get_coder_param()
        self.assertIsNotNone(encoder_param)
        self.assertIsNotNone(decoder_param)

    def test_prepare_inputs_for_ascend(self):
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        flash_model_inputs = self.get_flash_model_inputs(flash_model.device)
        flash_model.prepare_inputs_for_ascend(*flash_model_inputs)
        self.assertIsNotNone(flash_model.acl_operation_inputs)

    def test_prepare_inputs_for_ascend_with_bf16(self):
        self.weights.dtype = torch.bfloat16
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        self.assertEqual(flash_model.dtype, torch.bfloat16)
        flash_model_inputs = self.get_flash_model_inputs(flash_model.device)
        flash_model.prepare_inputs_for_ascend(*flash_model_inputs)
        self.assertIsNotNone(flash_model.acl_operation_inputs)

    def test_other_func(self):
        flash_model = FlashInternlm2ForCausalLM(self.config, self.weights)
        flash_model.init_ascend_operations(self.config)
        tok_embeddings = flash_model.get_embedding_layer()
        self.assertIsNotNone(flash_model.acl_encoder_operation)
        self.assertIsNotNone(tok_embeddings)

    def get_flash_model_inputs(self, device):
        max_seq_len = 6
        input_ids = torch.randint(self.config.vocab_size, (max_seq_len, ))
        position_ids = torch.tensor(range(max_seq_len)).to(device)
        is_prefill = True
        kv_cache = [(torch.zeros([9, 128, 8, 128]), torch.zeros([9, 128, 8, 128]))]
        block_tables = torch.tensor([[0]])
        slots = torch.tensor(range(max_seq_len))
        input_lengths = torch.tensor(max_seq_len)
        lm_head_indices = None
        flash_model_inputs = (input_ids, position_ids, is_prefill, kv_cache, block_tables,
                              slots, input_lengths, max_seq_len, lm_head_indices)
        return flash_model_inputs


if __name__ == '__main__':
    unittest.main()