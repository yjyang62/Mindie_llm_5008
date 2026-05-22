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
from ddt import ddt, data
import torch

from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.dist import FakeGroup
from atb_llm.models.qwen2.causal_qwen2_edge import Qwen2ForCausalLM
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


@ddt
class TestQwenForCausalLM(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.config = Qwen2Config(
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=6,
            hidden_size=512,
            pe_type="ROPE",
            rms_norm_eps=1e-6,
            vocab_size=1000,
            tie_word_embeddings=False,
            use_qk_norm=True
        )
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.mapping = Mapping(world_size=1, rank=0)
        self.weights.process_group = FakeGroup(0, 1)

        self.mock_init_so = patch('atb_llm.models.base.causal_lm.load_atb_speed').start()
        self.mock_qwen_edge_model = patch('atb_llm.models.qwen2.causal_qwen2_edge.FlashQwenModel').start()
        self.mock_load_column_multi = patch('atb_llm.models.qwen2.causal_qwen2_edge.load_column_multi').start()
        

    def tearDown(self):
        patch.stopall()
        

    @data(True, False)
    def test_init(self, tie_word_embeddings):
        self.config.tie_word_embeddings = tie_word_embeddings
        prefix = "model.embed_tokens" if tie_word_embeddings else "lm_head"
        _ = Qwen2ForCausalLM(self.config, self.weights)
        self.mock_qwen_edge_model.assert_called_once_with(self.config, self.weights)
        self.mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=[prefix],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )

    
    def test_init_ascend_operations(self):
        causal_model = Qwen2ForCausalLM(self.config, self.weights)
        causal_model.init_ascend_operations(self.config)
        self.assertIsNotNone(causal_model.acl_encoder_operation)
        self.assertIsNotNone(causal_model.acl_decoder_operation)

        
    @patch('atb_llm.models.qwen2.causal_qwen2_edge.Qwen2ForCausalLM.get_weights', return_value=MagicMock())
    def test_init_ascend_weight(self, mock_get_weights):
        causal_model = Qwen2ForCausalLM(self.config, self.weights)
        mock_weight_wrapper = mock_get_weights()
        mock_weight_wrapper.weights = list(torch.arange(100))
        mock_weight_wrapper.linear_type = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.pack_quant_type = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.linear_transpose_types = list(range(self.config.num_hidden_layers + 1))
        
        causal_model.lm_head.linear = MagicMock()
        causal_model.lm_head.linear.trans_flag = False
        causal_model.acl_encoder_operation = MagicMock()
        causal_model.acl_decoder_operation = MagicMock()

        causal_model.init_ascend_weight()
        causal_model.acl_encoder_operation.set_param.assert_called_once()
        causal_model.acl_decoder_operation.set_param.assert_called_once()
        causal_model.acl_encoder_operation.set_weight.assert_called_once()
        causal_model.acl_decoder_operation.set_weight.assert_called_once()
        

    @data(True, False)
    def test_prepare_inputs_for_ascend(self, is_prefill):
        causal_model = Qwen2ForCausalLM(self.config, self.weights)
        causal_model_inputs = self.get_model_inputs(causal_model.device, is_prefill)
        causal_model.token_offset = torch.full(
                (1,), 0, dtype=torch.int32, device=causal_model.device
            )
        causal_model.seq_len_encoder = torch.full(
                (1,), 0, dtype=torch.int32, device=causal_model.device
            )
        causal_model.prepare_inputs_for_ascend(*causal_model_inputs)
        self.assertIsNotNone(causal_model.acl_operation_inputs)
    

    def get_model_inputs(self, device, is_prefill):
        max_seq_len = 3
        input_ids = torch.tensor([[23561, 235, 18]]).npu()
        position_ids = torch.tensor([range(max_seq_len)]).to(device)
        kv_cache = [(torch.zeros([9, 128, 8, 128]), torch.zeros([9, 128, 8, 128]))] if not is_prefill else None
        model_inputs = (input_ids, position_ids, is_prefill, max_seq_len, kv_cache)
        return model_inputs


    @patch('atb_llm.models.qwen2.causal_qwen2_edge.CausalLMOutputWithPast')
    def test_forward(self, mock_causallm_output_with_past):
        causal_model = Qwen2ForCausalLM(self.config, self.weights)
        causal_model.prepare_inputs_for_ascend = MagicMock(return_value=([], {}))
        causal_model.init_ascend_weight = MagicMock()
        input_ids, position_ids, _, _, _ = self.get_model_inputs(causal_model.device, True)
        attention_mask = torch.ones_like(input_ids)
        causal_model.forward(input_ids, attention_mask, position_ids)
        causal_model.init_ascend_weight.assert_called_once()
        causal_model.prepare_inputs_for_ascend.assert_called_once()
        mock_causallm_output_with_past.assert_called_once()


if __name__ == '__main__':
    unittest.main()