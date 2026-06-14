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

from atb_llm.models.base.config import RopeScaling
from atb_llm.models.llama.causal_llama_edge import LlamaForCausalLM
from atb_llm.models.llama.config_llama import LlamaConfig
from atb_llm.utils.dist import FakeGroup
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.op_backend import OpBackend
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


@ddt
class TestLlamaForCausalLM(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.config = LlamaConfig(
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=6,
            hidden_size=512,
            pe_type="ROPE",
            rms_norm_eps=1e-6,
            vocab_size=1000,
        )
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.mapping = Mapping(world_size=1, rank=0)
        self.weights.process_group = FakeGroup(0, 1)

        self.mock_init_so = patch('atb_llm.models.base.causal_lm.load_atb_speed').start()
        self.mock_llama_edge_model = patch('atb_llm.models.llama.causal_llama_edge.FlashLlamaModel').start()
        self.mock_load_column_multi = patch('atb_llm.models.llama.causal_llama_edge.load_column_multi').start()
    

    def tearDown(self):
        patch.stopall()
        

    def test_init(self):
        _ = LlamaForCausalLM(self.config, self.weights)
        self.mock_llama_edge_model.assert_called_once_with(
            self.config,
            self.weights,
            attn_decode_backend=OpBackend.ATB,
            model_prefix="model")
        self.mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )


    def test_init_position_rotary_embedding(self):
        causal_model = LlamaForCausalLM(self.config, self.weights)
        max_seq_len = 1024
        position_ids = torch.tensor(range(max_seq_len)).to(causal_model.device)
        causal_model.init_position_rotary_embedding(position_ids, max_seq_len)
        self.assertIsNotNone(causal_model.cos_embed)
        self.assertIsNotNone(causal_model.sin_embed)
    
    
    @data(None, "linear", "llama3")
    def test_init_positional_embedding_with_rope_scaling(self, scaling_type):
        rope_scaling_config = {"rope_type": scaling_type, "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 1024}
        self.config.update({"rope_scaling": RopeScaling(**rope_scaling_config)})
        causal_model = LlamaForCausalLM(self.config, self.weights)
        max_seq_len = 1024
        position_ids = torch.tensor(range(max_seq_len)).to(causal_model.device)
        causal_model.init_position_rotary_embedding(position_ids, max_seq_len)
        self.assertIsNotNone(causal_model.cos_embed)
        self.assertIsNotNone(causal_model.sin_embed)


    def test_init_ascend_operations(self):
        causal_model = LlamaForCausalLM(self.config, self.weights)
        causal_model.init_ascend_operations(self.config)
        self.assertIsNotNone(causal_model.acl_encoder_operation)
        self.assertIsNotNone(causal_model.acl_decoder_operation)
    

    @patch('atb_llm.models.llama.causal_llama_edge.LlamaForCausalLM.get_weights', return_value=MagicMock())
    def test_init_ascend_weight(self, mock_get_weights):
        causal_model = LlamaForCausalLM(self.config, self.weights)
        mock_weight_wrapper = mock_get_weights()
        mock_weight_wrapper.weights = list(torch.arange(100))
        mock_weight_wrapper.linear_type = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.pack_quant_type = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.linear_transpose_types = list(range(self.config.num_hidden_layers + 1))
        
        causal_model.lm_head.linear = MagicMock()
        causal_model.lm_head.linear.trans_flag = False
        causal_model.acl_encoder_operation = MagicMock()
        causal_model.acl_decoder_operation = MagicMock()
        causal_model.model.parallel_embedding = False

        causal_model.init_ascend_weight()
        causal_model.acl_encoder_operation.set_param.assert_called_once()
        causal_model.acl_decoder_operation.set_param.assert_called_once()
        causal_model.acl_encoder_operation.set_weight.assert_called_once()
        causal_model.acl_decoder_operation.set_weight.assert_called_once()
    

    @data(True, False)
    def test_prepare_inputs_for_ascend(self, is_prefill):
        causal_model = LlamaForCausalLM(self.config, self.weights)
        causal_model_inputs = self.get_model_inputs(causal_model.device, is_prefill)
        causal_model.token_offset = torch.full(
                (1,), 0, dtype=torch.int32, device=causal_model.device
            )
        causal_model.prepare_inputs_for_ascend(*causal_model_inputs)
        self.assertIsNotNone(causal_model.acl_encoder_operation_inputs)
    

    def get_model_inputs(self, device, is_prefill):
        max_seq_len = 3
        input_ids = torch.tensor([[23561, 235, 18]]).npu()
        position_ids = torch.tensor([range(max_seq_len)]).to(device)
        kv_cache = [(torch.zeros([9, 128, 8, 128]), torch.zeros([9, 128, 8, 128]))] if not is_prefill else None
        model_inputs = (input_ids, position_ids, is_prefill, max_seq_len, kv_cache)
        return model_inputs
    

    @patch('atb_llm.models.llama.causal_llama_edge.CausalLMOutputWithPast')
    def test_forward(self, mock_causallm_output_with_past):
        causal_model = LlamaForCausalLM(self.config, self.weights)
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