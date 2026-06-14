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

from atb_llm.utils.mapping import Mapping
from atb_llm.utils.dist import FakeGroup
from atb_llm.utils.env import ENV
from atb_llm.models.deepseekv2.flash_causal_deepseekv2 import FlashDeepseekv2ForCausalLM
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


class TestFlashCausalDeepseekV2withMTP(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
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
            "num_hidden_layers": 28,
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
        self.config = DeepseekV2Config.from_dict(config_dict)
        self.config.routed_scaling_factor = 2
        self.config.n_group = 1
        self.config.topk_group = 1
        self.config.topk_method = None
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

        self.weights.process_group = FakeGroup(0, 1)
        self.weights.mapping = Mapping(world_size=2, rank=0)
        self.weights.mapping.attn_tp.rank = 1
        self.num_speculative_tokens = 3

        self.mock_load_atb_speed = patch("atb_llm.models.base.flash_causal_lm.load_atb_speed").start()
        self.mock_model = patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model',
                                return_value=MagicMock()).start()
        self.mock_load_column_multi = \
            patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.load_column_multi').start()
    
    def tearDown(self):
        patch.stopall()

    def test_init_tie_word_embeddings_false_num_speculative_tokens_positive(self):
        ENV.deepseek_mtp = 3
        self.weights.sharded = False
        _ = FlashDeepseekv2ForCausalLM(self.config, self.weights, 
                                              num_speculative_tokens=self.num_speculative_tokens)
        self.mock_model.assert_called_once_with(self.config, self.weights, llm_config=None,
                                                init_expert_table=None, mix_shared_routing=False)
        self.mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )
        self.assertEqual(self.config.num_speculative_tokens, self.num_speculative_tokens)

    @patch('torch.classes.ModelTorch.ModelTorch')
    def test_init_ascend_operations(self, mock_model_torch):
        instance = FlashDeepseekv2ForCausalLM(self.config, self.weights,
                                              num_speculative_tokens=self.num_speculative_tokens)
        instance.num_speculative_tokens = 3
        instance.init_ascend_operations(self.config)
        mock_model_torch.assert_called_with("deepseekV2_MtpDecoderModel")

    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.get_weights')
    def test_init_ascend_weight_mtp(self, mock_get_weights):
        self.weights.sharded = False
        instance = FlashDeepseekv2ForCausalLM(self.config, self.weights,
                                              num_speculative_tokens=self.num_speculative_tokens)
        instance.num_speculative_tokens = 3
        mock_weight_wrapper = MagicMock()
        mock_get_weights.return_value = mock_weight_wrapper
        mock_weight_wrapper.weights = list(torch.arange(100))
        mock_weight_wrapper.pack_quant_type = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.attn_linear_types = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.mlp_linear_types = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.moe_linear_types = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.attn_linear_transpose_types = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.mlp_linear_transpose_types = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.moe_linear_transpose_types = list(range(self.config.num_hidden_layers + 1))
        mock_weight_wrapper.moe_pack_type = list(range(self.config.num_hidden_layers + 1))

        instance.lm_head.linear = MagicMock()
        instance.lm_head.linear.trans_flag = False
        
        instance.acl_encoder_operation_mtp = MagicMock()
        instance.acl_decoder_operation_mtp = MagicMock()
        instance.kvcache_quant_layers = []
        instance.init_ascend_weight()

    
        instance.acl_encoder_operation_mtp.set_param.assert_called_once()
        instance.acl_decoder_operation_mtp.set_param.assert_called_once()
        instance.acl_encoder_operation_mtp.set_weight.assert_called_once()
        instance.acl_decoder_operation_mtp.set_weight.assert_called_once()


    def test_prepare_mtp_roll_inputs_for_ascend(self):
        
        acl_param_mtp = {}
        q_lens = torch.tensor([3], dtype=torch.int32)
        logits_mtp = torch.tensor([0.1, 0.9])
        hidden_states_mtp = torch.randn(1, 2)
        acl_inputs_mtp = [torch.tensor([1, 2, 3]), torch.tensor([0])] + [None] * 15
        
        instance = FlashDeepseekv2ForCausalLM(self.config, self.weights,
                                              num_speculative_tokens=self.num_speculative_tokens)
        updated_inputs, updated_param = instance.prepare_mtp_roll_inputs_for_ascend(
            acl_inputs_mtp, acl_param_mtp, q_lens, logits_mtp, hidden_states_mtp, True)
        self.assertTrue(torch.equal(updated_inputs[0], torch.tensor([2, 3, 1])))
        self.assertTrue(torch.equal(updated_inputs[1], torch.tensor([1])))
        self.assertTrue(torch.equal(updated_inputs[17], hidden_states_mtp))
        self.assertEqual(len(updated_inputs), 18)
        self.assertEqual(updated_param, acl_param_mtp)

    def test_prepare_repeated_batch(self):
        instance = FlashDeepseekv2ForCausalLM(self.config, self.weights,
                                              num_speculative_tokens=self.num_speculative_tokens)
        q_lens = torch.tensor([3], dtype=torch.int32).npu()
        block_tables = block_tables = torch.tensor([1, 2]).npu()
        input_lengths = torch.tensor([3], dtype=torch.int32).npu()
        acl_inputs = [None for _ in range(18)]
        acl_inputs[5] = block_tables
        acl_inputs[11] = input_lengths
        acl_param = None
        updated_inputs, updated_param = instance.prepare_repeated_batch(acl_inputs, acl_param, q_lens)
        self.assertTrue(torch.equal(updated_inputs[5], torch.tensor([1, 1, 1, 2, 2, 2]).npu()))
        self.assertTrue(torch.equal(updated_inputs[11], torch.tensor([1, 2, 3], dtype=torch.int32).npu()))
        self.assertNotEqual(acl_param, updated_param)

    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.'
           'prepare_mtp_roll_inputs_for_ascend')
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.get_adapter_ids')
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.init_ascend_weight')
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.init_kvcache')
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.prepare_inputs_for_ascend')
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.execute_ascend_operator')
    def test_forward_mtp_no_dp(self, mock_execute_ascend_operator, mock_prepare_inputs_for_ascend, mock_init_kvcache,
                         mock_init_ascend_weights, mock_get_adapter_ids, mock_prepare_mtp_roll):
        instance = FlashDeepseekv2ForCausalLM(self.config, self.weights,
                                              num_speculative_tokens=self.num_speculative_tokens)
        input_ids = torch.tensor([23561, 235, 18]).npu()
        position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        is_prefill = True
        kv_cache = [(torch.tensor([0]), torch.tensor([1])), (torch.tensor([2]), torch.tensor([3]))]
        block_tables = torch.tensor([1, 2]).npu()
        slots = torch.tensor([0, 1, 2]).npu()
        input_lengths = torch.tensor([2, 1]).npu()
        max_seq_len = 10
        lm_head_indicies = torch.tensor([1], dtype=torch.int32).npu()
        acl_inputs = [0] * 22  # 22: shard_effective_token_indices
        mock_prepare_inputs_for_ascend.return_value = (acl_inputs, None, 0)
        mock_execute_ascend_operator.return_value = ([1, 2, 3], [1, 2, 3])
        mock_prepare_mtp_roll.return_value = (None, None)
        instance.mtp_k_caches = torch.arange(10, dtype=torch.int32).npu()
        instance.mtp_v_caches = torch.arange(10, dtype=torch.int32).npu()
        instance.forward(input_ids, position_ids, is_prefill, kv_cache, block_tables, slots, input_lengths,
                         max_seq_len, lm_head_indicies)

    @patch('torch.classes.ModelTorch.ModelTorch.set_kv_cache')
    def test_init_kvcache(self, mock_set_kv_cache):
        instance = FlashDeepseekv2ForCausalLM(self.config, self.weights,
                                              num_speculative_tokens=self.num_speculative_tokens)
        instance.acl_dap_operation = None
        kv_cache = [(torch.tensor([0]), torch.tensor([1])), (torch.tensor([2]), torch.tensor([3]))]

        instance.ascend_kcache_id = None
        instance.ascend_vcache_id = None
        instance.ascend_kcache_shape = None
        instance.ascend_vcache_shape = None
        instance.soc_info.need_nz = False
        instance.init_kvcache(kv_cache)
      
if __name__ == '__main__':
    unittest.main()