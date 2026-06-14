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

from atb_llm.utils.env import ENV
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.dist import FakeGroup
from atb_llm.models.deepseek.flash_causal_deepseek import FlashDeepseekForCausalLM
from atb_llm.models.deepseek.config_deepseek import DeepseekConfig
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses



class TestFlashCausalDeepseek(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        config_dict = {
            "model_type": "deepseek",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "aux_loss_alpha": 0.001,
            "bos_token_id": 100000,
            "eos_token_id": 100001,
            "first_k_dense_replace": 1,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 10944,
            "max_position_embeddings": 4096,
            "moe_intermediate_size": 1408,
            "moe_layer_freq": 1,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "norm_topk_prob": False,
            "num_attention_heads": 16,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 28,
            "num_key_value_heads": 16,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "scoring_func": "softmax",
            "seq_aux": True,
            "tie_word_embedding": False,
            "use_cache": True,
            "vocab_size": 102400
        }
        self.config = DeepseekConfig.from_dict(config_dict)
        self.setconfig()

    def setconfig(self):
        self.config.routed_scaling_factor = 2
        self.config.n_group = 1
        self.config.topk_group = 1
        self.config.topk_method = None
        self.config.parallel_embedding = True
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.bfloat16
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

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.load_column_multi')
    def test_init_tie_word_embeddings_false(self, mock_load_column_multi, mock_model, mock_init_so):
        _ = FlashDeepseekForCausalLM(self.config, self.weights)
        mock_model.assert_called_once_with(self.config, self.weights)
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    def test_init_position_rotary_embedding(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekForCausalLM(self.config, self.weights)
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_max_seq_len = 20
        instance0.init_position_rotary_embedding(golden_position_ids, golden_max_seq_len)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    def test_model_init_weight(self, mock_get_weights, mock_model):
        self.config.num_hidden_layers = 0
        instance0 = FlashDeepseekForCausalLM(self.config, self.weights)
        instance0.get_weights()


    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    def test_model_prepare_inputs_prefill(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu()) for \
                           i in range(2)]
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(instance0.acl_operation_inputs), 27)
        self.assertEqual(len(instance0.acl_param), 30)
    
    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    def test_model_prepare_inputs_decode(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu()) for \
                           i in range(2)]
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, False, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(instance0.acl_operation_inputs), 27)
        self.assertEqual(len(instance0.acl_param), 30)
    

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    def test_init_kvcache(self, mock_model, mock_init_so): 
        mock_init_kvcache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu()) for \
                           i in range(2)]
        instance0 = FlashDeepseekForCausalLM(self.config, self.weights)
        instance0.init_kvcache(mock_init_kvcache)
        self.assertIsNotNone(instance0.ascend_kcache_id)
        self.assertIsNotNone(instance0.ascend_vcache_id)


    
    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    def test_model_prepare_inputs_prefill1(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekForCausalLM(self.config, self.weights)
        ENV.enable_dp_move_up = 1
        dep_inputs_default = [None] * 9
        dep_inputs_default[-2] = torch.empty((8, 1))
        instance0.mapping.attn_dp.group_size = 2
        instance0.mapping.mlp_tp.group_size = 2
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [
            (torch.tensor([23561, 235, 18], dtype=torch.float16).npu(), \
            torch.tensor([23561, 235, 18], dtype=torch.float16).npu()) for \
                           i in range(2)]
        token_size_per_dp_group = torch.tensor([6, 5]).npu()
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None,
            token_size_per_dp_group=token_size_per_dp_group, dep_inputs=dep_inputs_default
        )
        self.assertEqual(len(instance0.acl_operation_inputs), 31)
        self.assertEqual(len(instance0.acl_param), 30)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    def test_model_prepare_inputs_decode1(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekForCausalLM(self.config, self.weights)
        ENV.enable_dp_move_up = 1
        dep_inputs_default = [None] * 9
        dep_inputs_default[-2] = torch.empty((8, 1))
        instance0.mapping.attn_dp.group_size = 2
        instance0.mapping.mlp_tp.group_size = 2
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18], dtype=torch.float16).npu(), \
        torch.tensor([23561, 235, 18], dtype=torch.float16).npu()) for \
                           i in range(2)]
        token_size_per_dp_group = torch.tensor([6, 5]).npu()
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, False, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None,
            token_size_per_dp_group=token_size_per_dp_group, dep_inputs=dep_inputs_default
        )
        self.assertEqual(len(instance0.acl_operation_inputs), 31)
        self.assertEqual(len(instance0.acl_param), 30)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekModel', return_value=MagicMock())
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekForCausalLM.init_ascend_weight')
    @patch('atb_llm.models.deepseek.flash_causal_deepseek.FlashDeepseekForCausalLM.get_weights')
    def test_init_ascend_weight(self, mock_get_weights, mock_init_ascend_weight, mock_model, mock_init_so):
        instance0 = FlashDeepseekForCausalLM(self.config, self.weights)
        instance0.init_ascend_weight()


if __name__ == '__main__':
    unittest.main()
