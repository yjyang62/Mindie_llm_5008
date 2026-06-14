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
from atb_llm.models.ernie_moe.flash_causal_ernie_moe import FlashErniemoeForCausalLM
from atb_llm.models.ernie_moe.config_ernie_moe import ErniemoeConfig
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


class TestFlashCausalErniemoe(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        config_dict = {
            "model_type": "ernie_moe",
            "hidden_act": "silu",
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "max_position_embeddings": 131072,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_hidden_layers": 54,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "rms_norm_eps": 1e-05,
            "use_cache": False,
            "vocab_size": 103424,
            "rope_theta": 500000.0,
            "moe_num_experts": 64,
            "moe_num_shared_experts": 0,
            "moe_layer_start_index": 3,
            "moe_layer_end_index": 53,
            "moe_intermediate_size": 3584,
            "moe_gate": "topk_fused",
            "moe_k": 8,
            "moe_layer_interval": 1,
            "tie_word_embeddings": True,
            "quantize": "w8a8_dynamic"
        }
        self.config = ErniemoeConfig.from_dict(config_dict)
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
        self.weights.mapping = Mapping(world_size=2, rank=0, dp=2, tp=1, moe_tp=1, moe_ep=2)
        self.weights.mapping.attn_tp.rank = 1
        self.llm_config = MagicMock()

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.load_column_multi')
    def test_init_tie_word_embeddings(self, mock_load_column_multi, mock_model, mock_init_so):
        _ = FlashErniemoeForCausalLM(self.config, self.weights)
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=["model.embed_tokens"],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_attn_quantize(self, mock_model, mock_init_so):
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        golden_attn_quantize = "w8a8"
        self.assertEqual(instance0.attn_quantize, golden_attn_quantize)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_ep_level(self, mock_model, mock_init_so):
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        self.assertEqual(instance0.ep_level, 1)
        self.llm_config.llm.ep_level = 2
        self.llm_config.llm.communication_backend.prefill = "lccl"
        self.llm_config.llm.communication_backend.decode = "lccl"
        instance1 = FlashErniemoeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        self.assertEqual(instance1.ep_level, 2)
        self.assertEqual(instance1.dep_communication_backend["prefill"], "lccl")
        self.assertEqual(instance1.dep_communication_backend["decode"], "lccl")

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_init_position_rotary_embedding(self, mock_model, mock_init_so):
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_max_seq_len = 20
        instance0.init_position_rotary_embedding(golden_position_ids, golden_max_seq_len)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_model_init_weight(self, mock_get_weights, mock_model):
        self.config.num_hidden_layers = 0
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        instance0.get_weights()

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_invalid_parallel_param(self, mock_model, mock_init_so):
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        ENV.enable_dp_move_up = 0
        dep_inputs_default = [None] * 9
        dep_inputs_default[-2] = torch.empty((8, 1))
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu())
                           for i in range(2)]
        token_size_per_dp_group = torch.tensor([6, 5]).npu()
        with self.assertRaises(RuntimeError):
            instance0.prepare_inputs_for_ascend(
                golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
                golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None,
                token_size_per_dp_group=token_size_per_dp_group, dep_inputs=dep_inputs_default
            )
        self.weights.mapping = Mapping(world_size=2, rank=0, dp=1, tp=2, moe_tp=1, moe_ep=2)
        self.llm_config.llm.ep_level = 2
        instance1 = FlashErniemoeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        ENV.enable_dp_move_up = 0
        dep_inputs_default = [None] * 9
        dep_inputs_default[-2] = torch.empty((8, 1))
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu())
                           for i in range(2)]
        token_size_per_dp_group = torch.tensor([6, 5]).npu()
        with self.assertRaises(NotImplementedError):
            instance1.prepare_inputs_for_ascend(
                golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
                golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None,
                token_size_per_dp_group=token_size_per_dp_group, dep_inputs=dep_inputs_default
            )


    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_model_prepare_inputs_prefill(self, mock_model, mock_init_so):
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        ENV.enable_dp_move_up = 1
        dep_inputs_default = [None] * 9
        dep_inputs_default[-2] = torch.empty((8, 1))
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu())
                           for i in range(2)]
        token_size_per_dp_group = torch.tensor([6, 5]).npu()
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None,
            token_size_per_dp_group=token_size_per_dp_group, dep_inputs=dep_inputs_default
        )
        self.assertEqual(len(instance0.acl_operation_inputs), 31)
        self.assertEqual(len(instance0.acl_param), 30)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_model_prepare_inputs_decode(self, mock_model, mock_init_so):
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        ENV.enable_dp_move_up = 1
        dep_inputs_default = [None] * 9
        dep_inputs_default[-2] = torch.empty((8, 1))
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu())
                           for i in range(2)]
        token_size_per_dp_group = torch.tensor([6, 5]).npu()
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, False, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None,
            token_size_per_dp_group=token_size_per_dp_group, dep_inputs=dep_inputs_default
        )
        self.assertEqual(len(instance0.acl_operation_inputs), 31)
        self.assertEqual(len(instance0.acl_param), 30)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_init_kvcache(self, mock_model, mock_init_so): 
        mock_init_kvcache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu())
                             for _ in range(2)]
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        instance0.init_kvcache(mock_init_kvcache)
        self.assertIsNotNone(instance0.ascend_kcache_id)
        self.assertIsNotNone(instance0.ascend_vcache_id)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_model_prepare_inputs_prefill1(self, mock_model, mock_init_so):
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
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
            (torch.tensor([23561, 235, 18], dtype=torch.float16).npu(),
             torch.tensor([23561, 235, 18], dtype=torch.float16).npu()) for _ in range(2)
        ]
        token_size_per_dp_group = torch.tensor([6, 5]).npu()
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None,
            token_size_per_dp_group=token_size_per_dp_group, dep_inputs=dep_inputs_default
        )
        self.assertEqual(len(instance0.acl_operation_inputs), 31)
        self.assertEqual(len(instance0.acl_param), 30)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_model_prepare_inputs_decode1(self, mock_model, mock_init_so):
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
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
            (torch.tensor([23561, 235, 18], dtype=torch.float16).npu(),
             torch.tensor([23561, 235, 18], dtype=torch.float16).npu()) for _ in range(2)
        ]
        token_size_per_dp_group = torch.tensor([6, 5]).npu()
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, False, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None,
            token_size_per_dp_group=token_size_per_dp_group, dep_inputs=dep_inputs_default
        )
        self.assertEqual(len(instance0.acl_operation_inputs), 31)
        self.assertEqual(len(instance0.acl_param), 30)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_init_ascend_weight(self, mock_model, mock_init_so):
        self.config.num_hidden_layers = 0
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        instance0.init_ascend_weight()

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.ernie_moe.flash_causal_ernie_moe.FlashErniemoeModel', return_value=MagicMock())
    def test_select_logits(self, mock_model, mock_init_so):
        logits = [[1, 2, 3, 4], [5, 6, 7, 8]]
        instance0 = FlashErniemoeForCausalLM(self.config, self.weights)
        selected_logits0 = instance0.select_logits(logits)
        self.assertEqual(selected_logits0, logits)
        golden_selected_logits = [logits[0]]
        selected_logits1 = instance0.select_logits(logits, dp_logits_num=[1, 1])
        self.assertEqual(selected_logits1, golden_selected_logits)


if __name__ == '__main__':
    unittest.main()