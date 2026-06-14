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
import math
import torch

from atb_llm.utils.mapping import Mapping
from atb_llm.utils.dist import FakeGroup
from atb_llm.models.deepseekv2.flash_causal_deepseekv2 import FlashDeepseekv2ForCausalLM
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


class TestFlashCausalDeepseekV2(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_torch_classes = None

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
            "type": "yarn",
            "parallel_embedding": True
            }
        }
        self.config = DeepseekV2Config.from_dict(config_dict)
        self.config.routed_scaling_factor = 2
        self.config.n_group = 1
        self.config.topk_group = 1
        self.config.topk_method = None
        self.config.parallel_embedding = True
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

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.load_column_multi')
    def test_init_tie_word_embeddings_false(self, mock_load_column_multi, mock_model, mock_init_so):
        self.weights.sharded = False
        _ = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        mock_model.assert_called_once_with(self.config, self.weights, llm_config=None,
                                            init_expert_table=None, mix_shared_routing=False)
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_model_init_weight_wrapper(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        instance0.init_weight_wrapper()

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.get_weights')
    def test_model_init_weight(self, mock_get_weights, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        instance0.init_weight_wrapper()

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_model_prepare_inputs_prefill(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
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
    
    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_model_prepare_inputs_decode(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
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
    
    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_model_prepare_inputs_ep(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        instance0.expert_parallel_degree = 2
        instance0.mapping.attn_tp.group_size = 1
        instance0.config.num_experts_per_tok = 1
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

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_model_process_logits(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        self.assertEqual(instance0.get_process_logits_type(), 'scaling')
        instance0.routed_scaling_factor = 2.5
        instance0.norm_topk_prob = True
        self.assertEqual(instance0.get_process_logits_type(), "normScaling")
        instance0.routed_scaling_factor = 0
        self.assertEqual(instance0.get_process_logits_type(), "none")

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_model_routing_method(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        self.assertEqual(instance0.get_routing_method_type(), "softMaxTopK")
        instance0.topk_method = "noaux_tc"
        self.assertEqual(instance0.get_routing_method_type(), "noAuxTc")
        instance0.topk_method = "group_limited_greedy"
        self.assertEqual(instance0.get_routing_method_type(), "deviceLimited")

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_all2all_buffer_factor(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        length = 4096
        golden_buffer_factor = 1.0
        self.assertGreaterEqual(instance0.get_all2all_buffer_factor(length), golden_buffer_factor)
    
    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.init_ascend_weight')
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekv2ForCausalLM.get_weights')
    def test_init_ascend_weight(self, mock_get_weights, mock_init_ascend_weight, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        instance0.init_ascend_weight()

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_dap_forward(self, mock_model, mock_init_so):
        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        instance0.num_speculative_tokens = 0

        instance0.init_kvcache = MagicMock()
        instance0.init_ascend_weight = MagicMock()
        instance0.prepare_inputs_for_ascend = MagicMock(return_value=([], "{}", 0))
        instance0.acl_dap_operation = MagicMock()
        instance0.acl_dap_operation.execute = MagicMock(return_value=(torch.empty([1]), torch.empty([1])))

        instance0.execute_ascend_operator = MagicMock(return_value=(torch.empty([1]), torch.empty([1])))

        golden_input_ids = [torch.tensor([23561, 235, 18]).npu(), torch.tensor([17320, 35, 124]).npu()]
        golden_position_ids = [
            torch.tensor([0, 1, 2], dtype=torch.int32).npu(),
            torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        ]
        golden_kv_cache = [(torch.tensor([0]), torch.tensor([1])), (torch.tensor([2]), torch.tensor([3]))]
        golden_block_tables = [torch.tensor([1, 2]).npu(), torch.tensor([3, 4]).npu()]
        golden_slots = [torch.tensor([0, 1, 2]).npu(), torch.tensor([0, 1, 2]).npu()]
        golden_input_lengths = [torch.tensor([2, 1]).npu(), torch.tensor([1, 2]).npu()]
        golden_max_seq_len = [10, 20]

        instance0.dap_forward(
            golden_input_ids, golden_position_ids, [True, True], golden_kv_cache,
            golden_block_tables, golden_slots, golden_input_lengths, golden_max_seq_len, [None, None],
            dap_kwargs=[{}, {}]
        )

        instance0.init_ascend_weight.assert_called_once()
        instance0.init_kvcache.assert_called_once()
        self.assertEqual(instance0.prepare_inputs_for_ascend.call_count, 2)

        first_called_args, second_called_args = instance0.prepare_inputs_for_ascend.call_args_list
        self.assertTrue(torch.equal(first_called_args[0][0], golden_input_ids[0]))
        self.assertTrue(torch.equal(second_called_args[0][0], golden_input_ids[1]))
        self.assertTrue(torch.equal(first_called_args[0][1], golden_position_ids[0]))
        self.assertTrue(torch.equal(second_called_args[0][1], golden_position_ids[1]))
        self.assertTrue(first_called_args[0][2])
        self.assertTrue(second_called_args[0][2])
        self.assertTrue(torch.equal(first_called_args[0][4], golden_block_tables[0]))
        self.assertTrue(torch.equal(second_called_args[0][4], golden_block_tables[1]))
        self.assertTrue(torch.equal(first_called_args[0][5], golden_slots[0]))
        self.assertTrue(torch.equal(second_called_args[0][5], golden_slots[1]))
        self.assertTrue(torch.equal(first_called_args[0][6], golden_input_lengths[0]))
        self.assertTrue(torch.equal(second_called_args[0][6], golden_input_lengths[1]))
        self.assertEqual(first_called_args[0][7], golden_max_seq_len[0])
        self.assertEqual(second_called_args[0][7], golden_max_seq_len[1])

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_calc_moe_buffer_size(self, mock_model, mock_init_so):
        world_size = 16
        total_batch_size = 4096
        hidden_size = 7168
        num_experts = 256
        num_redundant_experts = 64
        moe_ep_size = 16
        moe_tp_size = 1
        golden_moe_ep_buffer_size = math.ceil(
            math.ceil(total_batch_size / world_size) * hidden_size * (num_experts + num_redundant_experts)
            * 4 / (1024 ** 2)
        ) + 1
        golden_moe_tp_buffer_size = math.ceil(
            math.ceil(total_batch_size / world_size) * hidden_size * moe_tp_size * 4 / (1024 ** 2)) + 1

        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        instance0.total_batch_size = total_batch_size
        instance0.hidden_size = hidden_size
        instance0.num_of_experts = num_experts
        instance0.num_redundant_experts = num_redundant_experts
        instance0.mapping.world_size = world_size
        instance0.mapping.moe_ep.group_size = moe_ep_size
        instance0.mapping.moe_tp.group_size = moe_tp_size
        moe_ep_buffer_size, moe_tp_buffer_size = instance0.calc_moe_buffer_size()

        self.assertEqual(moe_ep_buffer_size, golden_moe_ep_buffer_size)
        self.assertEqual(moe_tp_buffer_size, golden_moe_tp_buffer_size)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.deepseekv2.flash_causal_deepseekv2.FlashDeepseekV2Model', return_value=MagicMock())
    def test_get_all2all_buffer_factor(self, mock_model, mock_init_so):
        world_size = 16
        moe_ep_size = 16
        moe_tp_size = 1
        alltoall_ep_buffer_scale_factors = [[0, 3]]

        instance0 = FlashDeepseekv2ForCausalLM(self.config, self.weights)
        instance0.mapping.world_size = world_size
        instance0.mapping.moe_ep.group_size = moe_ep_size
        instance0.mapping.moe_tp.group_size = moe_tp_size
        num_tokens_per_rank = 128000
        alltoall_buff_scale = instance0.get_all2all_buffer_factor(num_tokens_per_rank, is_prefill=True)
        max_alltoall_buff_scale = 3
        self.assertEqual(alltoall_buff_scale, max_alltoall_buff_scale)
        num_tokens_per_rank = 1
        alltoall_buff_scale = instance0.get_all2all_buffer_factor(1, is_prefill=True)
        golden_alltoall_buff_scale = moe_ep_size * 2
        self.assertEqual(alltoall_buff_scale, golden_alltoall_buff_scale)
        num_tokens_per_rank = 128
        alltoall_buff_scale = instance0.get_all2all_buffer_factor(num_tokens_per_rank, is_prefill=True)
        max_scale = math.sqrt(moe_ep_size)
        min_scale = max(1, max_scale / 2)
        golden_alltoall_buff_scale = min_scale + (max_scale - min_scale) / (
            1 + math.exp(math.log2(num_tokens_per_rank) - math.log2(moe_ep_size))
        )
        self.assertEqual(alltoall_buff_scale, golden_alltoall_buff_scale)
        instance0.ds_config = MagicMock()
        instance0.ds_config.alltoall_ep_buffer_scale_factors = alltoall_ep_buffer_scale_factors
        num_tokens_per_rank = 4096
        alltoall_buff_scale = instance0.get_all2all_buffer_factor(num_tokens_per_rank)
        golden_alltoall_buff_scale = alltoall_ep_buffer_scale_factors[0][1]
        self.assertEqual(alltoall_buff_scale, golden_alltoall_buff_scale)


if __name__ == '__main__':
    unittest.main()