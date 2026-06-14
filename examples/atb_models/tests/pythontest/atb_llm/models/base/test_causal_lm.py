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
from atb_llm.models.base.config import BaseConfig
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


class TestFlashForCausalLM(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.config = BaseConfig(
            num_attention_heads=8,
            num_key_value_heads=4,
            rope_theta=1,
            num_hidden_layers=6,
            hidden_size=512,
            pe_type="ROPE",
            rope_scaling={"rope_theta": 1.0, "factor": 2.0},
            rms_norm_eps=1e-6,
            max_position_embeddings=1024
        )
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
    def test_init(self, mock_init_so):
        flash_causal_lm = FlashForCausalLM(self.config, self.weights)
        self.assertEqual(flash_causal_lm.num_key_value_heads, 4)
        self.assertEqual(flash_causal_lm.num_attention_heads, 8)
        tensor_size = flash_causal_lm.get_in_tensor_size()
        self.assertEqual(tensor_size, 9)
    
    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    def test_model_prepare_inputs_prefill(self, mock_init_so):
        instance0 = FlashForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        lm_head_indices = torch.tensor([23561]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu()) for \
                           i in range(2)]
        instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=lm_head_indices
        )
    
    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    def test_model_prepare_inputs_decode(self, mock_init_so):
        instance0 = FlashForCausalLM(self.config, self.weights)
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
    def test_update_adapter_weights(self, mock_init_so):
        instance0 = FlashForCausalLM(self.config, self.weights)
        adapter_weights = torch.tensor([1]).npu()
        in_tensor = torch.tensor([3, 3]).npu()
        start_idx = 0
        instance0.update_adapter_weights(adapter_weights, in_tensor, start_idx)
        self.assertIsNotNone(adapter_weights)
        self.assertIsNotNone(in_tensor)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    def test_weight_format_cast(self, mock_init_so):
        instance0 = FlashForCausalLM(self.config, self.weights)
        in_tensor = torch.tensor([23561, 235, 18]).npu()
        instance0.weight_format_cast(in_tensor)
        self.assertIsNotNone(in_tensor)

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    def test_init_kvcache(self, mock_init_so):
        instance0 = FlashForCausalLM(self.config, self.weights)
        instance0.acl_encoder_operation = MagicMock()
        instance0.acl_decoder_operation = MagicMock()
        mock_init_kvcache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu()) for \
                           i in range(2)]
        instance0.init_kvcache(mock_init_kvcache)
        self.assertIsNotNone(instance0.ascend_kcache_id)
        self.assertIsNotNone(instance0.ascend_vcache_id)

if __name__ == '__main__':
    unittest.main()