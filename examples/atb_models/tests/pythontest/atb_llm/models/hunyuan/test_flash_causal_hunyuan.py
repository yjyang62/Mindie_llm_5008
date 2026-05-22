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
from atb_llm.models.hunyuan.flash_causal_hunyuan import FlashHunyuanForCausalLM
from atb_llm.models.hunyuan.config_hunyuan import HunyuanConfig
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


class TestFlashCausalHunyuan(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        config_dict = {
            "cla_share_factor": 2,
            "hidden_size": 6400,
            "intermediate_size": 18304,
            "max_position_embeddings": 131072,
            "model_type": "hunyuan",
            "moe_topk": 1,
            "num_attention_heads": 80,
            "num_experts": 16,
            "num_hidden_layers": 2,
            "num_key_value_heads": 8,
            "num_shared_expert": 1,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "alpha": 1000.0,
                "factor": 1.0,
                "type": "dynamic"
            },
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "torch_dtype": "float16",
            "use_cache": True,
            "use_cla": True,
            "use_qk_norm": True,
            "vocab_size": 129024,
            "quantize": "w8a8",
            "moe_quantize": "w8a8_dynamic",
            "topk_method": "softmax",
        }
        self.config = HunyuanConfig.from_dict(config_dict)
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
    @patch('atb_llm.models.hunyuan.flash_causal_hunyuan.FlashHunyuanModel', return_value=MagicMock())
    @patch('atb_llm.models.hunyuan.flash_causal_hunyuan.load_column_multi')
    def test_init_tie_word_embeddings_false(self, mock_load_column_multi, mock_model, mock_init_so):
        _ = FlashHunyuanForCausalLM(self.config, self.weights)
        mock_model.assert_called_once_with(self.config, self.weights)
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=["model.embed_tokens"],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.hunyuan.flash_causal_hunyuan.FlashHunyuanModel', return_value=MagicMock())
    def test_model_init_weight_wrapper(self, mock_model, mock_init_so):
        instance0 = FlashHunyuanForCausalLM(self.config, self.weights)
        instance0.init_weight_wrapper()

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.hunyuan.flash_causal_hunyuan.FlashHunyuanModel', return_value=MagicMock())
    @patch('atb_llm.models.hunyuan.flash_causal_hunyuan.FlashHunyuanForCausalLM.get_weights')
    def test_model_init_weight(self, mock_get_weights, mock_model, mock_init_so):
        instance0 = FlashHunyuanForCausalLM(self.config, self.weights)
        instance0.init_weight_wrapper()

    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.hunyuan.flash_causal_hunyuan.FlashHunyuanModel', return_value=MagicMock())
    def test_model_prepare_inputs_prefill(self, mock_model, mock_init_so):
        instance0 = FlashHunyuanForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu()) for \
                           i in range(2)]
        acl_encoder_operation_inputs, _ = instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(acl_encoder_operation_inputs), 16)
    
    @patch("atb_llm.models.base.flash_causal_lm.load_atb_speed")
    @patch('atb_llm.models.hunyuan.flash_causal_hunyuan.FlashHunyuanModel', return_value=MagicMock())
    def test_model_prepare_inputs_decode(self, mock_model, mock_init_so):
        instance0 = FlashHunyuanForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 10
        golden_kv_cache = [(torch.tensor([23561, 235, 18]).npu(), torch.tensor([23561, 235, 18]).npu()) for \
                           i in range(2)]
        acl_decoder_operation_inputs, _ = instance0.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, False, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(acl_decoder_operation_inputs), 16)
    

if __name__ == '__main__':
    unittest.main()