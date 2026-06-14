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

from atb_llm.models.telechat.flash_causal_telechat import FlashTelechatForCausalLM
from atb_llm.models.telechat.flash_causal_telechat import (
    RMSNorm,
    RMSNormBias,
    RMSNormWrapper,
    FlashTelechatAttention,
    TelechatMLP,
    TelechatBlock,
)
from atb_llm.utils.mapping import Mapping
from atb_llm.models.telechat.config_telechat import TelechatConfig
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


LOAD_ATB_SPEED = "atb_llm.models.base.flash_causal_lm.load_atb_speed"
FLASH_TELECHAT = "atb_llm.models.telechat.flash_causal_telechat"


class TestFlashTelechatForCausalLM(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.config = TelechatConfig(
            n_head=8,
            n_layer=32,
            ffn_hidden_size=512,
            pe_type="ROPE",
            rms_norm_eps=1e-6,
            seq_length=1024,
            auto_map={
                    "AutoConfig": "configuration_telechat2.Telechat2Config",
                    "AutoModelForCausalLM": "modeling_telechat2.Telechat2ForCausalLM"
            }
        )
        self.weights = MagicMock()
        self.weights.sharded = False
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.process_group = MagicMock()
        self.weights.process_group.rank.return_value = 1
        self.weights.process_group.size.return_value = 2
        self.weights.mapping = Mapping(world_size=2, rank=0)
        self.weights.mapping.attn_tp.rank = 1
        self.weights.get_tensor.return_value = torch.empty(100, 100, dtype=torch.float16)
        self.weights.get_multi_weights_col.return_value = torch.empty(100, 100, dtype=torch.float16)
        self.weights.get_weights_col_packed_kv.return_value = torch.empty(100, 100, dtype=torch.float16)
        self.weights.get_multi_weights_row.return_value = torch.empty(100, 100, dtype=torch.float16)
        self.weights.get_whole_tensor.return_value = torch.empty(100, 100, dtype=torch.float16)
        self.weights.get_shape.return_value = [100, 100]

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f'{FLASH_TELECHAT}.FlashTelechatModel', return_value=MagicMock())
    @patch(f'{FLASH_TELECHAT}.load_column_multi')
    def test_init_tie_word_embeddings_false(self, mock_load_column_multi, mock_telechat_model, _):
        _ = FlashTelechatForCausalLM(self.config, self.weights)
        mock_telechat_model.assert_called_once_with(self.config, self.weights)
        mock_load_column_multi.assert_called_once_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True,
        )

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f'{FLASH_TELECHAT}.FlashTelechatModel', return_value=MagicMock())
    @patch(f'{FLASH_TELECHAT}.TensorParallelHead')
    def test_init_w8a8sc(self, mock_tensor_parallel_head, mock_telechat_model, _):
        self.config.quantize = "w8a8sc"
        _ = FlashTelechatForCausalLM(self.config, self.weights)
        mock_telechat_model.assert_called_once_with(self.config, self.weights)
        mock_tensor_parallel_head.load_weight.assert_called_once_with(
            self.config,
            prefix="lm_head",
            weights=self.weights,
            is_norm=True,
        )

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_TELECHAT}.load_column_multi")
    @patch(f"{FLASH_TELECHAT}.WeightWrapper")
    def test_init_ascend_weight(self, mock_weight_wrapper, mock_load_column_multi, mock_init_so):
        mock_weight_wrapper_ins = mock_weight_wrapper.return_value
        mock_weight_wrapper_ins.register_embedding = MagicMock()
        mock_weight_wrapper_ins.register_layer = MagicMock()
        mock_weight_wrapper_ins.register_model_norm = MagicMock()
        mock_weight_wrapper_ins.register_model_lmhead = MagicMock()
        mock_weight_wrapper_ins.weights = []
        mock_weight_wrapper_ins.linear_type = {}
        mock_weight_wrapper_ins.pack_quant_type = {}
        mock_weight_wrapper_ins.linear_transpose_types = {}
        mock_weight_wrapper_ins.linear_descs = []

        ins = FlashTelechatForCausalLM(self.config, self.weights)
        print("ins", ins)
        ins.graph_manager = MagicMock()
        ins.lm_head.linear.trans_flag = True
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()

        ins = FlashTelechatForCausalLM(self.config, self.weights)
        ins.speculate_enable = True
        ins.graph_manager = MagicMock()
        ins.lm_head.linear.trans_flag = True
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()
        ins.graph_manager.register_graph.assert_called()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_TELECHAT}.FlashTelechatModel", return_value=MagicMock())
    @patch(f"{FLASH_TELECHAT}.load_column_multi")
    def test_init_ascend_kvcache(self, mock_load_column_multi, mock_model, mock_init_so):
        ins = FlashTelechatForCausalLM(self.config, self.weights)
        ins.graph_manager = MagicMock()
        cache = torch.zeros([1, 128, 1, 128], dtype=torch.float16).npu()
        ins.init_kvcache([(cache, cache)])
        ins.graph_manager.set_kv_cache.assert_called_once()
        self.assertEqual(ins.ascend_kcache_id, id(cache))

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_TELECHAT}.FlashTelechatModel", return_value=MagicMock())
    @patch(f"{FLASH_TELECHAT}.load_column_multi")
    def test_model_prepare_inputs_prefill(self, mock_load_column_multi, mock_model, mock_init_so):
        ins = FlashTelechatForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 0], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 3
        golden_kv_cache = [(torch.zeros([1, 128, 1, 96]).npu(), torch.zeros([1, 128, 1, 96]).npu()) for \
                           _ in range(2)]
        ins.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(ins.acl_operation_inputs), 12)

    def test_rmsnorm(self):
        RMSNorm("transformer.h.0.input_layernorm", self.weights)
        self.weights.get_tensor.assert_called_once_with("transformer.h.0.input_layernorm.weight")

    def test_rmsnormbias(self):
        RMSNormBias("transformer.h.0.input_layernorm", self.weights)
        self.weights.get_tensor.assert_called_with("transformer.h.0.input_layernorm.bias")

    def test_rmsnormwrapper(self):
        RMSNormWrapper("transformer.h.0.input_layernorm", self.weights)
        self.weights.get_tensor.assert_called_with("transformer.h.0.input_layernorm.module.bias")

    @patch("atb_llm.models.telechat.flash_causal_telechat.load_column_multi")
    def test_flashtelechatattention(self, _):
        FlashTelechatAttention("transformer.h.0.self_attention", self.config, self.weights)

    def test_telechatmlp(self):
        TelechatMLP("transformer.h.0.mlp", self.config, self.weights)

    @patch("atb_llm.models.telechat.flash_causal_telechat.load_column_multi")
    def test_telechatblock(self, _):
        TelechatBlock(0, self.config, self.weights)


if __name__ == '__main__':
    unittest.main()