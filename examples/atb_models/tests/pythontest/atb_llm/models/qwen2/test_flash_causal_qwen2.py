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

from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.models.qwen2.flash_causal_qwen2 import FlashQwen2ForCausalLM
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.op_backend import OpBackend
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.adapter_manager import AdapterManager
from atb_llm.utils.data.layer_adapter import ParallelLMHead

from mindie_llm.runtime.utils.distributed import set_parallel_info_manager
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


LOAD_ATB_SPEED = "atb_llm.models.base.flash_causal_lm.load_atb_speed"
FLASH_QWEN2 = "atb_llm.models.qwen2.flash_causal_qwen2"


class TestFlashQwen2ForCausalLM(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_classes = MockTorchClasses()
        torch.classes = self.torch_classes

        self.config = Qwen2Config(
            model_type="qwen2",
            hidden_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_hidden_layers=28,
            rms_norm_eps=1e-6,
            torch_dtype=torch.float16,
            vocab_size=125696,
            tie_word_embeddings=False,
        )

        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.mapping = Mapping(world_size=2, rank=0)
        self.weights.mapping.attn_tp.rank = 1
        self.weights.process_group = MagicMock()
        self.weights.process_group.rank.return_value = 0
        self.weights.process_group.size.return_value = 2
        self.weights.quant_desc = None

        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.process_group = None

        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.world_size = 2
        self.mock_parallel_info_manager.word_embed_tp = self.mock_parallel_info
        self.mock_parallel_info_manager.attn_tp = self.mock_parallel_info
        self.mock_parallel_info_manager.lm_head_tp = self.mock_parallel_info

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.Qwen2Model", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.TensorHead")
    def test_init(
            self,
            mock_tensor_head,
            mock_load_column_multi,
            mock_new_qwen_model,
            mock_qwen_model,
            _mock_init_so
    ) -> None:
        FlashQwen2ForCausalLM(self.config, self.weights, prealloc_weight_mem_on_npu=True)
        mock_new_qwen_model.assert_called_once_with(
            self.config, "model", quant_config=None
        )

        self.config.quantize = "w8a8sc"
        FlashQwen2ForCausalLM(self.config, self.weights)
        mock_tensor_head.load_weight.assert_called_once_with(
            self.config,
            prefix="lm_head",
            weights=self.weights,
            is_norm=False,
        )

        self.config.quantize = ""
        self.config.tie_word_embeddings = True
        FlashQwen2ForCausalLM(self.config, self.weights)
        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=["model.embed_tokens"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.WeightWrapper")
    def test_init_ascend_weight(
        self, mock_weight_wrapper, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
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

        ins = FlashQwen2ForCausalLM(self.config, self.weights, prealloc_weight_mem_on_npu=True)
        ins.lm_head.get_weight_transpose_type()[0] = TransposeType.NOT_TRANSPOSE
        ins.graph_manager = MagicMock()
        ins.graph_manager.set_param.return_value = True
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()

        self.config.quantize = QuantType.W8A8_PDMIX
        ins = FlashQwen2ForCausalLM(self.config, self.weights, prealloc_weight_mem_on_npu=True)
        ins.graph_manager = MagicMock()
        ins.graph_manager.set_param.return_value = True
        ins.lm_head.get_weight_transpose_type()[0] = TransposeType.TRANSPOSE
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()

        ins.adapter_manager = AdapterManager(self.weights)
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_model_prepare_inputs_prefill(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 0], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 3
        golden_kv_cache = [(torch.zeros([1, 128, 1, 128]).npu(), torch.zeros([1, 128, 1, 128]).npu()) for _ in range(2)]
        ins.long_seq_modifier = MagicMock()
        ins.long_seq_modifier.modify_inputs.return_value = True
        ins.qlen_modifier = MagicMock()
        ins.qlen_modifier.modify_inputs.return_value = True
        ins.lora_modifier = MagicMock()
        ins.lora_modifier.modify_inputs.return_value = True
        ins.flash_comm_modifier = MagicMock()
        ins.flash_comm_modifier.modify_inputs.return_value = True
        ins.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, True, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(ins.acl_operation_inputs), 12) # 12: base inputs

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_model_prepare_inputs_decode(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        golden_input_ids = torch.tensor([23561, 235, 18]).npu()
        golden_position_ids = torch.tensor([0, 1, 0], dtype=torch.int32).npu()
        golden_block_tables = torch.tensor([1, 2]).npu()
        golden_slots = torch.tensor([0, 1, 2]).npu()
        golden_input_lengths = torch.tensor([2, 1]).npu()
        golden_max_seq_len = 3
        golden_kv_cache = [(torch.zeros([1, 128, 1, 128]).npu(), torch.zeros([1, 128, 1, 128]).npu()) for _ in range(2)]
        ins.long_seq_modifier = MagicMock()
        ins.long_seq_modifier.modify_inputs.return_value = True
        ins.qlen_modifier = MagicMock()
        ins.qlen_modifier.modify_inputs.return_value = True
        ins.lora_modifier = MagicMock()
        ins.lora_modifier.modify_inputs.return_value = True
        ins.flash_comm_modifier = MagicMock()
        ins.flash_comm_modifier.modify_inputs.return_value = True
        ins.prepare_inputs_for_ascend(
            golden_input_ids, golden_position_ids, False, golden_kv_cache, golden_block_tables,
            golden_slots, golden_input_lengths, golden_max_seq_len, lm_head_indices=None
        )
        self.assertEqual(len(ins.acl_operation_inputs), 12) # 12: base inputs

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_forward(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.graph_manager = MagicMock()
        ins.graph_manager.select_and_execute.return_value = torch.zeros([1024], dtype=torch.float16).npu()
        ins.execute_ascend_operator([], {}, True)
        ins.graph_manager.select_and_execute.assert_called_once()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_init_ascend_kvcache(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.acl_encoder_operation = MagicMock()
        ins.acl_decoder_operation = MagicMock()
        cache = torch.zeros([1, 128, 1, 128], dtype=torch.float16).npu()
        ins.init_kvcache([(cache, cache)])
        self.assertEqual(ins.ascend_kcache_id, id(cache))

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_flash_comm_gate(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.tp_world_size = 2
        ins.soc_info = MagicMock()
        ins.soc_info.soc_version = 255
        ins.soc_info.is_support_hccs.return_value = True
        self.assertTrue(ins.flash_comm_gate(self.weights))

        ins.tp_world_size = 1
        self.assertFalse(ins.flash_comm_gate(self.weights))

        ins.tp_world_size = 2
        ins.soc_info.soc_version = 100
        self.assertFalse(ins.flash_comm_gate(self.weights))

        ins.soc_info.soc_version = 200
        ins.tp_world_size = 8
        self.assertFalse(ins.flash_comm_gate(self.weights))

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_update_matmul_params(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        # A2 + float16 -> aclnn backend
        ins.soc_info.soc_version = 223
        ins.dtype = torch.float16
        ins._update_matmul_params(quantize=QuantType.FLOAT)
        self.assertTrue(ins.aclnn_matmul_backend)
        # W8A8 -> aclnn backend
        ins.aclnn_matmul_backend = False
        ins._update_matmul_params(quantize=QuantType.W8A8)
        self.assertTrue(ins.aclnn_matmul_backend)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_init_kvcache_all_ops(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        # prepare operations
        ins.graph_manager = MagicMock()
        ins.prefix_cache_enable = True
        cache = torch.zeros([1, 128, 1, 128], dtype=torch.float16).npu()
        ins.init_kvcache([(cache, cache)])
        # all available ops should receive kv cache
        ins.graph_manager.set_kv_cache.assert_called_once()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_init_yarn(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        config = Qwen2Config(
            model_type="qwen2",
            hidden_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_hidden_layers=28,
            rms_norm_eps=1e-6,
            torch_dtype=torch.float16,
            vocab_size=125696,
            tie_word_embeddings=False,
        )
        config.rope_scaling = MagicMock()
        config.rope_scaling.type = "yarn"
        config.rope_scaling.factor = None
        with self.assertRaises(ValueError):
            ins = FlashQwen2ForCausalLM(config, self.weights)

        config.rope_scaling.factor = 0.5
        ins = FlashQwen2ForCausalLM(config, self.weights)
        self.assertEqual(ins.mscale, 1)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.TensorHead")
    def test_init_layerwiseDisaggregated(
            self,
            mock_tensor_head,
            mock_load_column_multi,
            mock_qwen_model,
            _mock_init_so
    ) -> None:
        kwargs = {
            "layerwise_disaggregated": "true",
            "layerwise_disaggregated_role_type": "master",
            "inference_mode": MagicMock()
        }
        FlashQwen2ForCausalLM(self.config, self.weights, **kwargs)
        mock_qwen_model.assert_called_with(
            self.config, self.weights, model_prefix="model", lmhead_prefix="lm_head",
            attn_decode_backend=OpBackend.ATB,
            load_list=[0, 27],
            layerwise_disaggregated=True
        )
        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )
        

        kwargs = {
            "layerwise_disaggregated": "true",
            "layerwise_disaggregated_role_type": "slave",
            "inference_mode": MagicMock()
        }
        FlashQwen2ForCausalLM(self.config, self.weights, **kwargs)
        mock_qwen_model.assert_called()
        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )
        
    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.WeightWrapper")
    def test_init_edge_ascend_weight_layerwiseDisaggregated(self, mock_weight_wrapper, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
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
        
        kwargs = {
            "layerwise_disaggregated": "true",
            "layerwise_disaggregated_role_type": "slave",
            "inference_mode": MagicMock()
        }
        ins = FlashQwen2ForCausalLM(self.config, self.weights, **kwargs)
        with patch.object(ins.graph_manager, 'set_param') as mock_set_param:
            ins.init_ascend_weight()

        kwargs = {
            "layerwise_disaggregated": "true",
            "layerwise_disaggregated_role_type": "master",
            "inference_mode": MagicMock()
        }
        ins = FlashQwen2ForCausalLM(self.config, self.weights, **kwargs)
        with patch.object(ins.graph_manager, 'set_param') as mock_set_param:
            ins.init_ascend_weight()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_get_model_weights(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        """Test get_model_weights method."""
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.enable_swiglu_quant = True
        ins.num_layers = 2
        
        # Mock model components
        ins.model = MagicMock(spec=torch.nn.Module)
        ins.model.embed_tokens = MagicMock()
        ins.model.embed_tokens.get_weights_for_atb_graph.return_value = [0]
        
        ins.model.norm = MagicMock()
        ins.model.norm.get_weights_for_atb_graph.return_value = [100]
        
        ins.lm_head = MagicMock(spec=ParallelLMHead)
        ins.lm_head.get_weights_for_atb_graph.return_value = [200]
        
        # Mock layers
        mock_layers = []
        for _ in range(2):
            mock_layer = MagicMock()
            mock_layer.input_layernorm = MagicMock()
            mock_layer.input_layernorm.get_weights_for_atb_graph.return_value = [1, 2, 3, 4]
            
            mock_layer.self_attn = MagicMock()
            mock_layer.self_attn.qkv_proj = MagicMock()
            mock_layer.self_attn.qkv_proj.get_weights_for_atb_graph.return_value = list(range(18))
            mock_layer.self_attn.qkv_proj.get_linear_descs.return_value = list(range(7))
            mock_layer.self_attn.qkv_proj.get_weight_transpose_type.return_value = list(range(7))
            
            mock_layer.self_attn.o_proj = MagicMock()
            mock_layer.self_attn.o_proj.get_weights_for_atb_graph.return_value = list(range(6))
            mock_layer.self_attn.o_proj.get_linear_descs.return_value = list(range(7))
            mock_layer.self_attn.o_proj.get_weight_transpose_type.return_value = list(range(7))
            
            mock_layer.post_attention_layernorm = MagicMock()
            mock_layer.post_attention_layernorm.get_weights_for_atb_graph.return_value = [5, 6, 7, 8]
            
            mock_layer.mlp = MagicMock()
            mock_layer.mlp.gate_up_proj = MagicMock()
            mock_layer.mlp.gate_up_proj.get_weights_for_atb_graph.return_value = list(range(12))
            mock_layer.mlp.gate_up_proj.get_linear_descs.return_value = list(range(7))
            mock_layer.mlp.gate_up_proj.get_weight_transpose_type.return_value = list(range(7))
            
            mock_layer.mlp.down_proj = MagicMock()
            mock_layer.mlp.down_proj.get_weights_for_atb_graph.return_value = list(range(6))
            mock_layer.mlp.down_proj.get_linear_descs.return_value = list(range(7))
            mock_layer.mlp.down_proj.get_weight_transpose_type.return_value = list(range(7))
            
            mock_layers.append(mock_layer)
        
        ins.model.layers = mock_layers
        ins.config.use_qk_norm = False
        
        # Test get_model_weights
        weights, linear_descs, weight_transpose_types = ins.get_model_weights()
        
        # Verify structure
        # 1 (embed_tokens) + 2 * 50 (layers) + 1 (norm) + 1 (lm_head) = 103
        self.assertEqual(len(weights), 103)
        # linear_descs: 2 layers, each with 28 descriptors
        self.assertEqual(len(linear_descs), 2)
        self.assertEqual(len(linear_descs[0]), 28)
        self.assertEqual(len(linear_descs[1]), 28)
        # weight_transpose_types: 2 layers, each with 28 types
        self.assertEqual(len(weight_transpose_types), 2)
        self.assertEqual(len(weight_transpose_types[0]), 28)
        self.assertEqual(len(weight_transpose_types[1]), 28)
        
        # Verify embed_tokens was called
        ins.model.embed_tokens.get_weights_for_atb_graph.assert_called_once()
        # Verify norm was called
        ins.model.norm.get_weights_for_atb_graph.assert_called_once_with(padding=False)
        # Verify lm_head was called
        ins.lm_head.get_weights_for_atb_graph.assert_called_once_with(padding=False)
        
        # Test with quant_type
        quant_type = QuantType.W8A8
        weights, linear_descs, weight_transpose_types = ins.get_model_weights(quant_type=quant_type)
        
        # Verify quant_type was passed to layers
        for layer in mock_layers:
            layer.self_attn.qkv_proj.get_weights_for_atb_graph.assert_called_with(quant_type=quant_type)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    def test_get_layer_weights_with_swiglu_quant_disabled(self, _mock_load_column_multi, _mock_model, _mock_init_so):
        """Test get_layer_weights with enable_swiglu_quant=False."""
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.enable_swiglu_quant = False
        
        # Mock layer components
        mock_layer = MagicMock()
        mock_layer.input_layernorm = MagicMock()
        mock_layer.input_layernorm.get_weights_for_atb_graph.return_value = torch.tensor([1, 2, 3, 4])
        
        mock_layer.self_attn = MagicMock()
        mock_layer.self_attn.qkv_proj = MagicMock()
        mock_layer.self_attn.qkv_proj.get_weights_for_atb_graph.return_value = torch.tensor(list(range(18)))
        mock_layer.self_attn.qkv_proj.get_linear_descs.return_value = torch.tensor(list(range(7)))
        mock_layer.self_attn.qkv_proj.get_weight_transpose_type.return_value = torch.tensor(list(range(7)))
        
        mock_layer.self_attn.o_proj = MagicMock()
        mock_layer.self_attn.o_proj.get_weights_for_atb_graph.return_value = torch.tensor(list(range(6)))
        mock_layer.self_attn.o_proj.get_linear_descs.return_value = torch.tensor(list(range(7)))
        mock_layer.self_attn.o_proj.get_weight_transpose_type.return_value = torch.tensor(list(range(7)))
        
        mock_layer.post_attention_layernorm = MagicMock()
        mock_layer.post_attention_layernorm.get_weights_for_atb_graph.return_value = torch.tensor([5, 6, 7, 8])
        
        mock_layer.mlp = MagicMock()
        mock_layer.mlp.gate_up_proj = MagicMock()
        mock_layer.mlp.gate_up_proj.get_weights_for_atb_graph.return_value = torch.tensor(list(range(12)))
        mock_layer.mlp.gate_up_proj.get_linear_descs.return_value = torch.tensor(list(range(7)))
        mock_layer.mlp.gate_up_proj.get_weight_transpose_type.return_value = torch.tensor(list(range(7)))
        
        mock_layer.mlp.down_proj = MagicMock()
        mock_layer.mlp.down_proj.get_weights_for_atb_graph.return_value = torch.tensor(list(range(6)))
        mock_layer.mlp.down_proj.get_linear_descs.return_value = torch.tensor(list(range(7)))
        mock_layer.mlp.down_proj.get_weight_transpose_type.return_value = torch.tensor(list(range(7)))
        
        ins.config.use_qk_norm = False
        
        _, _, _ = ins.get_layer_weights(mock_layer)
        
        # Verify down_proj was called with enable_swiglu_quant=False
        mock_layer.mlp.down_proj.get_weights_for_atb_graph.assert_called_once_with(
            is_swiglu_quant_enabled=False, quant_type=None
        )

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.WeightWrapper")
    @patch(f"{FLASH_QWEN2}.MlpWrapper")
    @patch(f"{FLASH_QWEN2}.AttnWrapper")
    def test_get_weights_basic(self, mock_attn_wrapper_class, mock_mlp_wrapper_class, mock_weight_wrapper_class, _mock_load_column_multi, _mock_model, _mock_init_so):
        """Test get_weights method with basic configuration."""
        ins = FlashQwen2ForCausalLM(self.config, self.weights)
        ins.quantize = None
        ins.enable_rope_quant_kvcache = False
        ins.enable_swiglu_quant = True
        ins.soc_info = MagicMock()
        ins.tp_rank = 0
        ins.num_layers = 2
        ins.enable_intra_layer_add_norm = False
        ins.enable_inter_layer_add_norm = False
        ins.adapter_manager = None
        
        # Mock transformer
        ins.transformer = MagicMock()
        ins.transformer.wte = MagicMock()
        ins.transformer.ln_f = MagicMock()
        ins.transformer.h = []
        for _ in range(2):
            mock_layer = MagicMock()
            mock_layer.attn = MagicMock()
            ins.transformer.h.append(mock_layer)
        
        ins.config.use_qk_norm = False
        ins.config.quantization_config = MagicMock()
        ins.config.quantization_config.kv_quant_type = None
        ins.config.quantization_config.fa_quant_type = None
        
        mock_weight_wrapper = MagicMock()
        mock_weight_wrapper_class.return_value = mock_weight_wrapper
        
        result = ins.get_weights()
        
        # Verify AttnWrapper and MlpWrapper were created
        mock_attn_wrapper_class.assert_called_once_with(
            norm_name='ln_1',
            wrapper_name='attn',
            pack_name='c_attn',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='c_proj'
        )
        mock_mlp_wrapper_class.assert_called_once_with(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='w2_w1',
            sep_names=['w2', 'w1'],
            down_name='c_proj'
        )
        
        # Verify WeightWrapper was created with correct kwargs
        mock_weight_wrapper_class.assert_called_once()
        call_kwargs = mock_weight_wrapper_class.call_args[1]
        self.assertFalse(call_kwargs['enable_rope_quant_kvcache'])
        self.assertTrue(call_kwargs['enable_swiglu_quant'])
        
        # Verify registration calls
        mock_weight_wrapper.register_embedding.assert_called_once_with(ins.transformer.wte)
        self.assertEqual(mock_weight_wrapper.register_layer.call_count, 2)
        mock_weight_wrapper.register_model_norm.assert_called_once_with(ins.transformer.ln_f)
        mock_weight_wrapper.register_model_lmhead.assert_called_once_with(ins.lm_head)
        
        self.assertEqual(result, mock_weight_wrapper)


if __name__ == "__main__":
    unittest.main()
