# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
import json
import os
import tempfile

import unittest
from unittest.mock import MagicMock, patch
from unittest import TestCase
import torch

from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.models.deepseekv2.modeling_deepseekv2 import (
    DeepseekV2RMSNorm,
    DeepseekV2RMSNormBias,
    DeepseekV2RMSNormWrapper,
    DeepseekV2RMSNormAntiOutlierWrapper,
    FlashDeepseekV2Attention,
    DeepseekV2MLP,
    DeepseekV2MoE,
    FlashDeepseekV2DecoderLayer,
    FlashDeepseekV2Model
)
from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.utils.configuration_utils import LLMConfig


class TestFlashDeepseekV2Model(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path1 = os.path.join(self.temp_dir.name, 'test_config1.json')
        self.test_json_config1 = {
            "llm": {
                "ccl": {
                "enable_mc2": "true"
                },
                "stream_options": {
                "micro_batch": "false"
                },
                "engine": {
                "graph": "cpp"
                },
                "parallel_options": {
                "o_proj_local_tp": -1,
                "lm_head_local_tp": -1,
                "hccl_buffer": 128,
                "hccl_moe_ep_buffer": 512,
                "hccl_moe_tp_buffer": 64
                },
                "pmcc_obfuscation_options": {
                "enable_model_obfuscation": False,
                "data_obfuscation_ca_dir": "",
                "kms_agent_port": 1024
                },
                "kv_cache_options": {
                "enable_nz": False
                },
                "weights_options": {
                "low_cpu_memory_mode": False
                },
                "enable_reasoning": "false"
            },
            "models": {
                "qwen_moe": {
                "eplb": {
                    "level": 0,
                    "expert_map_file": ""
                },
                "ep_level": 2
                },
                "deepseekv2": {
                    "eplb": {
                        "level": 0,
                        "expert_map_file": "",
                        "num_redundant_experts": 0,
                        "aggregate_threshold": 128,
                        "num_expert_update_ready_countdown": 16
                    },
                    "ep_level": 1,
                    "enable_dispatch_combine_v2": False,
                    "communication_backend": {
                        "prefill":"lccl",
                        "decode": "lccl"
                    },
                    "enable_gmmswigluquant": False,
                    "enable_oproj_prefetch": False,
                    "enable_mlapo_prefetch": False,
                    "num_dangling_shared_experts": 0,
                    "enable_swiglu_quant_for_shared_experts": False,
                    "enable_init_routing_cutoff": False,
                    "topk_scaling_factor": 1.0,
                    "mlp_full_tp": False,
                    "h3p":{
                        "enable_qkvdown_dp": "true",
                        "enable_gating_dp": "true",
                        "enable_shared_expert_dp": "false",
                        "enable_shared_expert_overlap": "false"
                    }
                }
            },
            "enable_atlas_gmm_fused": False
        }
        with open(self.config_path1, 'w') as f:
            json.dump(self.test_json_config1, f)

        self.config_dict = {
            "q_lora_rank": None,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "v_head_dim": 128,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_hidden_layers": 27,
            "rope_scaling": {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 0.707,
                "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
            "mla_quantize": "w8a8",
            "quantization_config": {
                "fa_quant_type": "FAKQuant"
            }
        }

        self.config = DeepseekV2Config.from_dict(self.config_dict)
        self.llm_config = LLMConfig(self.config_path1)
        self.config.parallel_embedding = False
        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.quantize = None
        self.weights.get_tensor.return_value = torch.empty(128, 192, dtype=torch.bfloat16)
        self.weights.get_multi_weights_col.return_value = torch.empty(128, 192, dtype=torch.bfloat16)
        self.weights.get_replicated_weights.return_value = torch.ones(
            self.config.kv_lora_rank + self.config.qk_rope_head_dim, self.config.hidden_size, dtype=torch.bfloat16
            )
        self.weights.get_multi_weights_row.return_value = torch.empty(128, 192, dtype=torch.bfloat16)
        self.weights.get_whole_tensor.return_value = torch.empty(128, 192, dtype=torch.bfloat16)
        self.weights.get_partial_sharded.return_value = torch.empty(128, 192, dtype=torch.bfloat16)
        self.weights.get_shape.return_value = [128, 192]
        self.weights.process_group.size.return_value = 1
        self.weights.process_group.rank.return_value = 1
        self.weights.mapping.moe_ep.rank = 0
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.mlp_tp.rank = 0

    def test_deepseekv2rmsnorm(self):
        DeepseekV2RMSNorm("attn", self.weights)
        self.weights.get_tensor.assert_called_once_with("attn.weight")

    def test_deepseekv2rmsnormbias(self):
        DeepseekV2RMSNormBias("attn", self.weights)
        self.weights.get_tensor.assert_called_with("attn.bias")

    def test_deepseekv2rmsnormwrapper(self):
        DeepseekV2RMSNormWrapper("attn", self.weights)
        self.weights.get_tensor.assert_called_with("attn.module.bias")

    def test_deepseekv2rmsnormantioutlierwrapper(self):
        DeepseekV2RMSNormAntiOutlierWrapper("attn", self.weights)
        self.weights.get_tensor.assert_called_with("attn.anti.bias")

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2attention(self, mock_func):
        FlashDeepseekV2Attention("attn", self.config, self.weights)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2attention_rope_none(self, mock_func):
        self.config.rope_scaling_dict = None
        FlashDeepseekV2Attention("attn", self.config, self.weights)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2attention_not_yarn(self, mock_func):
        dict_temp = copy.deepcopy(self.config_dict)
        dict_temp["rope_scaling"]["type"] = None
        dict_temp_config = DeepseekV2Config.from_dict(dict_temp)
        try:
            FlashDeepseekV2Attention("attn", dict_temp_config, self.weights)
        except ValueError as e:
            self.assertEqual(type(e), ValueError)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2attention_mla_quantize_none(self, mock_func):
        delattr(self.config, "mla_quantize")
        self.config.rope_scaling_dict.pop("mscale_all_dim", None)
        FlashDeepseekV2Attention("attn", self.config, self.weights)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2attention_w8a8sc(self, mock_func):
        self.config.quantize = "w8a8sc"
        self.weights.quantize = "w8a8sc"
        FlashDeepseekV2Attention("attn", self.config, self.weights)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2attention_q_lora_rank(self, mock_func):
        setattr(self.config, "q_lora_rank", 1536)
        FlashDeepseekV2Attention("attn", self.config, self.weights, self.llm_config)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.calc_linear_pack_type", return_value=PackType.ALL_W8A8)
    def test_flashdeepseekv2attention_q_lora_rank_w8a8(self, mock_func1, mock_func2):
        setattr(self.config, "q_lora_rank", 1536)
        FlashDeepseekV2Attention("attn", self.config, self.weights, self.llm_config)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2attention_o_proj_local_tp(self, mock_func):
        setattr(self.llm_config.llm.parallel_options, "o_proj_local_tp", 2)
        FlashDeepseekV2Attention("attn", self.config, self.weights, self.llm_config)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2attention_fa_quant_type_none(self, mock_func):
        setattr(self.config.quantization_config, "fa_quant_type", None)
        FlashDeepseekV2Attention("attn", self.config, self.weights, self.llm_config)

    def test_deepseekv2mlp(self):
        DeepseekV2MLP("mlp", self.config, self.weights)

    def test_deepseekv2mlp_w8a8sc(self):
        self.config.quantize = "w8a8sc"
        self.weights.quantize = "w8a8sc"
        DeepseekV2MLP("mlp", self.config, self.weights)
    
    def test_deepseekv2moe(self):
        DeepseekV2MoE("moe", self.config, self.weights, DeepseekV2MLP, 0)

    def test_deepseekv2moe_shared_expert_none(self):
        moe = DeepseekV2MoE("moe", self.config, self.weights, DeepseekV2MLP, 0)
        delattr(moe, "shared_experts")

    def test_deepseekv2moe_w8a8sc(self):
        self.config.quantize = "w8a8sc"
        self.weights.quantize = "w8a8sc"
        DeepseekV2MoE("moe", self.config, self.weights, DeepseekV2MLP, 0)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2decoderlayer(self, mock_func):
        FlashDeepseekV2DecoderLayer(0, self.config, self.weights)
        FlashDeepseekV2DecoderLayer(1, self.config, self.weights)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2decoderlayer_mlp_full_tp(self, mock_func):
        delattr(self.config, "n_routed_experts")
        self.llm_config.models.deepseekv2.mlp_full_tp = True
        FlashDeepseekV2DecoderLayer(0, self.config, self.weights)
        FlashDeepseekV2DecoderLayer(1, self.config, self.weights)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2decoderlayer_w8a8(self, mock_func):
        self.config.quantize = "w8a8"
        self.weights.quantize = "w8a8"
        FlashDeepseekV2DecoderLayer(0, self.config, self.weights)
        FlashDeepseekV2DecoderLayer(1, self.config, self.weights)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2decoderlayer_w8a8sc(self, mock_func):
        self.config.quantize = "w8a8sc"
        self.weights.quantize = "w8a8sc"
        FlashDeepseekV2DecoderLayer(0, self.config, self.weights)
        FlashDeepseekV2DecoderLayer(1, self.config, self.weights)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2decoderlayer_w8a8_anti(self, mock_func):
        self.self_attn = FlashDeepseekV2Attention("attn", self.config, self.weights, self.llm_config)
        self.self_attn.pack_type = PackType.ALL_W8A8_ANTI
        FlashDeepseekV2DecoderLayer(0, self.config, self.weights, self.llm_config)
        FlashDeepseekV2DecoderLayer(1, self.config, self.weights, self.llm_config)

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2decoderlayer_w8a8sc_anti(self, mock_func):
        self.self_attn = FlashDeepseekV2Attention("attn", self.config, self.weights, self.llm_config)
        self.self_attn.pack_type = PackType.ALL_W8A8SC_ANTI
        layer0 = FlashDeepseekV2DecoderLayer(0, self.config, self.weights, self.llm_config)
        layer0.self_attn.pack_type = PackType.ALL_W8A8SC_ANTI
        layer1 = FlashDeepseekV2DecoderLayer(1, self.config, self.weights, self.llm_config)
        layer1.self_attn.pack_type = PackType.ALL_W8A8SC_ANTI

    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    def test_flashdeepseekv2decoderlayer_w8a8sc_anti_mlp(self, mock_func):
        self.mlp = DeepseekV2MoE("moe", self.config, self.weights, DeepseekV2MLP, 0, self.llm_config)
        self.mlp.pack_type = PackType.ALL_W8A8SC_ANTI
        FlashDeepseekV2DecoderLayer(0, self.config, self.weights, self.llm_config)
        FlashDeepseekV2DecoderLayer(1, self.config, self.weights, self.llm_config)
    
    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelEmbedding")
    def test_flash_deepseekv2_model_mtp(self, mock_embedding, mock_func):
        self.config.num_speculative_tokens = 3
        FlashDeepseekV2Model(self.config, self.weights)
    
    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelColumnLinear.load_multi")
    @patch("atb_llm.models.deepseekv2.modeling_deepseekv2.TensorParallelEmbedding")
    def test_flash_deepseekv2_model_mtp_quantize(self, mock_embedding, mock_func):
        self.config.num_speculative_tokens = 3
        setattr(self.config, "mtp_quantize", "w8a8_dynamic")
        FlashDeepseekV2Model(self.config, self.weights, self.llm_config)


if __name__ == '__main__':
    unittest.main()