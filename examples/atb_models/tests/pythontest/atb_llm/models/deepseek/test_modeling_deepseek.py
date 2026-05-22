# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import os
import tempfile

import unittest
from unittest.mock import MagicMock, patch
from unittest import TestCase
from ddt import ddt, data
import torch

from atb_llm.utils.quantize.pack_type import PackType
from atb_llm.models.deepseek.config_deepseek import DeepseekConfig
from atb_llm.models.deepseek.modeling_deepseek import (
    DeepseekRMSNorm,
    DeepseekMLP,
    DeepseekMoE,
    DeepseekEp,
    FlashDeepseekAttention,
    FlashDeepseekLayer,
    FlashDeepseekModel
)
from atb_llm.utils.configuration_utils import LLMConfig
from atb_llm.utils.moe_utils import random_generation


@ddt
class TestFlashDeepseekModel(TestCase):
    def setUp(self):
        self.config = DeepseekConfig()
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
        self.weights.process_group.size.return_value = 8
        self.weights.mapping.moe_ep.rank = 0
        self.init_expert_table = random_generation(4, 12, 4, 0, mix_shared_routing=False)

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
        self.llm_config = LLMConfig(self.config_path1)

    def test_deepseekrmsnorm(self):
        self.weights.sharded = False
        DeepseekRMSNorm("attn", self.weights)
        self.weights.get_tensor.assert_called_once_with("attn.weight")

    def test_deepseekmlp(self):
        self.weights.sharded = False
        deepseek_mlp = DeepseekMLP("mlp", self.config, self.weights)
        self.assertEqual(PackType.ALL_FP, deepseek_mlp.pack_type)

    @patch("atb_llm.models.deepseek.modeling_deepseek.calc_linear_pack_type", return_value=PackType.ALL_W8A8SC)
    def test_deepseekmlp_w8a8sc(self, mock_type):
        self.weights.sharded = False
        deepseek_mlp = DeepseekMLP("mlp", self.config, self.weights)
        self.assertEqual(PackType.ALL_W8A8SC, deepseek_mlp.pack_type)

    @patch("atb_llm.models.deepseek.modeling_deepseek.calc_linear_pack_type", return_value=None)
    def test_deepseekmlp_none(self, mock_type):
        self.weights.sharded = False
        deepseek_mlp = DeepseekMLP("mlp", self.config, self.weights)
        self.assertEqual(None, deepseek_mlp.pack_type)

    def test_flashdeepseekattention(self):
        self.weights.sharded = False
        flash_deepseek_attention = FlashDeepseekAttention("attn", self.config, self.weights)
        self.assertEqual(PackType.ALL_FP, flash_deepseek_attention.pack_type)

    def test_flashdeepseekattention_2(self):
        self.weights.sharded = False
        self.config.num_attention_heads = 31
        self.config.num_key_value_heads = 30
        with self.assertRaises(ValueError):
            FlashDeepseekAttention("attn", self.config, self.weights)

    def test_flashdeepseekattention_3(self):
        self.weights.sharded = False
        self.config.num_key_value_heads = 2
        FlashDeepseekAttention("attn", self.config, self.weights)

    @patch("atb_llm.models.deepseek.modeling_deepseek.calc_linear_pack_type", return_value=PackType.ALL_W8A8SC)
    def test_flashdeepseekattention_w8a8sc(self, mock_type):
        self.weights.sharded = False
        flash_deepseek_attention = FlashDeepseekAttention("attn", self.config, self.weights)
        self.assertEqual(PackType.ALL_W8A8SC, flash_deepseek_attention.pack_type)

    def test_flashdeepseeklayer(self):
        self.weights.sharded = False
        FlashDeepseekLayer(0, self.config, self.weights)
        FlashDeepseekLayer(1, self.config, self.weights)
        self.weights.get_tensor.assert_any_call("model.layers.0.input_layernorm.weight")

    def test_flashdeepseekmodel(self):
        self.weights.sharded = False
        FlashDeepseekModel(self.config, self.weights)
        self.weights.get_tensor.assert_any_call("model.layers.0.input_layernorm.weight")

    def test_deepseekmoe(self):
        self.weights.sharded = False
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    def test_deepseekmoe_noaux_tc(self):
        self.weights.sharded = False
        setattr(self.config, "topk_method", "noaux_tc")
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    def test_deepseekmoe_no_ep(self):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = False
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    def test_deepseekmoe_ep(self):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = True
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    def test_deepseekmoe_init_experts(self):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = False
        self.weights.mapping.has_moe_tp.return_value = False
        self.weights.mapping.mlp_tp.rank = 0
        self.weights.mapping.mlp_tp.group_size = 16
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    @patch("atb_llm.models.deepseek.modeling_deepseek.parse_ep_balance_file", return_value=[[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    @patch("os.path.exists", return_value=True)
    @patch("atb_llm.models.deepseek.modeling_deepseek.get_linear_quant_type")
    def test_deepseekmoe_eplb(self, mock1, mock2, mock3):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = True
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        mock1.return_value = False
        setattr(self.llm_config.models.deepseekv2.eplb, "level", 1)
        setattr(self.llm_config.models.deepseekv2.eplb, "expert_map_file", "/xxx/xxx")
        setattr(self.config, "ep_level", 3)
        setattr(self.config, "n_shared_experts", 16)
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, layer_id=0, llm_config=self.llm_config, init_expert_table=self.init_expert_table)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    @patch("atb_llm.models.deepseek.modeling_deepseek.parse_ep_balance_file", return_value=[[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    @patch("os.path.exists", return_value=True)
    def test_deepseekmoe_no_eplb_dangling(self, mock1, mock2):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = True
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        setattr(self.llm_config.models.deepseekv2, "num_dangling_shared_experts", 16)
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, layer_id=0, llm_config=self.llm_config, init_expert_table=self.init_expert_table)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    @patch("atb_llm.models.deepseek.modeling_deepseek.parse_ep_balance_file", return_value=[[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    @patch("os.path.exists", return_value=True)
    def test_deepseekmoe_eplb_dangling(self, mock1, mock2):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = True
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        setattr(self.llm_config.models.deepseekv2, "num_dangling_shared_experts", 16)
        setattr(self.llm_config.models.deepseekv2.eplb, "level", 1)
        setattr(self.llm_config.models.deepseekv2.eplb, "expert_map_file", "/xxx/xxx")
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, layer_id=0, llm_config=self.llm_config, init_expert_table=self.init_expert_table)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    @patch("atb_llm.models.deepseek.modeling_deepseek.parse_ep_balance_file", return_value=[[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    @patch("os.path.exists", return_value=True)
    def test_deepseekmoe_no_eplb_ep_lv3(self, mock1, mock2):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = True
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        setattr(self.llm_config.models.deepseekv2, "num_dangling_shared_experts", 16)
        setattr(self.config, "ep_level", 3)
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, layer_id=0, llm_config=self.llm_config, init_expert_table=self.init_expert_table)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    @patch("atb_llm.models.deepseek.modeling_deepseek.parse_ep_balance_file", return_value=[[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    @patch("os.path.exists", return_value=True)
    def test_deepseekmoe_no_eplb_ep_lv3_noaux_tc(self, mock1, mock2):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = True
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        setattr(self.llm_config.models.deepseekv2, "num_dangling_shared_experts", 16)
        setattr(self.config, "ep_level", 3)
        setattr(self.config, "topk_method", "noaux_tc")
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, layer_id=0, llm_config=self.llm_config, init_expert_table=self.init_expert_table)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    @patch("atb_llm.models.deepseek.modeling_deepseek.parse_ep_balance_file", return_value=None)
    @patch("os.path.exists", return_value=True)
    def test_deepseekmoe_eplb_without_loading_table(self, mock1, mock2):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = True
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        setattr(self.llm_config.models.deepseekv2.eplb, "level", 1)
        setattr(self.llm_config.models.deepseekv2.eplb, "expert_map_file", "/xxx/xxx")
        with self.assertRaises(ValueError):
            DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, layer_id=0, llm_config=self.llm_config)

    def test_deepseekmoe_no_eplb(self):
        self.weights.sharded = False
        self.weights.mapping.has_moe_ep.return_value = True
        self.weights.mapping.moe_tp.rank = 0
        self.weights.mapping.moe_tp.group_size = 16
        setattr(self.llm_config.models.deepseekv2.eplb, "level", 0)
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    def test_deepseekmoe_moe_quantize(self):
        self.weights.sharded = False
        setattr(self.config, "moe_quantize", "w8a8")
        setattr(self.config, "quantize", "w8a8")
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    def test_deepseekmoe_mix(self):
        self.weights.sharded = False
        DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config, mix_shared_routing=True)
        self.weights.get_tensor.assert_any_call("moe.gate.weight")

    @patch("atb_llm.models.deepseek.modeling_deepseek.calc_linear_pack_type", return_value=PackType.ALL_W8A8SC)
    def test_deepseekmoe_w8a8sc(self, mock_type):
        self.weights.sharded = False
        deepseek_moe = DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config, mix_shared_routing=True)
        self.assertEqual(PackType.ALL_W8A8SC, deepseek_moe.pack_type)

    @patch("atb_llm.models.deepseek.modeling_deepseek.calc_linear_pack_type", return_value=None)
    def test_deepseekmoe_none_type(self, mock_type):
        self.weights.sharded = False
        deepseek_moe = DeepseekMoE("moe", self.config, self.weights, DeepseekMLP, llm_config=self.llm_config, mix_shared_routing=True)
        self.assertEqual(None, deepseek_moe.pack_type)

    def test_deepseekep(self):
        self.weights.sharded = False
        DeepseekEp("moe", self.config, self.weights)
        self.weights.get_tensor.assert_any_call("moe.gate_proj.weight")



if __name__ == '__main__':
    unittest.main()