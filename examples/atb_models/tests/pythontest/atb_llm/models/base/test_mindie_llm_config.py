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
from dataclasses import dataclass

from atb_llm.models.base.mindie_llm_config import MindIELLMConfig, ModelStatus
from atb_llm.models.base.config import RopeScaling, QuantizationConfig
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType


@dataclass
class MockBaseConfig:
    num_attention_heads = 8
    num_key_value_heads = None
    hidden_size = 128
    num_hidden_layers = 12
    rope_scaling = RopeScaling()
    quantization_config = QuantizationConfig()
    rms_norm_eps = 1e-5
    quantize = None


class TestMindIELLMConfig(unittest.TestCase):
    def setUp(self):
        self.hf_config = MockBaseConfig
        self.llm_config = None
        self.mapping = Mapping(rank=0, world_size=2)
        self.mindie_llm_config = MindIELLMConfig(self.hf_config, self.llm_config, self.mapping)
    
    def test_mindie_llm_config_init(self):
        self.assertIsInstance(self.mindie_llm_config.soc_info, NPUSocInfo)
        self.assertFalse(hasattr(self.mindie_llm_config, "model_status"))
    
    def test_modify_hf_config(self):
        self.assertEqual(self.hf_config.num_key_value_heads, 8)
        self.assertEqual(self.hf_config.rope_scaling.rope_type, "linear")
        self.assertAlmostEqual(self.hf_config.rope_theta, 10000.0)
        self.assertAlmostEqual(self.hf_config.alibi_bias_max, 8.0)
        self.assertEqual(self.hf_config.pe_type, "ROPE")
        self.assertAlmostEqual(self.hf_config.epsilon, 1e-5)
    
    def test_create_model_status(self):
        self.mindie_llm_config.soc_info.soc_version = 220
        model_status = ModelStatus.from_config(self.mindie_llm_config)
        self.assertFalse(model_status.enable_matmul_nz)
        self.assertEqual(model_status.head_dim, 16)
        self.assertEqual(model_status.num_attention_heads, 4)
        self.assertEqual(model_status.num_key_value_heads, 4)
        self.assertEqual(model_status.num_hidden_layers, 12)
        self.assertEqual(model_status.position_embedding_type, PositionEmbeddingType.ROPE)


if __name__ == "__main__":
    unittest.main()