# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import copy
import unittest
from unittest.mock import patch, MagicMock
from ddt import ddt, unpack, data
from mindie_llm.runtime.models.deepseek_v3.input_builder_deepseek_v3 import DeepseekV3InputBuilder
from mindie_llm.runtime.models.deepseek_v3.config_deepseek_v3 import DeepseekV3Config
from mindie_llm.runtime.models.deepseek_v3.router_deepseek_v3 import DeepseekV3Router
from mindie_llm.runtime.utils.helpers.parameter_validators import (
    IntParameterValidator, FloatParameterValidator, BooleanParameterValidator, RangeParamaterValidator, 
    DictionaryParameterValidator, Field
)
from mindie_llm.runtime.config.configuration_utils import LLMConfig
from mindie_llm.runtime.models.base.router import BaseRouter


class MockLoadConfig:
    models_dict = {}
    llm_config_path = ""


class TestDeepseekv3Router(unittest.TestCase):

    def setUp(self):
        """Set up common test data for Deepseekv3Router tests."""
        self.mock_config_dict = {
            'model_type': 'deepseek_v3',
            'num_hidden_layers': 61,
            'max_position_embeddings': 163840,
            'vocab_size': 129280,
            'rope_scaling': None,
            'qk_nope_head_dim': 128,
            'qk_rope_head_dim': 64,
            'topk_method': "noaux_tc",
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "first_k_dense_replace": 3,
            "n_routed_experts": 256,
            "q_lora_rank": 1536,
        }
        self.mock_llm_config = MagicMock(spec=LLMConfig)
        self.mock_llm_config.llm = MagicMock()
        self.mock_llm_config.llm.tool_call_options = None
        self.mock_llm_config.llm.chat_template = None

        self.load_config = MockLoadConfig()
        self.load_config.tokenizer_path = 'test/deepseekv3'
        self.load_config.load_tokenizer = True
    
    @patch('mindie_llm.runtime.models.base.router.LLMConfig')
    def test_initialization(self, mock_llmconfig):
        """Verify DeepseekV3Router initializes correctly with minimal parameters."""

        router = DeepseekV3Router(
            config_dict=self.mock_config_dict,
            load_config=self.load_config
        )
        self.assertIsInstance(router, BaseRouter)
        self.assertEqual(router.config_dict, self.mock_config_dict)
        self.assertEqual(router._model_type, "deepseek_v3")
        self.assertEqual(router._model_type_cap, "DeepseekV3")

    @patch("mindie_llm.runtime.models.deepseek_v3.router_deepseek_v3.safe_get_tokenizer_from_pretrained")
    @patch('mindie_llm.runtime.models.base.router.LLMConfig')
    def test_get_llm_config_validators(self, mock_llmconfig, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        router = DeepseekV3Router(
            config_dict=self.mock_config_dict,
            load_config=self.load_config
        )

        llm_config_validators = router.get_llm_config_validators()

        self.assertIn("llm", llm_config_validators)
        llm_validator = llm_config_validators["llm"]
        self.assertIsInstance(llm_validator, DictionaryParameterValidator)
        self.assertIn("models", llm_config_validators)
        deepseekv2_config_validator = llm_config_validators["models"]["deepseekv2"]
        self.assertIsInstance(deepseekv2_config_validator, DictionaryParameterValidator)

    @patch("mindie_llm.runtime.models.deepseek_v3.router_deepseek_v3.safe_get_tokenizer_from_pretrained")
    @patch('mindie_llm.runtime.models.base.router.LLMConfig')
    def test_get_tokenizer(self, mock_llmconfig, mock_get_tokenizer):
        """Verify get_tokenizer correctly configures and returns tokenizer."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        router = DeepseekV3Router(
            config_dict=self.mock_config_dict,
            load_config=self.load_config
        )
        tokenizer = router._get_tokenizer()
        
        mock_get_tokenizer.assert_called_once_with(
            "test/deepseekv3",
            padding_side="left",
            trust_remote_code=False
        )
        self.assertEqual(tokenizer, mock_tokenizer)

    @patch("mindie_llm.runtime.models.deepseek_v3.router_deepseek_v3.safe_get_tokenizer_from_pretrained")
    @patch('mindie_llm.runtime.models.base.router.LLMConfig')
    def test_get_input_builder(self, mock_llmconfig, mock_get_tokenizer):
        """Verify get_input_builder correctly returns DeepseekV3InputBuilder."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        router = DeepseekV3Router(
            config_dict=self.mock_config_dict,
            load_config=self.load_config
        )
        
        input_builder = router._get_input_builder()
        self.assertIsInstance(input_builder, DeepseekV3InputBuilder)

    @patch("mindie_llm.runtime.models.deepseek_v3.router_deepseek_v3.safe_get_tokenizer_from_pretrained")
    @patch('mindie_llm.runtime.models.base.router.LLMConfig')
    def test_get_tool_calls_parser(self, mock_llmconfig, mock_get_tokenizer):
        """Verify get_tool_call_parser returns correct string."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        router = DeepseekV3Router(
            config_dict=self.mock_config_dict,
            load_config=self.load_config
        )
        self.assertEqual(router._get_tool_calls_parser(), "deepseek_v3")


if __name__ == "__main__":
    unittest.main()