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
import unittest
from unittest.mock import patch
from ddt import ddt, data, unpack
from atb_llm.models.base.config import (
    RopeScaling, QuantizationConfig, BaseConfig, 
    LoraConfig, ReasoningConfig, GenerationConfig, LoraModelConfig)
from atb_llm.models.base.model_utils import EXTRA_EXP_INFO

FAKE_INVALID_STR = '42'     # invalid str

FAKE_ROPE_SCALING_CONFIG = {
    'factor': 1.0,
    'type': 'linear',
    'long_factor': [1, 2, 3],
    'short_factor': [0, 1, 2],
    'low_freq_factor': 1.0,
    'high_freq_factor': 4.0,
    'original_max_position_embeddings': 2048,
    'rope_type': 'llama3',
    'rope_theta': 1,
    'attention_factor': 1,
    'beta_fast': 32,
    'beta_slow': 1,
    'mrope_section': [8, 12, 12],
    'mrope_interleaved': False
}

FAKE_QUANTIZATION_CONFIG = {
    'group_size': 64,
    'kv_quant_type': 'C8',
    'fa_quant_type': 'FAQuant',
    'reduce_quant_type': 'per_channel'
}

FAKE_REASONING_CONFIG = {
    'start_reasoning_token_id': 1000,
    'end_reasoning_token_id': 1001
}

FAKE_GENERATION_CONFIG = {
    'max_new_tokens': 2048,
    'pad_token_id': 0,
    'eos_token_id': 2
}

FAKE_CONFIG = {
    'rope_scaling': FAKE_ROPE_SCALING_CONFIG,
    'quantization_config': FAKE_QUANTIZATION_CONFIG,
    'reasoning_config': FAKE_REASONING_CONFIG,
    'generation_config': FAKE_GENERATION_CONFIG,
    'max_position_embeddings': 2048,
    'vocab_size': 65536
}


class TestRopeScaling(unittest.TestCase):
    def test_to_dict(self):
        rope_scaling = RopeScaling(**FAKE_ROPE_SCALING_CONFIG)
        actual_return = rope_scaling.to_dict()
        self.assertEqual(FAKE_ROPE_SCALING_CONFIG, actual_return)


class TestQuantizationConfig(unittest.TestCase):
    def test_to_dict(self):
        quantization_config = QuantizationConfig(**FAKE_QUANTIZATION_CONFIG)
        actual_return = quantization_config.to_dict()
        expected_quantization_config_dict = copy.deepcopy(FAKE_QUANTIZATION_CONFIG)
        expected_quantization_config_dict['group_size'] = '64'
        self.assertEqual(expected_quantization_config_dict, actual_return)


class TestReasoningConfig(unittest.TestCase):
    def test_init_with_valid_values(self):
        reasoning_config = ReasoningConfig(
            start_reasoning_token_id=1000,
            end_reasoning_token_id=1001
        )
        self.assertEqual(reasoning_config.start_reasoning_token_id, 1000)
        self.assertEqual(reasoning_config.end_reasoning_token_id, 1001)

    def test_init_with_none_values(self):
        reasoning_config = ReasoningConfig()
        self.assertIsNone(reasoning_config.start_reasoning_token_id)
        self.assertIsNone(reasoning_config.end_reasoning_token_id)

    def test_init_with_partial_none(self):
        reasoning_config = ReasoningConfig(
            start_reasoning_token_id=1000,
            end_reasoning_token_id=None
        )
        self.assertEqual(reasoning_config.start_reasoning_token_id, 1000)
        self.assertIsNone(reasoning_config.end_reasoning_token_id)


class TestGenerationConfig(unittest.TestCase):
    def test_init_with_valid_values(self):
        generation_config = GenerationConfig(
            max_new_tokens=2048,
            pad_token_id=0,
            eos_token_id=2
        )
        self.assertEqual(generation_config.max_new_tokens, 2048)
        self.assertEqual(generation_config.pad_token_id, 0)
        self.assertEqual(generation_config.eos_token_id, 2)

    def test_init_with_none_values(self):
        generation_config = GenerationConfig()
        self.assertIsNone(generation_config.max_new_tokens)
        self.assertIsNone(generation_config.pad_token_id)
        self.assertIsNone(generation_config.eos_token_id)

    def test_init_with_list_eos_token_id(self):
        generation_config = GenerationConfig(
            max_new_tokens=2048,
            pad_token_id=0,
            eos_token_id=[2, 3]
        )
        self.assertEqual(generation_config.max_new_tokens, 2048)
        self.assertEqual(generation_config.pad_token_id, 0)
        self.assertEqual(generation_config.eos_token_id, [2, 3])

    def test_init_with_nested_list_eos_token_id(self):
        generation_config = GenerationConfig(
            max_new_tokens=2048,
            pad_token_id=0,
            eos_token_id=[[1, 2], [3, 4]]
        )
        self.assertEqual(generation_config.max_new_tokens, 2048)
        self.assertEqual(generation_config.pad_token_id, 0)
        self.assertEqual(generation_config.eos_token_id, [[1, 2], [3, 4]])


@ddt
class TestBaseConfig(unittest.TestCase):
    def setUp(self):
        self.config = None
        self.config = BaseConfig.from_dict(FAKE_CONFIG)
    
    @patch('atb_llm.utils.file_utils.standardize_path')
    @patch('atb_llm.utils.file_utils.check_path_permission')
    @patch('transformers.configuration_utils.PretrainedConfig.from_pretrained')
    def test_from_pretrained_success(self, mock_from_pretrained,
                                     mock_check_path_permission, mock_standardize_path):
        mock_standardize_path.return_value = ""
        mock_check_path_permission.return_value = "None"
        mock_from_pretrained.return_value = self.config
        actual_return = BaseConfig.from_pretrained('', **FAKE_CONFIG)
        self.assertEqual(actual_return, self.config)
    
    @patch('atb_llm.utils.file_utils.standardize_path')
    @patch('atb_llm.utils.file_utils.check_path_permission')
    @patch('transformers.configuration_utils.PretrainedConfig.from_pretrained')
    def test_from_pretrained_environment_error(self, mock_from_pretrained,
                                              mock_check_path_permission, mock_standardize_path):
        mock_standardize_path.return_value = ""
        mock_check_path_permission.return_value = "None"
        mock_from_pretrained.side_effect = EnvironmentError("Mock EnvironmentError")
        with self.assertRaises(EnvironmentError) as context:
            BaseConfig.from_pretrained('')
        error_message = str(context.exception)
        self.assertIn("Get config from pretrained failed. " + EXTRA_EXP_INFO, error_message)

    @patch('atb_llm.utils.file_utils.standardize_path')
    @patch('atb_llm.utils.file_utils.check_path_permission')
    @patch('transformers.configuration_utils.PretrainedConfig.from_pretrained')
    def test_from_pretrained_value_error(self, mock_from_pretrained,
                                              mock_check_path_permission, mock_standardize_path):
        mock_standardize_path.return_value = ""
        mock_check_path_permission.return_value = "None"
        mock_from_pretrained.side_effect = ValueError("Mock ValueError")
        with self.assertRaises(ValueError) as context:
            BaseConfig.from_pretrained('')
        error_message = str(context.exception)
        self.assertIn("Get config from pretrained failed. " + EXTRA_EXP_INFO, error_message)

    @data(('rope_scaling.original_max_position_embeddings', 0), ('vocab_size', 0),
          ('rope_scaling.type', FAKE_INVALID_STR),
          ('rope_scaling.type', FAKE_INVALID_STR), ('rope_scaling.short_factor', [-65536, 65536]),
          ('rope_scaling.long_factor', [-65536, 65536]), 
          ('rope_scaling.low_freq_factor', 0), ('rope_scaling.high_freq_factor', 0),
          ('rope_scaling.original_max_position_embeddings', 0), ('rope_scaling.rope_theta', 0),
          ('rope_scaling.attention_factor', 0), ('rope_scaling.beta_fast', 0),
          ('rope_scaling.beta_slow', 0))
    @unpack
    def test_validate_fail_value_error(self, mock_key, mock_value):
        keys = mock_key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = getattr(config, k, type('TempClass', (object,), {})())
        setattr(config, keys[-1], mock_value)
        with self.assertRaises(ValueError) as _:
            self.config.validate()

    @data(('quantization_config.group_size', 512),
          ('quantize', FAKE_INVALID_STR), ('quantization_config.kv_quant_type', FAKE_INVALID_STR),
          ('quantization_config.reduce_quant_type', FAKE_INVALID_STR), ('rope_scaling.rope_type', FAKE_INVALID_STR),
          ('quantization_config.fa_quant_type', FAKE_INVALID_STR))
    @unpack
    def test_validate_fail_not_implemented_error(self, mock_key, mock_value):
        keys = mock_key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = getattr(config, k, type('TempClass', (object,), {})())
        setattr(config, keys[-1], mock_value)

        with self.assertRaises(NotImplementedError) as _:
            self.config.validate()


@ddt
class TestLoraConfig(unittest.TestCase):
    def setUp(self):
        self.lora_config = LoraConfig.from_dict({})

    @data(
        ('r', 0),
        ('lora_alpha', 0),
        ('rank_pattern', FAKE_INVALID_STR),
        ('rank_pattern', {'1': 0}),
        ('alpha_pattern', {'1': 0}),
        ('target_modules', ['lm_head'])
        )
    @unpack
    def test_validate_fail(self, key, value):
        setattr(self.lora_config, key, value)
        with self.assertRaises(ValueError) as _:
            self.lora_config.validate()


@ddt
class TestLoraModelConfig(unittest.TestCase):
    @data(
        (1, -1), 
        (-1, 1)
        )
    @unpack
    def test_validate_fail(self, max_loras, max_lora_rank):
        with self.assertRaises(ValueError):
            self.lora_model_config = LoraModelConfig(max_loras=max_loras, max_lora_rank=max_lora_rank)
    
if __name__ == '__main__':
    unittest.main()