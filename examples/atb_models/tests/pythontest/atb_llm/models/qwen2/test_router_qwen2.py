# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import MagicMock, patch, Mock
from atb_llm.models.qwen2.router_qwen2 import Qwen2Router
from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.models.qwen2.input_builder_qwen2 import Qwen2InputBuilder


FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_TOKENIZER_PATH = "fake_tokenizer_path"
FAKE_CONFIG_DICT_BASE = {
    "model_type": "qwen2",
    "transformers_version": "4.41.2",
    "vocab_size": 151936,
    "hidden_size": 4096,
    "intermediate_size": 22016,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "max_position_embeddings": 32768,
}


class MockTokenizer:
    def __init__(self, **kwargs):
        self.padding_side = kwargs.get("padding_side", "left")
        self.trust_remote_code = kwargs.get("trust_remote_code", False)


class MockConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class TestQwen2Router(unittest.TestCase):
    """Test cases for Qwen2Router."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_dict = FAKE_CONFIG_DICT_BASE.copy()

    def test_post_init(self):
        """Test __post_init__ method."""
        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=self.config_dict
        )

        self.assertEqual(router.transformers_version, "4.41.2")
        self.assertTrue(router.prealloc_weight_mem_on_npu)

    def test_embedding_model_name_with_gte(self):
        """Test embedding_model_name property with gte model."""
        config_dict = self.config_dict.copy()
        config_dict["auto_map"] = {
            "AutoModelForSequenceClassification": "some_path"
        }

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertEqual(router.embedding_model_name, "gte")

    def test_embedding_model_name_without_gte(self):
        """Test embedding_model_name property without gte model."""
        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=self.config_dict
        )

        self.assertEqual(router.embedding_model_name, "")

    def test_is_qwen1_5_or_2_qwen1_5_0_5b(self):
        """Test is_qwen1_5_or_2 with qwen1.5_0.5b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 2816

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen1_5_1_8b(self):
        """Test is_qwen1_5_or_2 with qwen1.5_1.8b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 5504

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen1_5_4b(self):
        """Test is_qwen1_5_or_2 with qwen1.5_4b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 6912

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen1_5_7b(self):
        """Test is_qwen1_5_or_2 with qwen1.5_7b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 11008

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen1_5_14b(self):
        """Test is_qwen1_5_or_2 with qwen1.5_14b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 13696

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen1_5_32b(self):
        """Test is_qwen1_5_or_2 with qwen1.5_32b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 27392

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen1_5_72b(self):
        """Test is_qwen1_5_or_2 with qwen1.5_72b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 24576

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen1_5_110b(self):
        """Test is_qwen1_5_or_2 with qwen1.5_110b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 49152

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen2_0_5b(self):
        """Test is_qwen1_5_or_2 with qwen2_0.5b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 4864
        config_dict["max_window_layers"] = 24

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen2_0_5b_wrong_max_window_layers(self):
        """Test is_qwen1_5_or_2 with qwen2_0.5b but wrong max_window_layers."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 4864
        config_dict["max_window_layers"] = 28  # Wrong value

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertFalse(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen2_1_5b(self):
        """Test is_qwen1_5_or_2 with qwen2_1.5b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 8960
        config_dict["max_window_layers"] = 28

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen2_7b(self):
        """Test is_qwen1_5_or_2 with qwen2_7b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 18944
        config_dict["transformers_version"] = "4.41.2"

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen2_7b_wrong_transformers_version(self):
        """Test is_qwen1_5_or_2 with qwen2_7b but wrong transformers_version."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 18944
        config_dict["transformers_version"] = "4.50.0"  # Wrong version

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertFalse(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_qwen2_72b(self):
        """Test is_qwen1_5_or_2 with qwen2_72b."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 29568
        config_dict["max_window_layers"] = 80

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertTrue(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_false(self):
        """Test is_qwen1_5_or_2 returns False for non-matching config."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 10000  # Not matching any known size

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertFalse(router.is_qwen1_5_or_2)

    def test_is_qwen1_5_or_2_wrong_model_type(self):
        """Test is_qwen1_5_or_2 with wrong model_type."""
        config_dict = self.config_dict.copy()
        config_dict["model_type"] = "llama"  # Wrong model type
        config_dict["intermediate_size"] = 2816

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        self.assertFalse(router.is_qwen1_5_or_2)

    @patch('atb_llm.models.qwen2.router_qwen2.Qwen2Config')
    def test_get_config(self, mock_config_class):
        """Test get_config method."""
        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=self.config_dict
        )

        mock_config = MockConfig()
        mock_config_class.from_dict.return_value = mock_config

        with patch.object(router, 'checkout_config_qwen') as mock_checkout:
            config = router.get_config()

            # Verify config_dict was updated with transformers_version
            self.assertIn("transformers_version", router.config_dict)
            # Verify Qwen2Config.from_dict was called
            mock_config_class.from_dict.assert_called_once()
            # Verify checkout_config_qwen was called
            mock_checkout.assert_called_once_with(mock_config)
            self.assertEqual(config, mock_config)

    @patch('atb_llm.models.qwen2.router_qwen2.safe_get_tokenizer_from_pretrained')
    def test_get_tokenizer(self, mock_get_tokenizer):
        """Test get_tokenizer method."""
        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=self.config_dict,
            tokenizer_path=FAKE_TOKENIZER_PATH,
            trust_remote_code=True
        )

        mock_tokenizer = MockTokenizer()
        mock_get_tokenizer.return_value = mock_tokenizer

        tokenizer = router.get_tokenizer()

        mock_get_tokenizer.assert_called_once_with(
            FAKE_TOKENIZER_PATH,
            padding_side='left',
            trust_remote_code=True
        )
        self.assertEqual(tokenizer, mock_tokenizer)

    @patch('atb_llm.models.qwen2.router_qwen2.Qwen2InputBuilder')
    def test_get_input_builder_with_custom_chat_template(self, mock_input_builder_class):
        """Test get_input_builder with custom_chat_template."""
        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=self.config_dict
        )
        router._tokenizer = MockTokenizer()
        router._config = MockConfig(max_position_embeddings=32768)
        router._custom_chat_template = "custom_template"

        mock_input_builder = MagicMock()
        mock_input_builder_class.return_value = mock_input_builder

        input_builder = router.get_input_builder()

        mock_input_builder_class.assert_called_once_with(
            router.tokenizer,
            is_qwen1_5_or_2=False,
            chat_template="custom_template",
            max_length=32768
        )
        self.assertEqual(input_builder, mock_input_builder)

    @patch('atb_llm.models.qwen2.router_qwen2.Qwen2InputBuilder')
    def test_get_input_builder_without_max_position_embeddings(self, mock_input_builder_class):
        """Test get_input_builder without max_position_embeddings."""
        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=self.config_dict
        )
        router._tokenizer = MockTokenizer()
        router._config = MockConfig()  # No max_position_embeddings

        mock_input_builder = MagicMock()
        mock_input_builder_class.return_value = mock_input_builder

        input_builder = router.get_input_builder()

        # Should not include max_length in kwargs
        call_kwargs = mock_input_builder_class.call_args[1]
        self.assertNotIn("max_length", call_kwargs)
        self.assertEqual(input_builder, mock_input_builder)

    def test_get_tool_call_parser_qwen1_5_or_2(self):
        """Test get_tool_call_parser when is_qwen1_5_or_2 is True."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 2816  # qwen1.5_0.5b

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        parser = router.get_tool_call_parser()

        self.assertEqual(parser, "qwen1_5_or_2")

    def test_get_tool_call_parser_qwen2_5(self):
        """Test get_tool_call_parser when is_qwen1_5_or_2 is False."""
        config_dict = self.config_dict.copy()
        config_dict["intermediate_size"] = 10000  # Not matching

        router = Qwen2Router(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=config_dict
        )

        parser = router.get_tool_call_parser()

        self.assertEqual(parser, "qwen2_5")


if __name__ == '__main__':
    unittest.main()
