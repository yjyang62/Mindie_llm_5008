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
from unittest.mock import patch
from mindie_llm.runtime.config.load_config import LoadConfig


class TestLoadConfig(unittest.TestCase):

    def test_post_init_tokenizer_path_default(self):
        config = LoadConfig(model_name_or_path="/path/to/model")
        self.assertEqual(config.tokenizer_path, "/path/to/model")

    def test_post_init_tokenizer_path_custom(self):
        config = LoadConfig(model_name_or_path="/path/to/model", tokenizer_path="/custom/tokenizer")
        self.assertEqual(config.tokenizer_path, "/custom/tokenizer")

    def test_from_dict_creates_instance(self):
        config_dict = {
            "model_name_or_path": "/path/to/model",
            "max_position_embeddings": 4096,
            "trust_remote_code": True,
            "load_tokenizer": False,
            "tokenizer_path": "/custom/tokenizer",
            "llm_config_path": "/config.json",
            "models_dict": {"key": "value"}
        }
        config = LoadConfig.from_dict(config_dict)
        self.assertEqual(config.model_name_or_path, "/path/to/model")
        self.assertEqual(config.max_position_embeddings, 4096)
        self.assertTrue(config.trust_remote_code)
        self.assertFalse(config.load_tokenizer)
        self.assertEqual(config.tokenizer_path, "/custom/tokenizer")
        self.assertEqual(config.llm_config_path, "/config.json")
        self.assertEqual(config.models_dict, {"key": "value"})

    def test_from_dict_ignores_invalid_fields(self):
        config_dict = {
            "model_name_or_path": "/path/to/model",
            "invalid_field": "should_be_ignored",
            "max_position_embeddings": 4096
        }
        config = LoadConfig.from_dict(config_dict)
        self.assertEqual(config.model_name_or_path, "/path/to/model")
        self.assertEqual(config.max_position_embeddings, 4096)
        self.assertFalse(hasattr(config, "invalid_field"))

    def test_validate_max_position_embeddings_valid(self):
        config = LoadConfig(model_name_or_path="model", max_position_embeddings=4096)
        config.validate()  # Should not raise exception

    def test_validate_max_position_embeddings_invalid(self):
        config = LoadConfig(model_name_or_path="model", max_position_embeddings=-1)
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("max_position_embeddings must be a positive integer", str(cm.exception))

    def test_validate_max_position_embeddings_zero(self):
        config = LoadConfig(model_name_or_path="model", max_position_embeddings=0)
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("max_position_embeddings must be a positive integer", str(cm.exception))

    def test_validate_models_dict_valid(self):
        config = LoadConfig(model_name_or_path="model", models_dict={str(i): i for i in range(4096)})
        config.validate()  # Should not raise exception

    def test_validate_models_dict_invalid_too_long(self):
        config = LoadConfig(model_name_or_path="model", models_dict={str(i): i for i in range(4097)})
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("The length of plugin_params (4097) is too long", str(cm.exception))

    def test_validate_models_dict_none(self):
        config = LoadConfig(model_name_or_path="model", models_dict=None)
        config.validate()  # Should not raise exception

    def test_validate_models_dict_empty(self):
        config = LoadConfig(model_name_or_path="model", models_dict={})
        config.validate()  # Should not raise exception


if __name__ == '__main__':
    unittest.main()
