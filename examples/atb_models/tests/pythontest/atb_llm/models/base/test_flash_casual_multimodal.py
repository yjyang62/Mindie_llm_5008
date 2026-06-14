#!/usr/bin/env python
# coding=utf-8
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
from unittest.mock import patch, MagicMock
import pytest
from ddt import ddt, data, unpack

from atb_llm.models.base.flash_causal_multimodal import get_supported_models, get_llm_model
from atb_llm.models.base.flash_causal_multimodal import MultiModalConfig, MultiModalLLm


class TestGetSupportedModels(unittest.TestCase):
    @patch("atb_llm.models.base.flash_causal_multimodal.safe_listdir")
    @patch("atb_llm.models.base.flash_causal_multimodal.standardize_path")
    @patch("atb_llm.models.base.flash_causal_multimodal.check_file_safety")
    @patch("os.path.isdir")
    def test_get_supported_models(
        self, 
        mock_isdir, 
        mock_check_file_safety, 
        mock_standardize_path, 
        mock_safe_listdir
        ):

        mock_standardize_path.return_value = "/mocked/path"
        mock_safe_listdir.return_value = ["model1", "model2", "base", "_invalid"]
        mock_isdir.side_effect = lambda path: not path.endswith("_invalid")
        
        result = get_supported_models()
        
        self.assertEqual(result, ["model1", "model2"])
        mock_check_file_safety.assert_called_once_with("/mocked/path")
        mock_standardize_path.assert_called_once()
        mock_safe_listdir.assert_called_once_with("/mocked/path")
        self.assertEqual(mock_isdir.call_count, 4)


class TestGetLLMModel(unittest.TestCase):
    @patch("atb_llm.models.base.flash_causal_multimodal.importlib.import_module")
    def test_get_llm_model_valid(self, mock_import_module):

        mock_module = MagicMock()
        mock_import_module.return_value = mock_module
        mock_model_class = MagicMock()
        setattr(mock_module, "FlashQwen2audioForCausalLM", mock_model_class)
        get_llm_model("qwen2_audio")
        mock_import_module.assert_called_once_with("atb_llm.models.qwen2_audio.flash_causal_qwen2_audio")

    @patch("atb_llm.models.base.flash_causal_multimodal.get_supported_models")
    def test_get_llm_model_valid_2(self, mock_get_supported_models):

        mock_get_supported_models.return_value = ["qwen3_audio", "qwen2_audio", "qwen2_testaudio"]
        with pytest.raises(NotImplementedError):
            get_llm_model("llama_audio")
        with pytest.raises(ImportError):
            get_llm_model("qwen2_testaudio")


@ddt
class TestMultiModalConfig(unittest.TestCase):
    @data((None, "test_model", None),
    ({}, {"test_model": "test_config"}, {}),
    ({"model_type": "test_model"}, {"test_model_1": "test_config"}, {"model_type": "test_model"}))
    @unpack
    @patch("transformers.models.auto.CONFIG_MAPPING")
    def test_init_config(self, config, mock_config_mappping_return, return_value, mock_config_mappping):
        mock_config_mappping = MagicMock()
        mock_config_mappping.return_value = mock_config_mappping_return
        self.assertEqual(MultiModalConfig.init_config(config), return_value)

    @patch("atb_llm.models.base.flash_causal_multimodal.MultiModalConfig.init_config")
    def test_vision_config_initialization(self, mock_init_config):
        mock_init_config.return_value = {"mocked": "vision_config"}
        config = MultiModalConfig(vision_config={"model_type": "vision_model"})
        mock_init_config.assert_any_call({"model_type": "vision_model"})
        self.assertEqual(config.vision_config, {"mocked": "vision_config"})

    @patch("atb_llm.models.base.flash_causal_multimodal.MultiModalConfig.init_config")
    def test_text_config_initialization(self, mock_init_config):
        mock_init_config.return_value = {"mocked": "text_config"}
        config = MultiModalConfig(text_config={"model_type": "text_model"})
        mock_init_config.assert_any_call({"model_type": "text_model"})
        self.assertEqual(config.text_config, {"mocked": "text_config"})

    @patch("atb_llm.models.base.flash_causal_multimodal.MultiModalConfig.init_config")
    def test_audio_config_initialization(self, mock_init_config):
        mock_init_config.return_value = {"mocked": "audio_config"}
        config = MultiModalConfig(audio_config={"model_type": "audio_model"})
        mock_init_config.assert_any_call({"model_type": "audio_model"})
        self.assertEqual(config.audio_config, {"mocked": "audio_config"})

    @patch("atb_llm.models.base.flash_causal_multimodal.MultiModalConfig.init_config")
    def test_all_configs_initialization(self, mock_init_config):
        mock_init_config.side_effect = [
            {"mocked": "vision_config"},
            {"mocked": "text_config"},
            {"mocked": "audio_config"},
        ]

        config = MultiModalConfig(
            vision_config={"model_type": "vision_model"},
            text_config={"model_type": "text_model"},
            audio_config={"model_type": "audio_model"},
        )

        self.assertEqual(mock_init_config.call_count, 3)
        mock_init_config.assert_any_call({"model_type": "vision_model"})
        mock_init_config.assert_any_call({"model_type": "text_model"})
        mock_init_config.assert_any_call({"model_type": "audio_model"})
        
        self.assertEqual(config.vision_config, {"mocked": "vision_config"})
        self.assertEqual(config.text_config, {"mocked": "text_config"})
        self.assertEqual(config.audio_config, {"mocked": "audio_config"})


@ddt
class TestMultimodalLLM(unittest.TestCase):
    @data((True), (False))
    @patch("atb_llm.models.base.flash_causal_multimodal.FlashForCausalLM.__init__", return_value=None)
    @patch("atb_llm.models.base.flash_causal_multimodal.AutoModel.from_config")
    @patch("atb_llm.models.base.flash_causal_multimodal.get_llm_model")
    @patch("atb_llm.models.base.flash_causal_multimodal.safe_from_pretrained")
    def test_initialization(
        self,
        has_quantize,
        mock_safe_from_pretrained, 
        mock_get_llm_model, 
        mock_auto_model, 
        mock_flash_init
        ):

        mock_config = MagicMock()
        mock_config.quantize = has_quantize
        mock_config.text_config = MagicMock()
        mock_config.vision_config = MagicMock()
        mock_config.audio_config = None
        mock_config.model_type = "test_model"
        mock_weights = MagicMock()

        mock_llm_model_class = MagicMock()
        mock_get_llm_model.return_value = mock_llm_model_class
        MultiModalLLm(config=mock_config, weights=mock_weights)
        mock_flash_init.assert_called_once_with(mock_config.text_config, mock_weights)
        mock_get_llm_model.assert_called_once_with(mock_config.text_config.model_type)
        mock_llm_model_class.assert_called_once_with(
            mock_config.text_config,
            mock_weights,
            lmhead_prefix="language_model.lm_head",
            model_prefix="language_model.model"
        )
        mock_auto_model.assert_called_once_with(mock_config.vision_config)

    @patch("atb_llm.models.base.flash_causal_multimodal.FlashForCausalLM.__init__", return_value=None)
    @patch("atb_llm.models.base.flash_causal_multimodal.AutoModel.from_config")
    @patch("atb_llm.models.base.flash_causal_multimodal.get_llm_model")
    @patch("atb_llm.models.base.flash_causal_multimodal.safe_from_pretrained")
    def test_init_vit(self, mock_safe_from_pretrained, mock_get_llm_model, mock_auto_model, mock_flash_init):

        mock_config = MagicMock()
        mock_config.text_config = MagicMock()
        mock_config.vision_config = MagicMock()
        mock_config.audio_config = None
        mock_config.model_type = "test_model"
        mock_weights = MagicMock()

        mock_llm_model_class = MagicMock()
        mock_get_llm_model.return_value = mock_llm_model_class
        
        model = MultiModalLLm(config=mock_config, weights=mock_weights)
        
        model.init_vit()
        mock_auto_model.assert_any_call(mock_config.vision_config)

    @patch("atb_llm.models.base.flash_causal_multimodal.FlashForCausalLM.__init__", return_value=None)
    @patch("atb_llm.models.base.flash_causal_multimodal.AutoModel.from_config")
    @patch("atb_llm.models.base.flash_causal_multimodal.get_llm_model")
    @patch("atb_llm.models.base.flash_causal_multimodal.safe_from_pretrained")
    @patch("atb_llm.models.base.flash_causal_multimodal.MultiModalLLm.forward")
    def test_forward(
        self, 
        mock_forward, 
        mock_safe_from_pretrained, 
        mock_get_llm_model, 
        mock_auto_model, 
        mock_flash_init
        ):

        mock_config = MagicMock()
        mock_config.text_config = MagicMock()
        mock_config.vision_config = None
        mock_config.audio_config = MagicMock()
        mock_config.model_type = "test_model"
        mock_weights = MagicMock()

        mock_llm_model_class = MagicMock()
        mock_get_llm_model.return_value = mock_llm_model_class
        
        model = MultiModalLLm(config=mock_config, weights=mock_weights)
        
        mock_input_ids = MagicMock()
        mock_position_ids = MagicMock()
        mock_kv_cache = MagicMock()
        mock_block_tables = MagicMock()
        mock_slots = MagicMock()
        mock_input_lengths = MagicMock()
        mock_max_seq_len = MagicMock()
        mock_lm_head_indices = MagicMock()

        model.forward(
            input_ids=mock_input_ids,
            position_ids=mock_position_ids,
            is_prefill=True,
            kv_cache=mock_kv_cache,
            block_tables=mock_block_tables,
            slots=mock_slots,
            input_lengths=mock_input_lengths,
            max_seq_len=mock_max_seq_len,
            lm_head_indices=mock_lm_head_indices
        )

        mock_forward.assert_called_once_with(
            input_ids=mock_input_ids,
            position_ids=mock_position_ids,
            is_prefill=True,
            kv_cache=mock_kv_cache,
            block_tables=mock_block_tables,
            slots=mock_slots,
            input_lengths=mock_input_lengths,
            max_seq_len=mock_max_seq_len,
            lm_head_indices=mock_lm_head_indices
        )


if __name__ == "__main__":
    unittest.main()