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
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase
from mindie_llm.runtime.utils.npu.device_utils import _NPUNodeInfo
from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig


class TestMindIELLMConfig(unittest.TestCase):
    def setUp(self):
        self.hf_config = MagicMock()
        self.llm_config = MagicMock()
        self.generation_config = MagicMock()

    @patch("mindie_llm.runtime.utils.npu.device_utils._NPUNodeInfo")
    @patch("mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config.QuantizationConfig")
    @patch("mindie_llm.runtime.utils.helpers.safety.file.safe_open", new_callable=mock_open, \
        read_data='{"quant_method": "ms_model_slim", "weight_bits": 8}')
    def test_init_quant_config_success(self, mock_safe_open, mock_quant_cls, mock_platform_info):
        mock_platform_info.return_value = MagicMock()
        mock_quant_instance = MagicMock(spec=QuantizationConfigBase)
        mock_quant_cls.from_config.return_value = mock_quant_instance
        mock_quant_cls.get_config_filenames.return_value = ["quant_config.json"]

        model_path = "/fake/model/path"
        with patch("pathlib.Path.exists", return_value=True), \
            patch("pathlib.Path.is_dir", return_value=True), \
            patch("pathlib.Path.glob", return_value=[Path(model_path) / "quant_config.json"]):

            config = MindIELLMConfig(
                model_name_or_path=model_path,
                hf_config=self.hf_config,
                llm_config=self.llm_config,
                generation_config=self.generation_config
            )

    @patch("mindie_llm.runtime.utils.npu.device_utils._NPUNodeInfo")
    @patch("mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config.QuantizationConfig")
    def test_init_quant_config_no_files(self, mock_quant_cls, mock_platform_info):
        mock_platform_info.return_value = MagicMock()
        mock_quant_cls.get_config_filenames.return_value = ["quant_config.json"]

        model_path = "/fake/model/path"
        with patch("pathlib.Path.exists", return_value=True), \
            patch("pathlib.Path.is_dir", return_value=True), \
            patch("pathlib.Path.glob", return_value=[]):

            config = MindIELLMConfig(
                model_name_or_path=model_path,
                hf_config=self.hf_config,
                llm_config=self.llm_config,
                generation_config=self.generation_config
            )

            self.assertIsNone(config.quant_config)

    @patch("mindie_llm.runtime.utils.npu.device_utils._NPUNodeInfo")
    @patch("mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config.QuantizationConfig")
    def test_init_quant_config_model_path_is_not_dir(self, mock_quant_cls, mock_platform_info):
        mock_platform_info.return_value = MagicMock()
        mock_quant_cls.get_config_filenames.return_value = ["quant_config.json"]

        model_path = "some_model_name"
        with patch("pathlib.Path.exists", return_value=False):
            config = MindIELLMConfig(
                model_name_or_path=model_path,
                hf_config=self.hf_config,
                llm_config=self.llm_config,
                generation_config=self.generation_config
            )

            self.assertIsNone(config.quant_config)

    def test_attributes_assignment(self):
        config = MindIELLMConfig(
            model_name_or_path="test-model",
            hf_config=self.hf_config,
            llm_config=self.llm_config,
            generation_config=self.generation_config,
            quant_config=None
        )
        self.assertEqual(config.model_name_or_path, "test-model")
        self.assertIs(config.hf_config, self.hf_config)
        self.assertIs(config.llm_config, self.llm_config)
        self.assertIs(config.generation_config, self.generation_config)
        self.assertIsNone(config.quant_config)


if __name__ == '__main__':
    unittest.main()