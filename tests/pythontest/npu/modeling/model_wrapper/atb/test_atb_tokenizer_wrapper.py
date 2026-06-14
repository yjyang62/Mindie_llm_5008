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

from mindie_llm.modeling.model_wrapper.atb.atb_tokenizer_wrapper import ATBTokenizerWrapper

FAKE_MODEL_ID = "test"


class TestATBTokenizerWrapper(unittest.TestCase):
    @patch("mindie_llm.modeling.model_wrapper.atb.atb_tokenizer_wrapper.TokenizerWrapper")
    def test_init(self, mock_tokenizerwrapper):
        mock_tokenizerwrapper.return_value = MagicMock()
        mock_tokenizerwrapper.return_value.config = MagicMock()
        mock_tokenizerwrapper.return_value.tokenizer = MagicMock()
        mock_tokenizerwrapper.return_value.input_builder = MagicMock()
        mock_tokenizerwrapper.return_value.postprocessor = MagicMock()
        mock_tokenizerwrapper.return_value.config.vocab_size.return_value = 1000
        mock_tokenizerwrapper.return_value.tokenizer_wrapper = MagicMock()
        instance_atbtokenizerwrapper = ATBTokenizerWrapper(FAKE_MODEL_ID)

        self.assertEqual(instance_atbtokenizerwrapper.tokenizer_wrapper, mock_tokenizerwrapper())
        self.assertEqual(instance_atbtokenizerwrapper.config, mock_tokenizerwrapper().config)
        self.assertEqual(instance_atbtokenizerwrapper.tokenizer, mock_tokenizerwrapper().tokenizer)
        self.assertEqual(instance_atbtokenizerwrapper.input_builder, mock_tokenizerwrapper().input_builder)
        self.assertEqual(instance_atbtokenizerwrapper.postprocessor, mock_tokenizerwrapper().postprocessor)
        self.assertEqual(instance_atbtokenizerwrapper.vocab_size(), 1000)        
        self.assertEqual(instance_atbtokenizerwrapper.toolscallprocessor, mock_tokenizerwrapper().tool_calls_parser)

if __name__ == "__main__":
    unittest.main()