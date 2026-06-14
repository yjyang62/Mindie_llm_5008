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
import sys


class MockContext:
    model_path = 'dummy_model'
    save_directory = 'dummy_save'
    calib_file = 'dummy_calib.jsonl'
    w_bit = 8
    a_bit = 8
    device_type = 'cpu'
    anti_method = 'm3'
    disable_names = []
    act_method = 1
    w_sym = True
    co_sparse = False
    fraction = 0.01
    sigma_factor = 3.5
    use_sigma = False
    is_lowbit = True
    do_smooth = True
    open_outlier = True
    group_size = 0
    use_kvcache_quant = False
    is_dynamic = True
    disable_last_linear = True
    use_fa_quant = False
    fa_amp = 0
    tokenizer_args = "{}"
    trust_remote_code = False


class TestQuantPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.AntiOutlier'] = MagicMock()
        sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.AntiOutlierConfig'] = MagicMock()
        sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.Calibrator'] = MagicMock()
        sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.QuantConfig'] = MagicMock()

    @classmethod
    def tearDownClass(cls):
        del sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.AntiOutlier']
        del sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.AntiOutlierConfig']
        del sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.Calibrator']
        del sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.QuantConfig']

    @patch('examples.convert.model_slim.quantifier.QuantConfig')
    @patch('examples.convert.model_slim.quantifier.Quantifier')
    @patch('examples.convert.model_slim.quantifier.model_utils.safe_get_config_from_pretrained')
    @patch('examples.convert.model_slim.quantifier.model_utils.safe_get_model_from_pretrained')
    @patch('examples.convert.model_slim.quantifier.model_utils.safe_get_tokenizer_from_pretrained')
    def setUp(self, mock_tokenizer, mock_model, mock_config, mock_quant_config, mock_quantifier):
        from examples.convert.model_slim.quantifier import QuantPipeline
        context = MockContext()
        self.mock_config = MagicMock()
        mock_config.return_value = self.mock_config
        self.mock_model = MagicMock()
        mock_model.return_value = self.mock_model
        self.mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        self.pipeline = QuantPipeline(context)

    @patch('examples.convert.model_slim.quantifier.QuantConfig')
    def test_quant_conf_creation(self, mock_quant_config):
        _ = self.pipeline.get_quant_conf()
        mock_quant_config.assert_called_once_with(
            w_bit=8, a_bit=8, disable_names=[], dev_type='cpu', dev_id=0, act_method=1, w_sym=True,
            mm_tensor=False, co_sparse=False, fraction=0.01, sigma_factor=3.5, use_sigma=False,
            is_lowbit=True, do_smooth=True, open_outlier=True, group_size=0, use_kvcache_quant=False,
            is_dynamic=True, disable_last_linear=True
        )


if __name__ == '__main__':
    unittest.main()
