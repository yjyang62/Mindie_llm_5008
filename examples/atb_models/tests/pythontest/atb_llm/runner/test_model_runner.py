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
from unittest.mock import patch

import pandas as pd
import torch
import torch_npu

from atb_llm.runner.model_runner import ModelRunner
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.models.base.model_utils import BaseModel
from atb_llm.utils.initial import NPUSocInfo
from tests.pythontest.atb_llm.models.base.test_router import FAKE_CONFIG_DICT, FakeTokenizer

dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8
}


class TestModelRunner(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.config = FAKE_CONFIG_DICT
        self.tensor_names = {}

    def setUp(self):
        load_atb_speed_patcher = patch('atb_llm.runner.model_runner.load_atb_speed', return_value=None)
        self.mock_load_atb_speed = load_atb_speed_patcher.start()
        
        tokenizer_patcher = patch('transformers.AutoTokenizer.from_pretrained', return_value=FakeTokenizer())
        self.mock_tokenizer = tokenizer_patcher.start()
        
        weight_files_patcher = patch("atb_llm.utils.weights.weight_files", return_value=[])
        self.mock_weight_files = weight_files_patcher.start()
        
        get_config_patcher = patch('transformers.configuration_utils.PretrainedConfig.get_config_dict')
        self.mock_get_config_dict = get_config_patcher.start()
        self.mock_get_config_dict.side_effect = self._get_config_dict_side_effect
        self.addCleanup(load_atb_speed_patcher.stop)
        self.addCleanup(tokenizer_patcher.stop)
        self.addCleanup(weight_files_patcher.stop)
        self.addCleanup(get_config_patcher.stop)
        
        self.model_runner = ModelRunner("", 0, 1)
    
    def tearDown(self):
        patch.stopall()

    def test_init(self):
        self.assertIsInstance(self.model_runner, ModelRunner)
    
    @patch('atb_llm.models.base.flash_causal_lm.load_atb_speed', return_value=None)   
    @patch("atb_llm.utils.weights.weight_files", return_value=[])
    def test_load_weights(self, _1, _2):
        self.model_runner.model_cls = FlashForCausalLM
        expected_soc_info = NPUSocInfo()
        expected_head_size = 16
        expected_num_heads = 32
        expected_num_kv_heads = 8
        expected_num_layers = 1
        self.model_runner.load_weights()
        self.assertEqual(self.model_runner.soc_info, expected_soc_info)
        self.assertEqual(self.model_runner.head_size, expected_head_size)
        self.assertEqual(self.model_runner.num_heads, expected_num_heads)
        self.assertEqual(self.model_runner.num_kv_heads, expected_num_kv_heads)
        self.assertEqual(self.model_runner.num_layers, expected_num_layers)

    @patch("atb_llm.models.base.model_utils.BaseModel.save_pretrained", return_value=None)
    def test_save_pretrained(self, mock_save_pretrained):
        test_kwargs = {'save_directory': ' ', 'a': 1, 'b': 2, 'c': 3}
        self.model_runner.model = BaseModel()
        self.model_runner.save_pretrained(**test_kwargs)
        test_kwargs['save_directory'] += '/part0-of-1'
        mock_save_pretrained.assert_called_once()
        mock_save_pretrained.assert_called_with(**test_kwargs)
        call_kwargs = mock_save_pretrained.call_args.kwargs
        self.assertEqual(call_kwargs, test_kwargs)
  
    def parse_tensor_info(self, csv_path):
        data = pd.read_csv(csv_path, sep='|')
        for i in range(len(data)):
            name = data.iloc[i, 1].strip()
            self.tensor_names[name] = {}
            self.tensor_names[name]["shape"] = list(map(int, data.iloc[i, 2].split(',')))
            self.tensor_names[name]["dtype"] = dtype_map.get(data.iloc[i, 3])

    def _get_config_dict_side_effect(self, model_path, **kwargs):
        kwargs.pop("cache_dir", None)
        kwargs.pop("force_download", False)
        kwargs.pop("resume_download", False)
        kwargs.pop("proxies", None)
        kwargs.pop("local_files_only", False)
        kwargs.pop("revision", None)
        
        return self.config, kwargs
    
if __name__ == '__main__':
    unittest.main()