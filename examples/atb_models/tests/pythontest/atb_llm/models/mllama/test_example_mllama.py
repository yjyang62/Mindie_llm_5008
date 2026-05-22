# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

path = os.getenv('ATB_SPEED_HOME_PATH')
sys.path.append(os.path.join(path, 'lib'))

from atb_llm.models.mllama.config_mllama import MllamaTextConfig
from examples.models.mllama.mllama import MultiModalPARunner, MultiModalInput, MllamaRequest
from examples.models.mllama.mllama import parse_arguments, is_image


class TestFlashMLlamaForCausal(unittest.TestCase):
    def setUp(self):
        self.args_dict = dict(model_path='fake_model_path',
                           image_path='fake_image_path',
                           input_texts='fake_input_text',
                           input_ids=None,
                           prediction_result=False,
                           max_position_embeddings=None,
                           max_input_length=4096,
                           max_output_length=256,
                           max_prefill_tokens=-1,
                           max_batch_size=1,
                           max_image_num=1,
                           block_size=128,
                           ignore_eos=False,
                           rank=0,
                           world_size=1,
                           local_rank=0,
                           max_memory=100,
                           warm_up_memory=100,
                           max_num_tiles=4,
                           num_patches=1024,
            )
        self.model_config = MllamaTextConfig()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--image_path', 'fake_image_path'])
    def test_parse_arguments(self):
        args = parse_arguments()
        self.assertEqual(args.max_input_length, 4096)
        self.assertEqual(args.max_output_length, 256)
        self.assertEqual(args.max_prefill_tokens, -1)
        self.assertEqual(args.max_batch_size, 1)
        self.assertEqual(args.block_size, 128)

    @patch('examples.models.mllama.mllama.CacheManager')
    @patch('examples.models.mllama.mllama.print_log')
    @patch('examples.models.mllama.mllama.MultiModalPARunner.__init__', return_value=None)
    def test_mllama_run_pa_warm_up(self, mock_pa_runner, mock_log, mock_cache_manager):
        pa_runner = MultiModalPARunner()
        for key, value in self.args_dict.items():
            setattr(pa_runner, key, value)
        pa_runner.model_config = self.model_config
        pa_runner.input_attrs = MultiModalInput('fake_input', 'fake_image')
        pa_runner.processor = MagicMock()
        pa_runner.model = MagicMock()

        with self.assertRaises(FileNotFoundError):
            pa_runner.warm_up()

        self.assertEqual(mock_pa_runner.call_count, 1)
        self.assertEqual(mock_cache_manager.call_count, 1)
        mock_log.assert_called()
        
    @patch('examples.models.mllama.mllama.decode_token')
    @patch('examples.models.mllama.mllama.generate_req')
    @patch('examples.models.mllama.mllama.request_from_text_and_image_mllama')
    @patch('examples.models.mllama.mllama.print_log')
    @patch('examples.models.mllama.mllama.MultiModalPARunner.__init__', return_value=None)
    def test_mllama_run_pa_infer(self, mock_pa_runner, mock_log, mock_request, mock_generate_req, mock_decode_token):
        mock_decode_token.return_value = (['fake_text'], [0])

        pa_runner = MultiModalPARunner()
        for key, value in self.args_dict.items():
            setattr(pa_runner, key, value)
        pa_runner.model_config = self.model_config
        pa_runner.processor = MagicMock()
        pa_runner.model = MagicMock()    
        pa_runner.cache_manager = MagicMock()
        pa_runner.tokenizer = MagicMock()
        pa_runner.predict_result = False

        infer_params = {
            "inputs": MultiModalInput(['fake_input'],
                                    ['fake_image']),
            "batch_size": 1,
            "max_output_length": 256,
            "ignore_eos": False,
        }
        pa_runner.infer(**infer_params)
        self.assertEqual(mock_pa_runner.call_count, 1)
        self.assertEqual(mock_generate_req.call_count, 1)
        self.assertEqual(mock_decode_token.call_count, 1)
        mock_log.assert_called()
        mock_request.assert_called()

    def test_is_image(self):
        self.assertTrue(is_image('a.jpg'))
        self.assertFalse(is_image('b.xlsx'))
        self.assertFalse(is_image('fake_path'))
    
    def test_mllama_request(self):
        request = MllamaRequest(max_out_length=self.args_dict['max_output_length'],
                          block_size=self.args_dict['block_size'],
                          req_id=0,
                          input_ids=torch.tensor([1, 2, 3, 4, 5]),
                          multi_modal_inputs=None,
                          adapter_id=None)
        self.assertEqual(request.input_length, 5)


if __name__ == '__main__':
    unittest.main()