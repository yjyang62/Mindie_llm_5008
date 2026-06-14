# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
import argparse
import sys

from ddt import ddt


@ddt
class TestQuantifier(unittest.TestCase):
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

    @patch('examples.convert.model_slim.quantifier.model_utils.safe_get_config_from_pretrained')
    @patch('examples.convert.model_slim.quantifier.model_utils.safe_get_model_from_pretrained')
    @patch('examples.convert.model_slim.quantifier.model_utils.safe_get_tokenizer_from_pretrained')
    def setUp(self, mock_tokenizer, mock_model, mock_config):
        from examples.convert.model_slim.quantifier import Quantifier
        # Mock config, model, and tokenizer
        self.mock_config = MagicMock()
        mock_config.return_value = self.mock_config
        self.mock_model = MagicMock()
        mock_model.return_value = self.mock_model
        self.mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer

        self.quantifier = Quantifier(model_path_or_name="dummy_model",
                                     device_type='cpu',
                                     trust_remote_code=False,
                                     tokenizer_args={})

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory'])
    def test_default_value(self):
        from examples.convert.model_slim.quantifier import parse_arguments, CPU
        args = parse_arguments()
        self.assertIsNone(args.part_file_size)
        self.assertEqual(args.w_bit, 8)
        self.assertEqual(args.a_bit, 8)
        self.assertIsNone(args.disable_names)
        self.assertEqual(args.device_type, CPU)
        self.assertEqual(args.fraction, 0.01)
        self.assertEqual(args.act_method, 1)
        self.assertFalse(args.co_sparse, False)
        self.assertEqual(args.anti_method, "")
        self.assertEqual(args.disable_level, "L0")
        self.assertFalse(args.do_smooth, False)
        self.assertFalse(args.use_sigma, False)
        self.assertFalse(args.use_reduce_quant, False)
        self.assertEqual(args.tp_size, 1)
        self.assertEqual(args.sigma_factor, 3.0)
        self.assertFalse(args.is_lowbit, False)
        self.assertTrue(args.mm_tensor, True)
        self.assertTrue(args.w_sym, True)
        self.assertFalse(args.use_kvcache_quant, False)
        self.assertFalse(args.use_fa_quant, False)
        self.assertEqual(args.fa_amp, 0)
        self.assertTrue(args.open_outlier, True)
        self.assertEqual(args.group_size, 64)
        self.assertFalse(args.is_dynamic, False)
        self.assertEqual(args.input_ids_name, 'input_ids')
        self.assertEqual(args.attention_mask_name, 'attention_mask')
        self.assertEqual(args.tokenizer_args, '{}')
        self.assertTrue(args.disable_last_linear, True)

    @patch('sys.argv', ['test_script.py', '--model_path', '', '--save_directory', 'fake_save_directory'])
    def test_model_path_too_short(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'A' * 4097, '--save_directory', 'fake_save_directory'])
    def test_model_path_too_long(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', ''])
    def test_save_directory_too_short(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'A' * 4097])
    def test_save_directory_too_long(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--input_ids_name', ''])
    def test_input_ids_name_too_short(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--input_ids_name', 'A' * 257])
    def test_input_ids_name_too_long(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--attention_mask_name', ''])
    def test_attention_mask_name_too_short(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--attention_mask_name', 'A' * 257])
    def test_attention_mask_name_too_long(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--tokenizer_args', 'A'])
    def test_tokenizer_args_too_short(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--tokenizer_args', 'A' * 4097])
    def test_tokenizer_args_too_long(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--input_ids_name', 'A', '--attention_mask_name', 'B',
                        '--tokenizer_args', '{"pad_token_id": 0}'])
    def test_valid_param(self):
        from examples.convert.model_slim.quantifier import parse_arguments
        args = parse_arguments()
        self.assertEqual(args.model_path, 'fake_model_path')
        self.assertEqual(args.save_directory, 'fake_save_directory')
        self.assertEqual(args.input_ids_name, 'A')
        self.assertEqual(args.attention_mask_name, 'B')
        self.assertEqual(args.tokenizer_args, '{"pad_token_id": 0}')

    def test_initialization(self):
        self.assertEqual(self.quantifier.model_path_or_name, 'dummy_model')
        self.assertEqual(self.quantifier.device_type, 'cpu')

    @patch('examples.convert.model_slim.quantifier.Calibrator')
    @patch('examples.convert.model_slim.quantifier.AntiOutlier')
    def test_convert(self, mock_anti_outlier, mock_calibrator):
        tokenized_data = [[[MagicMock(), MagicMock()]]]
        save_path = "dummy_save"
        disable_level = "L0"

        self.quantifier.convert(tokenized_data, save_path, disable_level)
        mock_calibrator.assert_called_once_with(self.quantifier.model, self.quantifier.quant_config,
                                                calib_data=tokenized_data, disable_level=disable_level)
        mock_calibrator.return_value.run.assert_called_once()
        mock_calibrator.return_value.save.assert_called_once_with(save_path, save_type=["ascendV1"], part_file_size=None)


if __name__ == '__main__':
    unittest.main()