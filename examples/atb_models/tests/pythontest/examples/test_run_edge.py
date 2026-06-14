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
from unittest.mock import MagicMock, patch
import json

# Ensure we can import from examples
# Assuming the test is run from a location where examples module is available
# If not, we might need to adjust sys.path, but based on existing tests, we assume it works.
from examples.run_edge import OtherRunner, parse_arguments, cmd_bool, parse_ids, parse_list_of_json, input_texts_parser, ENV


class TestRunEdgeHelper(unittest.TestCase):
    """
    Test helper functions in run_edge.py
    """
    def test_cmd_bool(self):
        """Test cmd_bool function for True, False and invalid inputs"""
        self.assertTrue(cmd_bool("True"))
        self.assertFalse(cmd_bool("False"))
        with self.assertRaises(ValueError):
            cmd_bool("Invalid")

    def test_parse_list_of_json(self):
        """Test parsing list of json strings"""
        json_str = '[{"key": "value"}]'
        result = parse_list_of_json(json_str)
        self.assertEqual(result, [{"key": "value"}])
        
        with self.assertRaises(json.JSONDecodeError):
            parse_list_of_json("invalid json")

    def test_parse_ids(self):
        """Test parsing comma separated ids"""
        ids_str = "1,2,3"
        result = parse_ids(ids_str)
        self.assertEqual(result, [1, 2, 3])
        
        with self.assertRaises(ValueError):
            parse_ids("1,a,3")

    @patch("examples.run_edge.os.path.isfile")
    @patch("examples.run_edge.file_utils.safe_open")
    def test_input_texts_parser(self, mock_safe_open, mock_isfile):
        """Test input_texts_parser for file path and raw text"""
        # Case 1: File exists
        mock_isfile.return_value = True
        mock_f = MagicMock()
        mock_f.read.return_value = "file content"
        mock_safe_open.return_value.__enter__.return_value = mock_f
        self.assertEqual(input_texts_parser("dummy_path"), "file content")

        # Case 2: Not a file (raw text)
        mock_isfile.return_value = False
        self.assertEqual(input_texts_parser("raw text"), "raw text")


class TestOtherRunner(unittest.TestCase):
    """
    Test OtherRunner class in run_edge.py
    """
    def setUp(self):
        self.default_args = {
            'model_path': '/dummy/model/path',
            'input_text': 'Hello world',
            'max_prefill_batch_size': 1,
            'max_batch_size': 1,
            'max_input_length': 128,
            'max_output_length': 128,
            'max_position_embeddings': 512,
            'max_prefill_tokens': -1,
            'enable_atb_torch': False,
            'block_size': 128,
            'chat_template': None,
            'load_tokenizer': True,
            'rank': '0',
            'local_rank': '0',
            'world_size': '1',
            'kw_args': ''
        }

    @patch("examples.run_edge.ModelRunner")
    @patch("examples.run_edge.ModelConfig")
    def test_init(self, _, mock_model_runner):
        """Test initialization of OtherRunner"""
        # Setup mocks
        mock_instance = MagicMock()
        mock_model_runner.return_value = mock_instance
        mock_instance.tokenizer = MagicMock()
        
        # Test valid init
        runner = OtherRunner(**self.default_args)
        self.assertIsInstance(runner, OtherRunner)
        
        # Test kw_args parsing error
        invalid_args = self.default_args.copy()
        invalid_args['kw_args'] = '{invalid_json}'
        with self.assertRaises(ValueError):
            OtherRunner(**invalid_args)
            
        # Test kw_args too long
        long_args = self.default_args.copy()
        long_args['kw_args'] = "a" * 4097
        with self.assertRaises(ValueError):
            OtherRunner(**long_args)
            
        # Test kw_args valid
        valid_kw_args = self.default_args.copy()
        valid_kw_args['kw_args'] = '{"key": "value"}'
        runner = OtherRunner(**valid_kw_args)
        mock_instance.load_weights.assert_called_with(key="value")

    @patch("examples.run_edge.ModelRunner")
    @patch("examples.run_edge.ModelConfig")
    @patch("examples.run_edge.print_log")
    def test_check_limits(self, mock_model_runner, _, mock_print_log):
        """Test check_limits method for various boundary conditions"""
        mock_model_runner.return_value = MagicMock()
        
        # Helper to run check_limits with specific args
        def run_check(key, value):
            args = self.default_args.copy()
            args[key] = value
            # check_limits is called in __init__, so we just init
            OtherRunner(**args)

        # Test scenario when max_input_length is non-positive
        run_check('max_input_length', 0)
        self.assertTrue(mock_print_log.called)
        mock_print_log.reset_mock()
        
        # Test scenario when max_output_length is non-positive
        run_check('max_output_length', 0)
        self.assertTrue(mock_print_log.called)
        mock_print_log.reset_mock()

        # Test scenario when max_position_embeddings is insufficient for input plus output
        args = self.default_args.copy()
        args['max_input_length'] = 100
        args['max_output_length'] = 100
        args['max_position_embeddings'] = 150
        OtherRunner(**args)
        self.assertTrue(mock_print_log.called)
        mock_print_log.reset_mock()

        # Test scenario when max_prefill_batch_size is non-positive
        run_check('max_prefill_batch_size', -1)
        self.assertTrue(mock_print_log.called)
        mock_print_log.reset_mock()

        # Test scenario when max_batch_size is non-positive
        run_check('max_batch_size', 0)
        self.assertTrue(mock_print_log.called)

    def test_load_chat_template(self):
        """Test _load_chat_template method"""
        with patch("examples.run_edge.ModelRunner") as mock_model_runner, \
             patch("examples.run_edge.ModelConfig"), \
             patch("examples.run_edge.os.path.exists") as mock_exists, \
             patch("examples.run_edge.file_utils.safe_open") as mock_safe_open:
            mock_model_runner.return_value = MagicMock()
            runner = OtherRunner(**self.default_args)
            
            # Case 1: File exists
            mock_exists.return_value = True
            mock_f = MagicMock()
            mock_f.read.return_value = "template content"
            mock_safe_open.return_value.__enter__.return_value = mock_f
            
            result = runner._load_chat_template("path/to/template")
            self.assertEqual(result, "template content")
            
            # Case 2: File does not exist (raw string)
            mock_exists.return_value = False
            result = runner._load_chat_template("raw template string")
            self.assertEqual(result, "raw template string")

    def test_warm_up(self):
        """Test warm_up method"""
        with patch("examples.run_edge.ModelRunner") as mock_model_runner, \
             patch("examples.run_edge.ModelConfig"), \
             patch("examples.run_edge.torch") as mock_torch:
            mock_instance = MagicMock()
            mock_model_runner.return_value = mock_instance
            
            # Mock tokenizer behavior
            mock_tokenizer = MagicMock()
            mock_inputs = MagicMock()
            mock_inputs.input_ids = MagicMock()
            mock_inputs.input_ids.__getitem__.return_value.__len__.return_value = 10 # length
            mock_tokenizer.return_value = mock_inputs
            mock_instance.tokenizer = mock_tokenizer
            
            runner = OtherRunner(**self.default_args)
            runner.warm_up()
            
            # Verify calls
            mock_torch.ones.assert_called()
            mock_instance.forward.assert_called()

    def test_infer(self):
        """Test infer method"""
        with patch("examples.run_edge.rank", 0, create=True), \
             patch("examples.run_edge.ModelRunner") as mock_model_runner, \
             patch("examples.run_edge.ModelConfig"), \
             patch("examples.run_edge.torch") as mock_torch, \
             patch("examples.run_edge.time") as mock_time:
            mock_instance = MagicMock()
            mock_model_runner.return_value = mock_instance
            
            # Mock time to calculate average decode time
            # Mocking time for initial call and decode loop assuming max_output_length equals 2
            mock_time.time.side_effect = [100.0, 101.0, 102.0, 103.0]
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_inputs = MagicMock()
            mock_inputs.input_ids = MagicMock()
            # Mock length of input_ids[0]
            mock_inputs.input_ids.__getitem__.return_value = [1, 2, 3]
            
            mock_tokenizer.return_value = mock_inputs
            mock_tokenizer.decode.return_value = "Generated Response"
            mock_instance.tokenizer = mock_tokenizer
            
            # Mock torch operations
            mock_torch.arange.return_value = MagicMock()
            mock_torch.ones.return_value = MagicMock()
            mock_torch.Tensor.return_value = MagicMock()
            
            # Mock forward return values
            # The forward method is expected to return a tuple of logits and past_key_values
            mock_logits = MagicMock()
            # Mocking the reshape operation on the last token logits
            mock_reshaped = MagicMock()
            mock_logits.__getitem__.return_value.__getitem__.return_value.reshape.return_value = mock_reshaped
            
            # Mocking the argmax operation to find the most likely token
            mock_argmax = MagicMock()
            mock_reshaped.argmax.return_value = mock_argmax
            
            # Mocking the token tensor to be appended to output
            mock_token_tensor = MagicMock()
            mock_token_tensor.item.return_value = 123
            mock_argmax.__getitem__.return_value = mock_token_tensor # dummy token tensor
            
            # Mocking the reshape operation on logits for the next input iteration
            mock_argmax.reshape.return_value = MagicMock()
    
            mock_instance.forward.return_value = (mock_logits, MagicMock())
            
            runner = OtherRunner(**self.default_args)
            
            response, decode_times, avg_time = runner.infer("input text", max_output_length=2)
            
            self.assertEqual(response, "Generated Response")
            self.assertEqual(decode_times, 2)
            mock_torch.npu.synchronize.assert_called()

    @patch("examples.run_edge.ModelRunner")
    @patch("examples.run_edge.ModelConfig")
    def test_repr(self, _, mock_model_runner):
        """Test __repr__ method"""
        runner = OtherRunner(**self.default_args)
        repr_str = repr(runner)
        self.assertIn("OtherRunner", repr_str)
        self.assertIn("model_path", repr_str)
        self.assertIn("input_text", repr_str)


class TestArgParse(unittest.TestCase):
    """
    Test argument parsing
    """
    @patch("examples.run_edge.argparse.ArgumentParser")
    def test_parse_arguments(self, mock_parser_cls):
        """Test parse_arguments function"""
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse_args.return_value = MagicMock()
        
        args = parse_arguments()
        
        self.assertTrue(mock_parser.add_argument.called)
        # Verify some arguments are added
        call_args_list = mock_parser.add_argument.call_args_list
        arg_names = [args[0][0] for args in call_args_list]
        self.assertIn('--model_path', arg_names)
        self.assertIn('--input_texts', arg_names)