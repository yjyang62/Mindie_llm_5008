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
from unittest.mock import MagicMock, patch, mock_open
import torch

from atb_llm.utils.multimodal_utils import MultimodalRequestOut, RunReturns
from examples.multimodal_runner import MultimodalPARunner


def pa_init_side_effect(self, **kwargs):
    self.model_path = kwargs.get('model_path', '')
    self.rank = kwargs.get('rank', 0)
    self.world_size = kwargs.get('world_size', 1)
    self.local_rank = kwargs.get('local_rank', 0)
    self.model = MagicMock()
    self.model.tokenizer = MagicMock()
    self.model.soc_info.need_nz = False
    self.max_batch_size = kwargs.get('max_batch_size', 1)
    self.max_input_length = kwargs.get('max_input_length', 1024)
    self.max_output_length = kwargs.get('max_output_length', 128)
    self.block_size = kwargs.get('block_size', 128)
    self.dtype = torch.float16
    self.max_memory = 1024 * 1024 * 1024
    self.init_memory = 0
    self.model_config = MagicMock()
    self.tokenizer = MagicMock()
    return None


class TestMultimodalPARunner(unittest.TestCase):
    
    def setUp(self):
        self.kwargs = {
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'input_texts': ['test'],
            'image_path': ['img.jpg'],
            'video_path': None,
            'audio_path': None,
            'model_path': '/dummy/path',
            'max_batch_size': 1,
            'max_input_length': 1024,
            'max_output_length': 128,
            'block_size': 128,
            'trust_remote_code': True,
            'skip_special_tokens': True,
            'prediction_result': True,
            'performance': True,
            'max_prefill_tokens': -1,
            'perf_file': 'perf.csv',
            'pred_file': 'pred.json'
        }

    def test_init(self):
        with patch('examples.multimodal_runner.safe_from_pretrained') as mock_safe, \
             patch('examples.multimodal_runner.AutoProcessor'), \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True) as mock_pa_init:
            runner = MultimodalPARunner(**self.kwargs)
            mock_pa_init.assert_called_once()
            mock_safe.assert_called_once()
            self.assertTrue(runner.trust_remote_code)
            self.assertTrue(runner.skip_special_tokens)
            self.assertEqual(runner.max_prefill_tokens, -1)

    def test_init_processor_failure(self):
        with patch('examples.multimodal_runner.safe_from_pretrained') as mock_safe, \
             patch('examples.multimodal_runner.AutoProcessor'), \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True):
            mock_safe.side_effect = AssertionError
            runner = MultimodalPARunner(**self.kwargs)
            # Manually set model mock since PARunner init is mocked
            runner.model = MagicMock()
            runner.model.tokenizer = "tokenizer_fallback"
            runner.init_processor()
            self.assertEqual(runner.processor, "tokenizer_fallback")

    def test_prepare_request(self):
        with patch('examples.multimodal_runner.request_from_multimodalinputs') as mock_req_from_inputs, \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.model = MagicMock()
            runner.rank = 0
            runner.block_size = 128
            runner.adapter_id = None
            
            inputs = MagicMock()
            inputs.input_texts = ["t1"]
            inputs.image_path = ["i1"]
            inputs.video_path = ["v1"]
            inputs.audio_path = ["a1"]
            
            mock_req = MagicMock()
            mock_req.input_ids = [1, 2, 3]
            mock_req_from_inputs.return_value = mock_req
            
            res = runner.prepare_request(inputs, batch_size=1, max_output_length=10, current_iter=0)
            self.assertIsInstance(res, MultimodalRequestOut)
            self.assertEqual(res.batch, 1)

    def test_make_cache_manager(self):
        with patch('examples.multimodal_runner.CacheConfig'), \
             patch('examples.multimodal_runner.CacheManager') as mock_cache_manager, \
             patch('examples.multimodal_runner.ENV') as mock_env, \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.cache_manager = None
            runner.max_prefill_tokens = -1
            runner.max_batch_size = 1
            runner.max_input_length = 100
            runner.max_output_length = 100
            runner.block_size = 128
            runner.dtype = torch.float16
            runner.model = MagicMock()
            runner.model.num_kv_heads = 1
            runner.model.head_size = 1
            runner.model.num_layers = 1
            runner.max_memory = 10000000000
            runner.warm_up_memory = 0
            runner.init_memory = 0
            runner.rank = 0
            runner.model_config = MagicMock()
            
            mock_env.memory_fraction = 1.0
            mock_env.reserved_memory_gb = 0
            
            runner.make_cache_manager()
            mock_cache_manager.assert_called()
            self.assertIsNotNone(runner.cache_manager)

    def test_save_token_ids(self):
        with patch('examples.multimodal_runner.torch.save') as mock_torch_save, \
             patch('examples.multimodal_runner.file_utils') as mock_file_utils, \
             patch('examples.multimodal_runner.ENV') as mock_env, \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            mock_env.token_ids_save_folder = "/tmp"
            
            req = MagicMock()
            req.input_ids.cpu.return_value = "cpu_tensor"
            req.out_token_list = [1, 2]
            req_list = [req]
            
            mock_file_open = mock_open()
            mock_file_utils.safe_open = mock_file_open
            
            runner.save_token_ids(req_list)
            
            mock_torch_save.assert_called()
            mock_file_open.assert_called()

    def test_performance_save(self):
        with patch('examples.multimodal_runner.csv') as mock_csv, \
             patch('examples.multimodal_runner.file_utils') as mock_file_utils, \
             patch('examples.multimodal_runner.ENV') as mock_env, \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.local_rank = 0
            runner.perf_file = "perf.csv"
            
            mock_env.benchmark_filepath = None
            
            mock_csv_reader = MagicMock()
            # csv.reader returns an iterator. 
            # We need it to return at least 2 rows: header and second_row.
            # next(csv_reader) -> header
            # next(csv_reader) -> second_row: ["", "10", "", "", "5.0", "6.0"]
            mock_csv_reader.__next__.side_effect = [["header"], ["", "10", "", "", "5.0", "6.0"]] * 10
            mock_csv.reader.return_value = mock_csv_reader
    
            mock_csv_file = mock_open()
            
            file_mock = MagicMock()
            # Configure side_effect to cycle through the values indefinitely if needed, or provide enough values
            # Each call to performance_save calls safe_open twice: once for reading benchmark csv, once for writing perf file.
            # We call performance_save twice in this test.
            mock_file_utils.safe_open.side_effect = [mock_csv_file.return_value, file_mock] * 10
            mock_file_utils.standardize_path.return_value = "path"
            
            with patch('examples.multimodal_runner.os.path.exists', return_value=False):
                runner.performance_save(100, [[1, 10]], 1.0, 1)
                
            file_mock.write.assert_called()
    
            # Case 2: File exists
            file_mock.reset_mock()
            with patch('examples.multimodal_runner.os.path.exists', return_value=True):
                runner.performance_save(100, [[1, 10]], 1.0, 1)
            # write called once (data)
            self.assertEqual(file_mock.write.call_count, 1)

    def test_performance_test(self):
        with patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.tokenizer = MagicMock()
            runner.tokenizer.return_value = {"input_ids": MagicMock(flatten=MagicMock(return_value=MagicMock(shape=[5])))}
            
            inputs = MagicMock()
            inputs.all_input_texts = ["test"]
            inputs.all_token_num_list = [[1, 10]]
            inputs.e2e_time_all = 1.0
            inputs.batch = 1
            
            with patch.object(runner, 'performance_save') as mock_save:
                runner.performance_test(inputs)
                mock_save.assert_called()

    def test_precision_save(self):
        with patch('examples.multimodal_runner.file_utils') as mock_file_utils, \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.pred_file = "pred.json"
            
            inputs = MagicMock()
            
            # Case 1: New file
            with patch('examples.multimodal_runner.os.path.exists', return_value=False):
                with patch('json.dump') as mock_json_dump:
                    mock_file_utils.safe_open = mock_open()
                    runner.precision_save(inputs, answer_pairs={"a": 1})
                    mock_json_dump.assert_called()
    
            # Case 2: Existing file
            with patch('examples.multimodal_runner.os.path.exists', return_value=True):
                with patch('json.load', return_value={"b": 2}), patch('json.dump') as mock_json_dump:
                    mock_file_utils.safe_open = mock_open()
                    runner.precision_save(inputs, answer_pairs={"a": 1})
                    mock_json_dump.assert_called()

    def test_precision_save_error(self):
        with patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            with self.assertRaises(ValueError):
                runner.precision_save(MagicMock(), answer_pairs=None)

    def test_make_cache_manager_custom_tokens(self):
        with patch('examples.multimodal_runner.CacheConfig'), \
             patch('examples.multimodal_runner.CacheManager') as mock_cache_manager, \
             patch('examples.multimodal_runner.ENV') as mock_env, \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.cache_manager = None
            runner.max_prefill_tokens = 500
            runner.block_size = 128
            runner.dtype = torch.float16
            runner.model = MagicMock()
            runner.model.num_kv_heads = 1
            runner.model.head_size = 1
            runner.model.num_layers = 1
            runner.max_memory = 10000000000
            runner.warm_up_memory = 100
            runner.init_memory = 0
            runner.rank = 0
            runner.model_config = MagicMock()
            
            mock_env.memory_fraction = 1.0
            mock_env.reserved_memory_gb = 0
            
            runner.make_cache_manager()
            self.assertEqual(runner.max_prefill_tokens, 500)
            mock_cache_manager.assert_called()

    def test_profiling(self):
        with patch('examples.multimodal_runner.generate_req') as mock_gen_req, \
             patch('examples.multimodal_runner.torch_npu.profiler') as mock_profiler, \
             patch('examples.multimodal_runner.torch.npu.synchronize', create=True), \
             patch('examples.multimodal_runner.ENV') as mock_env, \
             patch('examples.multimodal_runner.file_utils'), \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.rank = 0
            runner.model = MagicMock()
            runner.cache_manager = MagicMock()
            runner.max_batch_size = 1
            
            mock_env.profiling_filepath = "prof_path"
            mock_profiler.profile.return_value.__enter__.return_value = None
            
            with patch.object(runner, 'prepare_request') as mock_prep:
                mock_req_out = MagicMock()
                mock_req_out.req_list = []
                mock_req_out.input_texts = []
                mock_prep.return_value = mock_req_out
                
                res = runner.profiling("inputs", 1, 10)
                self.assertIsInstance(res, RunReturns)
                mock_gen_req.assert_called()

    def test_run(self):
        with patch('examples.multimodal_runner.decode_token') as mock_decode, \
             patch('examples.multimodal_runner.generate_req') as mock_gen_req, \
             patch('examples.multimodal_runner.torch.npu.synchronize', create=True), \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.rank = 0
            runner.model = MagicMock()
            runner.model.postprocessor = MagicMock()
            runner.cache_manager = MagicMock()
            runner.tokenizer = MagicMock()
            runner.max_batch_size = 1
            
            mock_decode.return_value = ([], [])
            
            with patch.object(runner, 'prepare_request') as mock_prep:
                mock_req_out = MagicMock()
                mock_req_out.req_list = [MagicMock(input_ids=[])]
                mock_req_out.input_texts = []
                mock_prep.return_value = mock_req_out
                
                res = runner.run("inputs", 1, 10, ignore_eos=True, max_iters=1)
                
                mock_gen_req.assert_called()
                mock_decode.assert_called()
                self.assertEqual(runner.model.postprocessor.eos_token_id, [])

    def test_warm_up(self):
        with patch('examples.multimodal_runner.CacheManager'), \
             patch('examples.multimodal_runner.CacheConfig'), \
             patch('examples.multimodal_runner.generate_req') as mock_gen_req, \
             patch('examples.multimodal_runner.NpuHbmInfo') as mock_hbm, \
             patch('examples.multimodal_runner.request_from_multimodalinputs'), \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.rank = 0
            runner.warm_up_inputs = MagicMock()
            runner.warm_up_inputs.input_texts = ["t"]
            runner.warm_up_inputs.image_path = ["i"]
            runner.model = MagicMock()
            runner.model.postprocessor = MagicMock()
            runner.model_config = MagicMock()
            runner.max_memory = 100
            runner.local_rank = 0
            runner.world_size = 1
            runner.model.soc_info.need_nz = False
            
            mock_hbm.get_hbm_usage.return_value = 0.1
            
            runner.warm_up()
            
            mock_gen_req.assert_called()
            self.assertGreater(runner.warm_up_memory, 0)

    def test_infer(self):
        with patch('examples.multimodal_runner.ENV') as mock_env, \
             patch('examples.multimodal_runner.MultimodalInput'), \
             patch('examples.run_pa.PARunner.__init__', side_effect=pa_init_side_effect, autospec=True), \
             patch('examples.multimodal_runner.safe_from_pretrained'):
            runner = MultimodalPARunner(**self.kwargs)
            runner.rank = 0
            runner.model = MagicMock()
            runner.model.postprocessor = MagicMock()
            runner.local_rank = 0
            
            # Case 1: No profiling
            mock_env.profiling_enable = False
            mock_env.token_ids_save_enable = True
            
            with patch.object(runner, 'make_cache_manager'), \
                 patch.object(runner, 'run') as mock_run, \
                 patch.object(runner, 'precision_save') as mock_prec, \
                 patch.object(runner, 'performance_test') as mock_perf, \
                 patch.object(runner, 'save_token_ids') as mock_save_ids:
                
                mock_res = MagicMock()
                mock_res.all_generate_text_list = []
                mock_res.all_token_num_list = []
                mock_res.e2e_time_all = 0
                mock_run.return_value = mock_res
                
                runner.infer("inputs", 1, 10, False, max_iters=1)
                
                mock_run.assert_called()
                mock_prec.assert_called()
                mock_perf.assert_called()
                mock_save_ids.assert_called()
                
            # Case 2: Profiling enabled
            mock_env.profiling_enable = True
            with patch.object(runner, 'make_cache_manager'), \
                 patch.object(runner, 'profiling') as mock_prof, \
                 patch.object(runner, 'precision_save') as mock_prec, \
                 patch.object(runner, 'performance_test') as mock_perf, \
                 patch.object(runner, 'save_token_ids') as mock_save_ids:
                 
                mock_res = MagicMock()
                mock_res.all_generate_text_list = []
                mock_res.all_token_num_list = []
                mock_res.e2e_time_all = 0
                mock_prof.return_value = mock_res
                
                runner.infer("inputs", 1, 10, False, max_iters=1)
                mock_prof.assert_called()

if __name__ == '__main__':
    unittest.main()
