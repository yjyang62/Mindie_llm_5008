# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import time
import uuid
import unittest
import io
import json
from contextlib import nullcontext
from unittest.mock import patch, MagicMock, call
from collections import deque

from mindie_llm.utils.env import ENV
from mindie_llm.utils.tensor import npu
from mindie_llm.utils.decorators.time_decorator import Timer


class TestTimer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use temporary file path for testing
        cls.test_benchmark_file = "/tmp/test_benchmark.log"
        ENV.benchmark_filepath = cls.test_benchmark_file
        ENV.benchmark_reserving_ratio = 0.1  # Example reserving ratio

    def setUp(self):
        # Reset ENV config before each test
        ENV.benchmark_enable = True
        ENV.benchmark_enable_async = True
        self.mock_logger = MagicMock()
        self.timer = Timer(self.mock_logger)
        # Clean test file
        if os.path.exists(ENV.benchmark_filepath):
            os.remove(ENV.benchmark_filepath)

    def tearDown(self):
        # Ensure background thread stops
        if hasattr(self.timer, 'flush_thread'):
            self.timer.flush_thread.join(timeout=1)
        # Clean up test file
        if os.path.exists(ENV.benchmark_filepath):
            os.remove(ENV.benchmark_filepath)

    def test_initialization(self):
        """Test Timer initialization correctness"""
        self.assertIsInstance(self.timer.cache, deque)
        self.assertEqual(self.timer.max_cache_size, 10000)
        self.assertTrue(self.timer.flush_thread.is_alive())

    def test_track_time_decorator_enabled(self):
        """Test decorator behavior when benchmarking is enabled"""
        ENV.benchmark_enable = True

        @self.timer.track_time("test_func")
        def mock_func():
            time.sleep(0.01)
            return "result"

        with patch.object(npu, 'synchronize') as mock_sync:
            result = mock_func()

        self.assertEqual(result, "result")
        mock_sync.assert_has_calls([call(), call()])
        self.assertIn("test_func", self.timer.time_cache)
        self.assertGreaterEqual(self.timer.time_cache["test_func"][0], 0.01)

    def test_track_time_decorator_disabled(self):
        """Test decorator behavior when benchmarking is disabled"""
        ENV.benchmark_enable = False

        @self.timer.track_time("test_func")
        def mock_func():
            return "result"

        result = mock_func()
        self.assertEqual(result, "result")
        self.assertEqual(len(self.timer.time_cache), 0)

    def test_flush_to_file(self):
        """Test flushing cached data to file"""
        test_data = [{"key": "value1"}, {"key": "value2"}]
        self.timer.cache = deque(test_data)

        with patch.object(self.timer, '_write_to_file') as mock_write:
            self.timer.flush_to_file()
        
        mock_write.assert_called_once_with(test_data)
        self.assertEqual(len(self.timer.cache), 0)

    def test_log_time_async(self):
        """Test asynchronous logging functionality"""
        request_ids = [uuid.uuid4(), uuid.uuid4()]
        token_indices = [10, 20]
        self.timer.time_cache = {"stage1": [0.1], "stage2": [0.2]}

        with patch('os.remove') as _, \
            patch.object(self.timer, 'cache', new_callable=deque) as mock_cache:
            
            self.timer.log_time_async(
                rank=0,
                request_ids=request_ids,
                token_indices=token_indices,
                input_metadata=MagicMock(is_prefill=True)
            )
            
            # Verify cache entries
            self.assertEqual(len(mock_cache), 2)
            for entry in mock_cache:
                self.assertIn('batch_id', entry)
                self.assertIn('request_id', entry)
                self.assertEqual(entry['unit'], 'ms')
    
    def test_log_time(self):
        request_ids = ['req-1', 'req-2']
        token_indices = [0, 1]
        self.timer.time_cache = {'stage1': [0.1234]}
        expected_stage_ms = 0.1234 * 1000

        fake_file = io.StringIO()

        with patch('mindie_llm.utils.decorators.time_decorator.create_log_dir_and_check_permission') as mock_create_log_dir_and_check_permission, \
             patch('mindie_llm.utils.decorators.time_decorator.safe_open') as mock_safe_open, \
             patch('os.path.exists', return_value=False), \
             patch('os.path.getsize', return_value=0), \
             patch('uuid.uuid4', return_value='batch-uuid'):

            mock_safe_open.return_value = nullcontext(fake_file)

            self.timer.log_time(0, request_ids, token_indices)

        mock_create_log_dir_and_check_permission.assert_called_once_with(ENV.benchmark_filepath)
        mock_safe_open.assert_called_once()
        self.assertEqual(mock_safe_open.call_args.args[1], 'a')
        self.assertEqual(mock_safe_open.call_args.kwargs.get('encoding'), 'utf-8')

        written_lines = fake_file.getvalue().strip().splitlines()
        self.assertEqual(len(written_lines), len(request_ids))

        for index, line in enumerate(written_lines):
            data = json.loads(line)
            self.assertEqual(data['batch_id'], 'batch-uuid')
            self.assertEqual(data['request_id'], request_ids[index])
            self.assertEqual(data['token_idx'], token_indices[index])
            self.assertEqual(data['unit'], 'ms')
            self.assertIn('stage1', data)
            self.assertAlmostEqual(data['stage1'], expected_stage_ms)

        self.assertEqual(self.timer.time_cache, {})

    def test_log_time_rank_not_0(self):
        """Test log_time when rank is not 0"""
        request_ids = [uuid.uuid4(), uuid.uuid4()]
        token_indices = [10, 20]
        self.timer.time_cache = {"stage1": [0.1], "stage2": [0.2]}

        self.timer.log_time(
            rank=1,  # rank is not 0
            request_ids=request_ids,
            token_indices=token_indices
        )

        # Verify log file is not created
        self.assertFalse(os.path.exists(ENV.benchmark_filepath))

    def test_max_cache_size_flush(self):
        """Test cache flush when reaching maximum size"""
        self.timer.max_cache_size = 2
        test_data = [{"d1": 1}, {"d2": 2}, {"d3": 3}]
        
        with patch.object(self.timer, 'flush_to_file') as mock_flush:
            with self.timer.cache_lock:
                for data in test_data:
                    self.timer.cache.append(data)

                if len(self.timer.cache) >= self.timer.max_cache_size:
                    self.timer.flush_condition.notify()
            # make sure flush thread done
            time.sleep(0.1)  
            self.assertTrue(mock_flush.called)
            self.assertEqual(len(self.timer.cache), 3)
    
    def test_track_time_async_decorator_enabled(self):
        """Test async decorator when async benchmarking is enabled"""
        ENV.benchmark_enable_async = True

        with patch('mindie_llm.utils.decorators.time_decorator.time.time', side_effect=[1.0, 1.1]) as mock_time, \
             patch.object(self.timer.logger, 'debug') as mock_debug:

            @self.timer.track_time_async("async_func")
            def mock_func():
                return "async_result"

            result = mock_func()

        self.assertEqual(result, "async_result")
        mock_time.assert_called()
        mock_debug.assert_called_once()
        self.assertIn("async_func", self.timer.time_cache)
        self.assertAlmostEqual(self.timer.time_cache["async_func"][-1], 0.1)
    
    def test_write_to_file_file_not_exists(self):
        """Test _write_to_file when file does not exist"""
        cache_to_write = [
            {"batch_id": "1", "request_id": "req1", "token_idx": 10, "unit": "ms"},
            {"batch_id": "2", "request_id": "req2", "token_idx": 20, "unit": "ms"}
        ]
        
        with patch('mindie_llm.utils.decorators.time_decorator.create_log_dir_and_check_permission'), \
             patch('mindie_llm.utils.decorators.time_decorator.safe_open'):
            self.timer._write_to_file(cache_to_write)

        # Verify log file is created and contains the expected content
        self.assertFalse(os.path.exists(ENV.benchmark_filepath))
    
    def test_write_to_file_file_exists(self):
        """Test _write_to_file when file exists"""
        # Create an existing file with some content
        with open(ENV.benchmark_filepath, 'w') as file:
            file.write(json.dumps({"batch_id": "0", "request_id": "req0", "token_idx": 5, "unit": "ms"}) + '\n')

        cache_to_write = [
            {"batch_id": "1", "request_id": "req1", "token_idx": 10, "unit": "ms"},
            {"batch_id": "2", "request_id": "req2", "token_idx": 20, "unit": "ms"}
        ]
        
        with patch('mindie_llm.utils.decorators.time_decorator.create_log_dir_and_check_permission'), \
             patch('mindie_llm.utils.decorators.time_decorator.safe_open'):
            self.timer._write_to_file(cache_to_write)

        # Verify log file contains the existing content and the new content
        with open(ENV.benchmark_filepath, 'r') as file:
            lines = file.readlines()
            print(len(lines))
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), {"batch_id": "0", "request_id": "req0", "token_idx": 5, "unit": "ms"})
            for i, line in enumerate(lines[1:], start=1):
                log_entry = json.loads(line)
                self.assertEqual(log_entry, cache_to_write[i - 1])

if __name__ == '__main__':
    unittest.main()