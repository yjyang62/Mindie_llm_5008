# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from io import StringIO
import os
import sys
import threading
import unittest
import logging
import tempfile
import shutil
from collections import namedtuple
from unittest.mock import patch, mock_open, MagicMock
from mindie_llm.utils.log.logging_base import (
    Component,
    HandlerType,
    CustomLogger,
    CustomLoggerAdapter,
    ErrorCodeFormatter,
    SelectiveConsoleHandler,
    SelectiveRotatingFileHandler,
    LoggerManager,
    MAX_OPEN_LOG_FILE_PERM,
    MAX_CLOSE_LOG_FILE_PERM,

)


# Define Mock configuration
MockConfig = namedtuple('MockConfig', [
    'create_log',
    'makedir',
    'standardize',
    'check_perm',
    'expanduser',
    'exists',
    'join',
    'makedirs'
])


class TestCustomLogger(unittest.TestCase):
    """Test core functionality of CustomLogger: error code handling, stacklevel adjustment, and parameter passing"""
    
    def setUp(self):
        """Create logger instance for testing"""
        self.logger = logging.getLogger('test_logger')
        self.logger.__class__ = CustomLogger
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        """Clean up test environment"""
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def test_non_error_levels_without_error_code(self):
        """Test that non-error level logs do not handle error codes"""
        with patch.object(self.logger, '_log') as mock_log:
            self.logger.info("Test message")
            
            # Verify no error code handling
            call_args = mock_log.call_args
            self.assertNotIn('error_code', call_args[1].get('extra', {}))

    def test_args_preservation(self):
        """Test preservation of non-error code parameters"""
        with patch.object(self.logger, '_log') as mock_log:
            self.logger.error("Test %s message %d", "formatted", 42)
            
            # Verify call parameters
            call_args = mock_log.call_args
            # Get all parameters
            args, kwargs = call_args
            
            # Verify parameter structure
            self.assertEqual(len(args), 3)  # level, msg, args
            self.assertEqual(args[0], logging.ERROR)
            self.assertEqual(args[1], "Test %s message %d")
            self.assertEqual(args[2], ("formatted", 42))


class TestCustomLoggerAdapter(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('test_logger')
        self.base_extra = {'default_key': 'default_value'}
        self.adapter = CustomLoggerAdapter(self.logger, self.base_extra)

    def test_process_with_empty_extra(self):
        """Test processing with empty extra"""
        msg, kwargs = self.adapter.process("test message", {})
        expected = {'default_key': 'default_value'}
        self.assertEqual(kwargs['extra'], expected)

    def test_process_with_none_extra(self):
        """Test processing with None extra"""
        msg, kwargs = self.adapter.process("test message", {"extra": None})
        expected = {'default_key': 'default_value'}
        self.assertEqual(kwargs['extra'], expected)

    def test_process_with_new_extra(self):
        """Test processing with new extra"""
        new_extra = {'new_key': 'new_value'}
        msg, kwargs = self.adapter.process("test message", {"extra": new_extra})
        expected = {'default_key': 'default_value', 'new_key': 'new_value'}
        self.assertEqual(kwargs['extra'], expected)

    def test_process_with_overlapping_keys(self):
        """Test processing with overlapping keys between new extra and base_extra"""
        new_extra = {'default_key': 'new_default_value', 'another_key': 'another_value'}
        msg, kwargs = self.adapter.process("test message", {"extra": new_extra})
        expected = {'default_key': 'new_default_value', 'another_key': 'another_value'}
        self.assertEqual(kwargs['extra'], expected)

    def test_process_without_extra_in_kwargs(self):
        """Test processing when kwargs doesn't contain extra"""
        msg, kwargs = self.adapter.process("test message", {"other": "value"})
        expected = {'default_key': 'default_value'}
        self.assertEqual(kwargs['extra'], expected)
        self.assertEqual(kwargs['other'], "value")  # Ensure other parameters remain unchanged

    def test_process_preserves_original_extra(self):
        """Test that original extra dictionary is not modified"""
        original_extra = {'key': 'value'}
        original_copy = original_extra.copy()
        msg, kwargs = self.adapter.process("test message", {"extra": original_extra})
        
        # Verify original dictionary is not modified
        self.assertEqual(original_extra, original_copy)
        
        # Verify merge result
        expected = {'default_key': 'default_value', 'key': 'value'}
        self.assertEqual(kwargs['extra'], expected)

    def test_process_with_empty_base_extra(self):
        """Test processing with empty base_extra"""
        empty_adapter = CustomLoggerAdapter(self.logger, {})
        new_extra = {'key': 'value'}
        msg, kwargs = empty_adapter.process("test message", {"extra": new_extra})
        self.assertEqual(kwargs['extra'], new_extra)

    def test_process_with_nested_extra(self):
        """Test processing with nested extra dictionary"""
        nested_extra = {'nested': {'inner_key': 'inner_value'}}
        msg, kwargs = self.adapter.process("test message", {"extra": nested_extra})
        expected = {'default_key': 'default_value', 'nested': {'inner_key': 'inner_value'}}
        self.assertEqual(kwargs['extra'], expected)

    def test_process_with_complex_types(self):
        """Test processing with complex types in extra"""
        complex_extra = {
            'list': [1, 2, 3],
            'dict': {'inner': 'value'},
            'none': None,
            'bool': True
        }
        msg, kwargs = self.adapter.process("test message", {"extra": complex_extra})
        expected = {
            'default_key': 'default_value',
            'list': [1, 2, 3],
            'dict': {'inner': 'value'},
            'none': None,
            'bool': True
        }
        self.assertEqual(kwargs['extra'], expected)

    def test_process_message_unchanged(self):
        """Test that message content remains unchanged"""
        test_msg = "test message"
        msg, kwargs = self.adapter.process(test_msg, {})
        self.assertEqual(msg, test_msg)

    def test_process_with_multiple_overlapping_keys(self):
        """Test processing with multiple overlapping keys"""
        new_extra = {
            'default_key': 'new_value',
            'another_key': 'another_value',
            'third_key': 'third_value'
        }
        msg, kwargs = self.adapter.process("test message", {"extra": new_extra})
        expected = {
            'default_key': 'new_value',
            'another_key': 'another_value',
            'third_key': 'third_value'
        }
        self.assertEqual(kwargs['extra'], expected)

    def test_process_with_empty_dict_extra(self):
        """Test processing with empty dictionary extra"""
        msg, kwargs = self.adapter.process("test message", {"extra": {}})
        expected = {'default_key': 'default_value'}
        self.assertEqual(kwargs['extra'], expected)


class TestErrorCodeFormatter(unittest.TestCase):
    """Unit test class for ErrorCodeFormatter"""
    
    def setUp(self):
        """Pre-test setup - Create objects that don't depend on configuration"""
        self.logger = logging.getLogger('test_logger')
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.component = Component.LLM
        
    def tearDown(self):
        """Post-test cleanup"""
        self.logger.removeHandler(self.handler)
        self.handler.close()
        self.stream.close()

    def _create_test_record(self, level=logging.ERROR, msg="Test message", error_code=None):
        """Create test LogRecord"""
        record = logging.LogRecord(
            name='test_logger',
            level=level,
            pathname='/path/to/test_file.py',
            lineno=42,
            msg=msg,
            args=(),
            exc_info=None
        )
        
        record.created = 1735565368.595
        record.asctime = '2024-12-30 21:29:28,595'
        record.process = 390625
        record.thread = 281473386389536
        record.msecs = 595
        record.relativeCreated = 1234567.89
        record.stack_info = None
        record.threadName = 'MainThread'
        record.processName = 'MainProcess'
        record.funcName = 'test_function'
        record.module = 'test_file'
        record.filename = 'test_file.py'
        
        if error_code is not None:
            record.error_code = error_code
        
        return record
    
    def test_error_code_verbose_format(self):
        """Test error code format in verbose mode"""
        formatter = ErrorCodeFormatter(self.component)
        formatter.verbose = True
        
        record = self._create_test_record(
            level=logging.ERROR,
            msg="Test error message",
            error_code='12345'
        )
        
        formatted = formatter.format(record)
        print(f"Verbose formatted: {formatted}")
        
        self.assertIn('[390625]', formatted)
        self.assertIn('[281473386389536]', formatted)
        self.assertIn('[ERROR]', formatted)
        self.assertIn('[12345]', formatted)
    
    def test_error_code_non_verbose_format(self):
        """Test error code format in non-verbose mode"""
        formatter = ErrorCodeFormatter(self.component)
        formatter.verbose = False
        
        record = self._create_test_record(
            level=logging.ERROR,
            msg="Test error message",
            error_code='12345'
        )
        
        formatted = formatter.format(record)
        print(f"Non-verbose formatted: {formatted}")
        
        self.assertNotIn('[390625]', formatted)
        self.assertNotIn('[281473386389536]', formatted)
        self.assertIn('[ERROR]', formatted)
        self.assertIn('[12345]', formatted)
    
    def test_debug_log_non_verbose_format(self):
        """Test DEBUG level log format in non-verbose mode"""
        formatter = ErrorCodeFormatter(self.component)
        formatter.verbose = False
        record = self._create_test_record(
            level=logging.DEBUG,
            msg="Test debug message"
        )
        
        formatted = formatter.format(record)
        print(f"Debug non-verbose formatted: {formatted}")
        
        self.assertNotIn('[281473386389536]', formatted)
        self.assertIn('[DEBUG]', formatted)
        self.assertIn('Test debug message', formatted)
    
    def test_info_log_verbose_format(self):
        """Test INFO level log format in verbose mode"""
        formatter = ErrorCodeFormatter(self.component)
        formatter.verbose = True
        
        record = self._create_test_record(
            level=logging.INFO,
            msg="Test info message"
        )
        
        formatted = formatter.format(record)
        print(f"Info verbose formatted: {formatted}")
        
        self.assertIn('[390625]', formatted)
        self.assertIn('[281473386389536]', formatted)
        self.assertIn('[INFO]', formatted)
        self.assertIn('Test info message', formatted)
    
    def test_error_without_error_code_verbose(self):
        """Test ERROR level log in verbose mode without error_code"""
        formatter = ErrorCodeFormatter(self.component)
        formatter.verbose = True
        
        record = self._create_test_record(
            level=logging.ERROR,
            msg="Error without error code"
        )
        
        formatted = formatter.format(record)
        print(f"Error without error code (verbose): {formatted}")
        
        self.assertIn('[390625]', formatted)
        self.assertIn('[ERROR]', formatted)
        self.assertIn('Error without error code', formatted)
    
    def test_warning_with_error_code_non_verbose(self):
        """Test WARNING level log with error_code in non-verbose mode"""
        formatter = ErrorCodeFormatter(self.component)
        formatter.verbose = False
        
        record = self._create_test_record(
            level=logging.WARNING,
            msg="Warning with error code",
            error_code='WARN001'
        )
        
        formatted = formatter.format(record)
        print(f"Warning with error code (non-verbose): {formatted}")
        
        self.assertNotIn('[390625]', formatted)
        self.assertIn('[WARN]', formatted)
        self.assertIn('Warning with error code', formatted)
    
    def test_critical_level_with_error_code_verbose(self):
        """Test CRITICAL level log with error_code in verbose mode"""
        formatter = ErrorCodeFormatter(self.component)
        formatter.verbose = True
        
        record = self._create_test_record(
            level=logging.CRITICAL,
            msg="Critical error",
            error_code='CRIT001'
        )
        
        formatted = formatter.format(record)
        print(f"Critical with error code (verbose): {formatted}")
        
        self.assertIn('[390625]', formatted)
        self.assertIn('[CRITICAL]', formatted)
        self.assertIn('[CRIT001]', formatted)
        self.assertIn('Critical error', formatted)
    
    def test_format_method_returns_string(self):
        """Test that format method returns string"""
        formatter = ErrorCodeFormatter(self.component)
        record = self._create_test_record(
            level=logging.INFO,
            msg="Simple test"
        )
        
        result = formatter.format(record)
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestSelectiveConsoleHandler(unittest.TestCase):
    def setUp(self):
        self.handler = SelectiveConsoleHandler()
        self.logger = logging.getLogger('test')
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def test_emit_token_record(self):
        """Test that TOKEN type log records are not output"""
        with patch('sys.stdout') as mock_stdout:
            record = logging.LogRecord(
                name='test',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg='test message',
                args=(),
                exc_info=None
            )
            record.handler_ids = HandlerType.TOKEN
            self.handler.emit(record)
            mock_stdout.write.assert_not_called()


class TestSelectiveRotatingFileHandler(unittest.TestCase):
    def setUp(self):
        """Pre-test setup: Create temporary directory and test files"""
        self.test_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.test_dir, "test.log")
        self.handler_type = HandlerType.LLM
        self.handler = SelectiveRotatingFileHandler(
            self.handler_type,
            self.log_file,
            max_bytes=100,  # Small file size for testing
            backup_count=3
        )

    def tearDown(self):
        """Post-test cleanup: Close handler and delete temporary files"""
        self.handler.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_emit_with_default_handler(self):
        """Test log records using default handler"""
        # Create test record
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='test message',
            args=(),
            exc_info=None
        )
        record.default_file_handler = self.handler_type
        record.formatter = logging.Formatter('%(message)s')

        # Send record
        self.handler.emit(record)

        # Verify file creation and content
        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn('test message', content)

    def test_emit_with_selected_handler(self):
        """Test log records using selected handler"""
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='test message',
            args=(),
            exc_info=None
        )
        record.handler_ids = self.handler_type
        record.formatter = logging.Formatter('%(message)s')

        self.handler.emit(record)

        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn('test message', content)

    def test_get_rotation_filename(self):
        """Test rotation filename generation"""
        # Test base filename
        self.assertEqual(self.handler.get_rotation_filename(0), self.log_file)
        
        # Test rotation filenames
        self.assertEqual(
            self.handler.get_rotation_filename(1),
            os.path.join(self.test_dir, "test.01.log")
        )
        self.assertEqual(
            self.handler.get_rotation_filename(2),
            os.path.join(self.test_dir, "test.02.log")
        )
        self.assertEqual(
            self.handler.get_rotation_filename(10),
            os.path.join(self.test_dir, "test.10.log")
        )

    @patch('mindie_llm.utils.file_utils.safe_chmod')
    def test_doRollover(self, mock_safe_chmod):
        """Test log file rotation"""
        # Create a new handler with explicit rotation parameters
        handler = SelectiveRotatingFileHandler(
            self.handler_type,
            self.log_file,
            max_bytes=100,  # Set smaller file size
            backup_count=5   # Set more backup count
        )

        # Get filename base and extension
        base_name = self.log_file.rsplit('.', 1)[0]
        ext = self.log_file.rsplit('.', 1)[1]
        
        # Create test files
        with open(self.log_file, 'w') as f:
            f.write("Current log\n")
        
        # Create backup files (using new format)
        with open(f"{base_name}.01.{ext}", 'w') as f:
            f.write("Backup 1\n")
        with open(f"{base_name}.02.{ext}", 'w') as f:
            f.write("Backup 2\n")

        # Trigger rotation directly
        handler.doRollover()

        # Verify file renaming (new format)
        self.assertTrue(os.path.exists(f"{base_name}.03.{ext}"))  # Original .02
        self.assertTrue(os.path.exists(f"{base_name}.02.{ext}"))  # Original .01
        self.assertTrue(os.path.exists(f"{base_name}.01.{ext}"))  # Original base
        self.assertTrue(os.path.exists(self.log_file))            # New base

        # Verify permission settings
        expected_calls = [
            unittest.mock.call(f"{base_name}.03.{ext}", MAX_CLOSE_LOG_FILE_PERM),
            unittest.mock.call(f"{base_name}.02.{ext}", MAX_CLOSE_LOG_FILE_PERM),
            unittest.mock.call(f"{base_name}.01.{ext}", MAX_CLOSE_LOG_FILE_PERM),
        ]
        mock_safe_chmod.assert_has_calls(expected_calls, any_order=True)

    @patch('mindie_llm.utils.file_utils.safe_chmod')
    def test_doRollover_without_backup(self, mock_safe_chmod):
        """Test rotation without backup enabled"""
        # Create handler without backup
        handler = SelectiveRotatingFileHandler(
            self.handler_type,
            self.log_file,
            max_bytes=100,
            backup_count=0
        )

        # Create test file
        with open(self.log_file, 'w') as f:
            f.write("Current log\n")

        # Trigger rotation
        handler.doRollover()

        # Get filename base and extension
        base_name = self.log_file.rsplit('.', 1)[0]
        ext = self.log_file.rsplit('.', 1)[1]

        # Verify only base file exists
        self.assertTrue(os.path.exists(self.log_file))
        self.assertFalse(os.path.exists(f"{base_name}.01.{ext}"))

    @patch('mindie_llm.utils.file_utils.safe_chmod')
    def test_close(self, mock_safe_chmod):
        """Test handler closing"""
        # Create test file
        with open(self.log_file, 'w') as f:
            f.write("Test content\n")

        # Close handler
        self.handler.close()

        # Verify permission settings
        mock_safe_chmod.assert_called_once_with(self.log_file, MAX_CLOSE_LOG_FILE_PERM)

    @patch('mindie_llm.utils.file_utils.safe_chmod')
    def test_open(self, mock_safe_chmod):
        """Test file opening"""
        # Simulate existing file
        with patch('builtins.open', mock_open()) as mock_file:
            # Call _open method
            self.handler._open()

            # Verify permission settings
            expected_calls = [
                unittest.mock.call(self.log_file, MAX_OPEN_LOG_FILE_PERM),
                unittest.mock.call(self.log_file, MAX_OPEN_LOG_FILE_PERM)
            ]
            mock_safe_chmod.assert_has_calls(expected_calls)

    def test_handler_initialization(self):
        """Test handler initialization"""
        self.assertEqual(self.handler.handler_type, self.handler_type)
        self.assertEqual(self.handler.maxBytes, 100)
        self.assertEqual(self.handler.backupCount, 3)
        self.assertEqual(self.handler.mode, "a")

    def test_error_handling_in_emit(self):
        """Test error handling in emit method"""
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='test message',
            args=(),
            exc_info=None
        )
        record.default_file_handler = self.handler_type
        record.formatter = None  # Intentionally not setting formatter

        # Should not raise exception
        self.handler.emit(record)


class TestLoggerManager(unittest.TestCase):
    def setUp(self):
        """Pre-test setup"""
        self.logger_manager = LoggerManager()
        self.test_component = Component.LLM

    def test_set_log_level(self):
        """Test log level setting functionality"""
        logger = logging.getLogger("test")
        
        # Test valid log levels
        with patch('mindie_llm.utils.log.logging_base.ENV') as mock_env:
            mock_env.log_file_level = "INFO"
            with patch('mindie_llm.utils.log.utils.get_component_config', return_value="INFO"):
                self.logger_manager._set_log_level(logger)
                self.assertEqual(logger.level, logging.INFO)

        # Test invalid log level
        logger.setLevel(logging.DEBUG)
        with patch('mindie_llm.utils.log.logging_base.ENV') as mock_env:
            mock_env.log_file_level = "INVALID"
            with patch('mindie_llm.utils.log.utils.get_component_config', return_value="INVALID"):
                self.logger_manager._set_log_level(logger)
                self.assertEqual(logger.level, logging.INFO)  # Should fallback to INFO level

        # Test other valid levels
        test_cases = [
            ("DEBUG", logging.DEBUG),
            ("WARN", logging.WARN),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL)
        ]
        
        for level_name, expected_level in test_cases:
            with patch('mindie_llm.utils.log.logging_base.ENV') as mock_env:
                mock_env.log_file_level = level_name
                with patch('mindie_llm.utils.log.utils.get_component_config', return_value=level_name):
                    self.logger_manager._set_log_level(logger)
                    self.assertEqual(logger.level, expected_level)

    def test_get_log_rotate_config(self):
        """Test log rotation configuration retrieval"""
        # Test normal case
        max_size, max_files = self.logger_manager._get_log_rotate_config(self.test_component)
        self.assertIsInstance(max_size, int)
        self.assertIsInstance(max_files, int)

        # Test LLMMODELS component using old environment variable
        with patch.dict(os.environ, {"PYTHON_LOG_MAXSIZE": "100"}):
            with patch('logging.warning') as mock_warning:
                max_size, max_files = self.logger_manager._get_log_rotate_config(Component.LLMMODELS)
                mock_warning.assert_called()

    def test_get_logger(self):
        """Test logger retrieval functionality"""
        # Test creating new logger
        logger = self.logger_manager.get_logger(self.test_component)
        self.assertIsNotNone(logger)

        # Test getting existing logger
        logger2 = self.logger_manager.get_logger(self.test_component)
        self.assertEqual(logger, logger2)

    def test_create_logger(self):
        """Test logger creation functionality"""
        # Test logger creation
        self.logger_manager._create_logger(self.test_component)
        self.assertIn(self.test_component, self.logger_manager._loggers)
        self.assertIsNotNone(self.logger_manager._shared_console_handler)

    @patch('os.makedirs')
    @patch('os.path.join')
    @patch('os.path.exists')
    @patch('os.path.expanduser')
    @patch('mindie_llm.utils.file_utils.check_path_permission')
    @patch('mindie_llm.utils.file_utils.standardize_path')
    @patch('mindie_llm.utils.file_utils.makedir_and_change_permissions')
    @patch('mindie_llm.utils.log.logging_base.create_log_dir_and_check_permission')
    def test_create_file_handlers(self, *mocks):
        """Test file handler creation functionality"""
        # Create mock configuration object
        mock_config = MockConfig(*mocks)
        
        # Set mock return values
        mock_config.expanduser.return_value = "/home/test"
        mock_config.standardize.return_value = "/home/test"
        mock_config.exists.return_value = True
        mock_config.join.return_value = "/test/path"

        # Execute test
        self.logger_manager._create_file_handlers(self.test_component)
        
        # Verify handler creation
        expected_handlers = self.logger_manager._logger_to_handlers_map[self.test_component]
        for handler_type in expected_handlers:
            self.assertIn(handler_type, self.logger_manager._handlers)
            handler = self.logger_manager._handlers[handler_type]
            self.assertIsInstance(handler, SelectiveRotatingFileHandler)
            self.assertEqual(handler.handler_type, handler_type)

    def test_add_file_handlers(self):
        """Test file handler addition functionality"""
        # Create mock logger
        mock_logger = MagicMock()
        mock_logger.name = self.test_component.value  # Ensure name is Component.LLM.value
        
        # Add mock handlers
        mock_handler = MagicMock()
        # Use correct handler_type
        self.logger_manager._handlers[HandlerType.LLM] = mock_handler

        # Test enabling file output
        with patch('mindie_llm.utils.log.logging_base.ENV') as mock_env:
            mock_env.log_to_file = "ON"
            with patch('mindie_llm.utils.log.utils.get_component_config', return_value="ON"):
                self.logger_manager._add_file_handlers(mock_logger)
                mock_logger.addHandler.assert_called_once_with(mock_handler)

    def test_add_console_handler(self):
        """Test console handler addition functionality"""
        # Create mock logger
        mock_logger = MagicMock()
        mock_logger.name = self.test_component.value
        
        # Create mock console handler
        mock_console_handler = MagicMock()
        self.logger_manager._shared_console_handler = mock_console_handler

        # Test disabling console output
        mock_logger.reset_mock()
        with patch('mindie_llm.utils.log.logging_base.ENV') as mock_env:
            mock_env.log_to_stdout = "OFF"
            with patch('mindie_llm.utils.log.utils.get_component_config', return_value="OFF"):
                self.logger_manager._add_console_handler(mock_logger)
                mock_logger.addHandler.assert_not_called()
                
    def test_concurrent_logger_creation(self):
        """Test thread safety of concurrent logger creation"""
        def create_logger():
            self.logger_manager.get_logger(self.test_component)

        # Create multiple threads to get loggers simultaneously
        threads = [threading.Thread(target=create_logger) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify only one logger instance was created
        self.assertEqual(len([logger for logger in self.logger_manager._loggers.values() if logger.extra['component'] == self.test_component]), 1)


if __name__ == '__main__':
    unittest.main()
