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
from unittest.mock import MagicMock, patch
from mindie_llm.utils.log.logging import message_filter, print_log, \
    get_logger, Component, _MAX_MSG_LEN


class TestMessageFilter(unittest.TestCase):
    """Test message filtering functionality"""
    
    def test_message_length(self):
        """Test truncation of overly long strings"""
        long_msg = "llmm" * 1025
        filtered_msg = message_filter(long_msg)
        expect_msg = "llmm" * 1024 + "..."
        self.assertEqual(filtered_msg, expect_msg)

    def test_message_no_truncation(self):
        """Test that normal length strings are not truncated"""
        normal_msg = "Short message"
        filtered_msg = message_filter(normal_msg)
        self.assertEqual(filtered_msg, normal_msg)

    def test_special_char_removal(self):
        """Test replacement of special characters with spaces"""
        msg_with_special_chars = "This is a message with \n special \t characters \u000D and \r signs."
        filtered_msg = message_filter(msg_with_special_chars)
        self.assertEqual(filtered_msg, "This is a message with   special   characters   and   signs.")

    def test_multiple_spaces_replacement(self):
        """Test replacement of multiple consecutive spaces with 4 spaces"""
        msg_with_spaces = "This    is    a    message    with    excessive    spaces."
        filtered_msg = message_filter(msg_with_spaces)
        self.assertEqual(filtered_msg, "This    is    a    message    with    excessive    spaces.")

    def test_mixed_cases(self):
        """Test mixed case with special characters and spaces in message"""
        mixed_msg = "A long message with special \v characters \u000C and multiple    spaces."
        filtered_msg = message_filter(mixed_msg)
        self.assertEqual(filtered_msg, "A long message with special   characters   and multiple    spaces.")

    def test_empty_message(self):
        """Test empty message"""
        self.assertEqual(message_filter(""), "")

    def test_max_length_boundary(self):
        """Test maximum length boundary"""
        # Exactly maximum length
        msg = "x" * _MAX_MSG_LEN
        self.assertEqual(message_filter(msg), msg)
        # One character over maximum length
        msg = "x" * (_MAX_MSG_LEN + 1)
        result = message_filter(msg)
        self.assertTrue(result.endswith("..."))
        self.assertEqual(len(result), _MAX_MSG_LEN + 3)


class TestPrintLog(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()

    def test_non_zero_rank(self):
        """Test that non-zero rank does not print"""
        print_log(rank_id=1, logger_fn=self.mock_logger.info, msg="test")
        self.mock_logger.info.assert_not_called()

    def test_zero_rank_no_filter(self):
        """Test that zero rank does not filter"""
        print_log(rank_id=0, logger_fn=self.mock_logger.info, msg="test", need_filter=False)
        self.mock_logger.info.assert_called_once_with("test", stacklevel=2)

    def test_zero_rank_with_filter(self):
        """Test message filtering for zero rank"""
        msg = "test\nmessage"
        print_log(rank_id=0, logger_fn=self.mock_logger.info, msg=msg, need_filter=True)
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        self.assertNotIn('\n', args[0])  # Verify newline character is filtered

    def test_different_logger_functions(self):
        """Test different logger functions"""
        # Test info
        print_log(rank_id=0, logger_fn=self.mock_logger.info, msg="info")
        self.mock_logger.info.assert_called_once()
        
        # Test warning
        print_log(rank_id=0, logger_fn=self.mock_logger.warning, msg="warning")
        self.mock_logger.warning.assert_called_once()
        
        # Test error
        print_log(rank_id=0, logger_fn=self.mock_logger.error, msg="error")
        self.mock_logger.error.assert_called_once()

    def test_stacklevel_propagation(self):
        """Test stack level propagation"""
        print_log(rank_id=0, logger_fn=self.mock_logger.info, msg="test")
        self.mock_logger.info.assert_called_once_with("test", stacklevel=2)

    def test_message_conversion(self):
        """Test message type conversion"""
        # Test numeric message
        print_log(rank_id=0, logger_fn=self.mock_logger.info, msg=123, need_filter=True)
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        self.assertEqual(args[0], "123")

    def test_filter_flag_effect(self):
        """Test effect of filter flag"""
        msg = "test\nmessage"
        # No filtering
        print_log(rank_id=0, logger_fn=self.mock_logger.info, msg=msg, need_filter=False)
        args_no_filter = self.mock_logger.info.call_args[0]
        
        # With filtering
        print_log(rank_id=0, logger_fn=self.mock_logger.info, msg=msg, need_filter=True)
        args_filtered = self.mock_logger.info.call_args[0]
        
        # Verify filtering effect
        self.assertNotEqual(args_no_filter[0], args_filtered[0])
        self.assertNotIn('\n', args_filtered[0])


class TestLoggerInitialization(unittest.TestCase):
    def test_get_llm_logger(self):
        """Test getting logger for LLM component"""
        logger = get_logger(Component.LLM)
        
        # Verify logger instance exists
        self.assertIsNotNone(logger)
        
        # Verify logger configuration is correct
        self.assertEqual(logger.extra['component'], Component.LLM)
        
        # Verify logger can be used normally
        with patch.object(logger, 'info') as mock_info:
            logger.info("test message")
            mock_info.assert_called_once_with("test message")