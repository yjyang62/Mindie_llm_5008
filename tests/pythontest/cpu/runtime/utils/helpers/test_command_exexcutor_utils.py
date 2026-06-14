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
from unittest.mock import MagicMock, patch, Mock
import subprocess

from mindie_llm.runtime.utils.helpers.command_executor import execute_command


class TestExecuteCommandUtils(unittest.TestCase):
    """Test cases for execute_command function."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def tearDown(self):
        """Clean up after tests."""
        pass

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_success(self, mock_popen):
        """Test execute_command with successful command execution."""
        # Create mock process
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        
        # Mock communicate to return stdout and stderr
        mock_process.communicate.return_value = (b"Hello World\n", b"")
        mock_popen.return_value = mock_process
        
        # Execute command
        result = execute_command(["echo", "Hello World"])
        
        # Verify Popen was called with correct arguments
        mock_popen.assert_called_once_with(
            ["echo", "Hello World"],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Verify communicate was called with timeout
        mock_process.communicate.assert_called_once_with(timeout=1000)
        
        # Verify result
        self.assertEqual(result, "Hello World\n")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_with_multiple_args(self, mock_popen):
        """Test execute_command with multiple command arguments."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (b"file1.txt\nfile2.txt\n", b"")
        mock_popen.return_value = mock_process
        
        result = execute_command(["ls", "-l", "/tmp"])
        
        # Verify Popen was called with all arguments
        mock_popen.assert_called_once_with(
            ["ls", "-l", "/tmp"],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.assertEqual(result, "file1.txt\nfile2.txt\n")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_empty_output(self, mock_popen):
        """Test execute_command with empty output."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (b"", b"")
        mock_popen.return_value = mock_process
        
        result = execute_command(["true"])
        
        self.assertEqual(result, "")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_multiline_output(self, mock_popen):
        """Test execute_command with multiline output."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (
            b"Line 1\nLine 2\nLine 3\n",
            b""
        )
        mock_popen.return_value = mock_process
        
        result = execute_command(["echo", "-e", "Line 1\nLine 2\nLine 3"])
        
        self.assertEqual(result, "Line 1\nLine 2\nLine 3\n")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_unicode_output(self, mock_popen):
        """Test execute_command with unicode output."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        # UTF-8 encoded unicode string
        mock_process.communicate.return_value = (
            "测试中文\n".encode('utf-8'),
            b""
        )
        mock_popen.return_value = mock_process
        
        result = execute_command(["echo", "测试中文"])
        
        self.assertEqual(result, "测试中文\n")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_shell_false(self, mock_popen):
        """Test that execute_command uses shell=False."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (b"output", b"")
        mock_popen.return_value = mock_process
        
        execute_command(["test", "command"])
        
        # Verify shell=False was passed
        call_kwargs = mock_popen.call_args[1]
        self.assertFalse(call_kwargs['shell'])

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_captures_stdout_stderr(self, mock_popen):
        """Test that execute_command captures both stdout and stderr."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (b"stdout output", b"stderr output")
        mock_popen.return_value = mock_process
        
        result = execute_command(["test", "command"])
        
        # Verify stdout and stderr pipes were set
        call_kwargs = mock_popen.call_args[1]
        self.assertEqual(call_kwargs['stdout'], subprocess.PIPE)
        self.assertEqual(call_kwargs['stderr'], subprocess.PIPE)
        
        # Result should only contain stdout (stderr is ignored)
        self.assertEqual(result, "stdout output")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_timeout(self, mock_popen):
        """Test that execute_command uses timeout of 1000 seconds."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (b"output", b"")
        mock_popen.return_value = mock_process
        
        execute_command(["test", "command"])
        
        # Verify timeout was passed to communicate
        mock_process.communicate.assert_called_once_with(timeout=1000)

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_timeout_error(self, mock_popen):
        """Test execute_command raises TimeoutExpired when command times out."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.side_effect = subprocess.TimeoutExpired("test", 1000)
        mock_popen.return_value = mock_process
        
        # Should propagate TimeoutExpired exception
        with self.assertRaises(subprocess.TimeoutExpired):
            execute_command(["slow", "command"])

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_process_error(self, mock_popen):
        """Test execute_command when process creation fails."""
        mock_popen.side_effect = OSError("Command not found")
        
        # Should propagate the error
        with self.assertRaises(OSError) as context:
            execute_command(["nonexistent", "command"])
        
        self.assertIn("Command not found", str(context.exception))

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_decode_error(self, mock_popen):
        """Test execute_command when decoding fails."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        # Invalid UTF-8 bytes
        mock_process.communicate.return_value = (b'\xff\xfe\x00', b"")
        mock_popen.return_value = mock_process
        
        # Should raise UnicodeDecodeError
        with self.assertRaises(UnicodeDecodeError):
            execute_command(["test", "command"])

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_context_manager(self, mock_popen):
        """Test that execute_command uses context manager properly."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (b"output", b"")
        mock_popen.return_value = mock_process
        
        execute_command(["test", "command"])
        
        # Verify context manager methods were called
        mock_process.__enter__.assert_called_once()
        mock_process.__exit__.assert_called_once()

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_single_argument(self, mock_popen):
        """Test execute_command with single command argument."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (b"version 1.0.0\n", b"")
        mock_popen.return_value = mock_process
        
        result = execute_command(["python", "--version"])
        
        mock_popen.assert_called_once_with(
            ["python", "--version"],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.assertEqual(result, "version 1.0.0\n")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_stderr_ignored(self, mock_popen):
        """Test that stderr output is ignored in return value."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        # stdout and stderr both have content
        mock_process.communicate.return_value = (
            b"stdout content",
            b"stderr content"
        )
        mock_popen.return_value = mock_process
        
        result = execute_command(["test", "command"])
        
        # Only stdout should be returned
        self.assertEqual(result, "stdout content")
        # stderr is captured but not returned

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_with_special_characters(self, mock_popen):
        """Test execute_command with special characters in output."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        # Output with special characters
        mock_process.communicate.return_value = (
            b"Special chars: !@#$%^&*()\n",
            b""
        )
        mock_popen.return_value = mock_process
        
        result = execute_command(["echo", "Special chars: !@#$%^&*()"])
        
        self.assertEqual(result, "Special chars: !@#$%^&*()\n")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_with_newlines_and_tabs(self, mock_popen):
        """Test execute_command with newlines and tabs in output."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (
            b"Line1\n\tTabbed line\nLine3\n",
            b""
        )
        mock_popen.return_value = mock_process
        
        result = execute_command(["test", "command"])
        
        self.assertEqual(result, "Line1\n\tTabbed line\nLine3\n")

    @patch('mindie_llm.runtime.utils.helpers.command_executor.subprocess.Popen')
    def test_execute_command_empty_command_list(self, mock_popen):
        """Test execute_command with empty command list."""
        mock_process = MagicMock()
        mock_process.__enter__ = Mock(return_value=mock_process)
        mock_process.__exit__ = Mock(return_value=False)
        mock_process.communicate.return_value = (b"", b"")
        mock_popen.return_value = mock_process
        
        result = execute_command([])
        
        mock_popen.assert_called_once_with(
            [],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main()
