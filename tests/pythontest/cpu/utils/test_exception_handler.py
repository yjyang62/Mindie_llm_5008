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
import unittest
from unittest.mock import patch

# Set required env var before any mindie import triggers EnvVar() validation
os.environ.setdefault("MINDIE_LLM_BENCHMARK_FILEPATH", "/tmp/benchmark.jsonl")

import torch

from mindie_llm.utils.decorators.exception_handler import (
    _is_target_method,
    _torch_oom_handler,
    exception_handler,
)


class TestTargetMethodFilter(unittest.TestCase):
    """Test _is_target_method filtering logic."""

    def test_includes_forward(self):
        self.assertTrue(_is_target_method("forward"))

    def test_includes_compile(self):
        self.assertTrue(_is_target_method("compile"))

    def test_includes_load_weights(self):
        self.assertTrue(_is_target_method("load_weights"))

    def test_includes_forward_with_suffix(self):
        self.assertTrue(_is_target_method("forward_layerwise_disaggregated"))

    def test_excludes_generate_position_ids(self):
        self.assertFalse(_is_target_method("generate_position_ids"))

    def test_excludes_dunder_methods(self):
        self.assertFalse(_is_target_method("__init__"))
        self.assertFalse(_is_target_method("__call__"))

    def test_excludes_arbitrary_method(self):
        self.assertFalse(_is_target_method("some_other_method"))


class TestTorchOomHandler(unittest.TestCase):
    """Test _torch_oom_handler wrapper behavior."""

    def test_oom_caught_and_re_raised_as_runtime_error(self):
        """torch.OutOfMemoryError should be caught, logged, and re-raised as RuntimeError."""

        def failing_func():
            raise torch.OutOfMemoryError("test OOM")

        wrapped = _torch_oom_handler(failing_func)

        with self.assertRaises(RuntimeError) as cm:
            wrapped()

        self.assertIn("MIE05E000006", str(cm.exception))

    def test_oom_logger_called_with_error_code(self):
        """Logger.error should be called with ACL_GRAPH_OUT_OF_MEMORY error code."""
        from mindie_llm.utils.log.error_code import ErrorCode

        def failing_func():
            raise torch.OutOfMemoryError("test OOM")

        wrapped = _torch_oom_handler(failing_func)

        with patch("mindie_llm.utils.decorators.exception_handler.logger") as mock_logger:
            with self.assertRaises(RuntimeError):
                wrapped()

            mock_logger.error.assert_called_once()
            args, _ = mock_logger.error.call_args
            self.assertIs(args[1], ErrorCode.ACL_GRAPH_OUT_OF_MEMORY)

    def test_oom_preserves_original_exception_chain(self):
        """The original torch.OutOfMemoryError should be chained via __cause__."""

        def failing_func():
            raise torch.OutOfMemoryError("test OOM")

        wrapped = _torch_oom_handler(failing_func)

        with self.assertRaises(RuntimeError) as cm:
            wrapped()

        self.assertIsInstance(cm.exception.__cause__, torch.OutOfMemoryError)

    def test_non_oom_exception_passes_through(self):
        """Non-OOM exceptions should be re-raised unchanged."""

        def failing_func():
            raise ValueError("some other error")

        wrapped = _torch_oom_handler(failing_func)

        with self.assertRaises(ValueError):
            wrapped()

    def test_successful_call_returns_result(self):
        """A successful call should return the original result."""

        def successful_func():
            return 42

        wrapped = _torch_oom_handler(successful_func)

        result = wrapped()
        self.assertEqual(result, 42)

    def test_oom_skipped_when_torch_lacks_oom_error(self):
        """If torch.OutOfMemoryError is unavailable, non-OOM errors should still pass through."""
        oom_cls = getattr(torch, "OutOfMemoryError", None)
        if oom_cls is None:
            self.skipTest("torch.OutOfMemoryError not available, nothing to mask")

        def failing_func():
            raise ValueError("non-OOM error")

        wrapped = _torch_oom_handler(failing_func)

        # Temporarily hide OutOfMemoryError to simulate old torch
        del torch.OutOfMemoryError
        try:
            with self.assertRaises(ValueError):
                wrapped()
        finally:
            torch.OutOfMemoryError = oom_cls


class TestExceptionHandlerDecorator(unittest.TestCase):
    """Test @exception_handler class decorator."""

    def test_wraps_forward_method(self):
        """forward() should be wrapped by the handler."""

        @exception_handler
        class FakeRunner:
            def forward(self):
                raise torch.OutOfMemoryError("test OOM")

        runner = FakeRunner()
        with self.assertRaises(RuntimeError):
            runner.forward()

    def test_wraps_compile_method(self):
        """compile() should be wrapped by the handler."""

        @exception_handler
        class FakeRunner:
            def compile(self):
                raise torch.OutOfMemoryError("test OOM")

        runner = FakeRunner()
        with self.assertRaises(RuntimeError):
            runner.compile()

    def test_wraps_load_weights_method(self):
        """load_weights() should be wrapped by the handler."""

        @exception_handler
        class FakeRunner:
            def load_weights(self):
                raise torch.OutOfMemoryError("test OOM")

        runner = FakeRunner()
        with self.assertRaises(RuntimeError):
            runner.load_weights()

    def test_skips_generate_position_ids(self):
        """generate_position_ids() should NOT be wrapped."""

        @exception_handler
        class FakeRunner:
            def generate_position_ids(self):
                return "not wrapped"

        runner = FakeRunner()
        result = runner.generate_position_ids()
        self.assertEqual(result, "not wrapped")

    def test_non_oom_passes_through_on_wrapped_method(self):
        """Wrapped methods should still let non-OOM exceptions through."""

        @exception_handler
        class FakeRunner:
            def forward(self):
                raise ValueError("not OOM")

        runner = FakeRunner()
        with self.assertRaises(ValueError):
            runner.forward()

    def test_works_with_classmethod(self):
        """@exception_handler should handle classmethod correctly."""

        @exception_handler
        class FakeRunner:
            @classmethod
            def forward(cls):
                raise torch.OutOfMemoryError("test OOM")

        with self.assertRaises(RuntimeError):
            FakeRunner.forward()

    def test_works_with_staticmethod(self):
        """@exception_handler should handle staticmethod correctly."""

        @exception_handler
        class FakeRunner:
            @staticmethod
            def forward():
                raise torch.OutOfMemoryError("test OOM")

        with self.assertRaises(RuntimeError):
            FakeRunner.forward()

    def test_non_oom_methods_unchanged(self):
        """Methods not in the target list should behave normally."""

        @exception_handler
        class FakeRunner:
            def some_helper(self):
                return "unchanged"

        runner = FakeRunner()
        self.assertEqual(runner.some_helper(), "unchanged")

    def test_wraps_functions_starting_with_compile(self):
        """Any method starting with 'compile' should be wrapped."""

        @exception_handler
        class FakeRunner:
            def compile_special(self):
                raise torch.OutOfMemoryError("test OOM")

        runner = FakeRunner()
        with self.assertRaises(RuntimeError):
            runner.compile_special()


if __name__ == "__main__":
    unittest.main()
