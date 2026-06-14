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
import sys
import unittest
from unittest.mock import MagicMock

# Set required env var before any mindie import triggers EnvVar() validation
os.environ.setdefault("MINDIE_LLM_BENCHMARK_FILEPATH", "/tmp/benchmark.jsonl")

# Mock torch.npu before importing any mindie modules
import torch  # noqa: E402

if not hasattr(torch, "npu"):
    torch.npu = MagicMock()
torch.npu.config.allow_internal_format = True
torch.npu.current_stream.return_value.synchronize.return_value = None
torch.npu.FloatTensor = MagicMock
torch.npu.IntTensor = MagicMock

# Import DeviceType safely (enum only, no hardware access)
from mindie_llm.runtime.utils.npu.device_utils import DeviceType  # noqa: E402

# Mock get_npu_node_info before any chain import triggers hardware detection
mock_node_info = MagicMock()
mock_node_info.get_device_type.return_value = DeviceType.ASCEND_910_93
mock_node_info.get_hbm_capacity.return_value = 0
mock_node_info.get_hbm_usage.return_value = 0

# Patch device_utils before the model_runner_exp import chain
import mindie_llm.runtime.utils.npu.device_utils as device_utils_mod  # noqa: E402

device_utils_mod.get_npu_node_info = MagicMock(return_value=mock_node_info)
device_utils_mod.get_npu_hbm_info = MagicMock()

# Replace mie_ops with a mock to skip hardware-specific imports
mock_mie_ops = MagicMock()
sys.modules["mindie_llm.runtime.ops.mie_ops"] = mock_mie_ops

# Now import model_runner_exp with all mocks in place
if "mindie_llm.runtime.model_runner.model_runner_exp" in sys.modules:
    del sys.modules["mindie_llm.runtime.model_runner.model_runner_exp"]

from mindie_llm.runtime.model_runner import model_runner_exp  # noqa: E402
from mindie_llm.runtime.model_runner.model_runner_exp import ModelRunnerExp  # noqa: E402

# ModelRunnerExp is wrapped by @auto_speculative_method_router which replaces
# the class with a factory function.  The original class is accessible via
# __wrapped__ (set by functools.wraps).
_ModelRunnerExpClass = getattr(ModelRunnerExp, "__wrapped__", ModelRunnerExp)


class TestModelRunnerExpSourceStructure(unittest.TestCase):
    """Structural tests that verify the source has the expected decorators."""

    def test_decorator_import_exists(self):
        """The file should import the exception_handler."""
        import ast
        import inspect

        source = inspect.getsource(model_runner_exp)
        tree = ast.parse(source)

        found_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
                if "exception_handler" in names:
                    found_import = True
                    break
        self.assertTrue(found_import, "@exception_handler import not found in model_runner_exp.py")

    def test_exception_handler_decorator_before_class(self):
        """The @exception_handler decorator should appear before class ModelRunnerExp."""
        import ast
        import inspect

        source = inspect.getsource(model_runner_exp)
        tree = ast.parse(source)

        found_decorator = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerExp":
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "exception_handler":
                        found_decorator = True
                        break
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "exception_handler":
                        found_decorator = True
                        break
                break

        self.assertTrue(found_decorator, "@exception_handler decorator not found on ModelRunnerExp")

    def test_auto_speculative_method_router_present(self):
        """@auto_speculative_method_router should still be present as outer decorator."""
        import ast
        import inspect

        source = inspect.getsource(model_runner_exp)
        tree = ast.parse(source)

        found_router = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerExp":
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        if isinstance(func, ast.Attribute) and "auto_speculative_method_router" in func.attr:
                            found_router = True
                            break
                        elif isinstance(func, ast.Name) and func.id == "auto_speculative_method_router":
                            found_router = True
                            break
                break

        self.assertTrue(found_router, "@auto_speculative_method_router decorator not found on ModelRunnerExp")


class TestModelRunnerExpOomContract(unittest.TestCase):
    """Verify the OOM contract: forward/compile/load_weights are wrapped."""

    def test_forward_is_wrapped(self):
        """forward method should be wrapped by _torch_oom_handler (has __wrapped__)."""
        forward = _ModelRunnerExpClass.__dict__.get("forward")
        self.assertIsNotNone(forward)
        self.assertTrue(hasattr(forward, "__wrapped__"), "forward should be wrapped by exception_handler")

    def test_compile_is_wrapped(self):
        """compile method should be wrapped by _torch_oom_handler."""
        compile_method = _ModelRunnerExpClass.__dict__.get("compile")
        self.assertIsNotNone(compile_method)
        self.assertTrue(hasattr(compile_method, "__wrapped__"), "compile should be wrapped by exception_handler")

    def test_load_weights_is_wrapped(self):
        """load_weights method should be wrapped by _torch_oom_handler."""
        lw = _ModelRunnerExpClass.__dict__.get("load_weights")
        self.assertIsNotNone(lw)
        self.assertTrue(hasattr(lw, "__wrapped__"), "load_weights should be wrapped by exception_handler")


if __name__ == "__main__":
    unittest.main()
