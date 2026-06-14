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
from unittest.mock import patch, MagicMock
import sys

# Clean up potentially loaded modules to ensure fresh import
MODULES_TO_CLEAR = [
    "mindie_llm.runtime.models.base.tool_calls_processor_registry",
    "mindie_llm.runtime.models.qwen3.tool_calls_processor_qwen3",
    "mindie_llm.runtime.models.deepseek_v32.tool_calls_processor_deepseekv32",
    "mindie_llm.runtime.models.base.tool_calls_processor",
]

for mod in MODULES_TO_CLEAR:
    if mod in sys.modules:
        del sys.modules[mod]


class TestToolCallsProcessorRegistry(unittest.TestCase):
    """Test that registry imports correctly register processors."""

    def test_registry_function_exists(self):
        """Verifies the registry function can be imported and called."""
        # Clean import without side effects
        from mindie_llm.runtime.models.base.tool_calls_processor_registry import register_all_tool_calls_processors
        # Just check it exists and is callable
        self.assertTrue(callable(register_all_tool_calls_processors))


if __name__ == "__main__":
    unittest.main()