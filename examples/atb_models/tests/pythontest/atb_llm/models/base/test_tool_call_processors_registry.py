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
from unittest.mock import patch, MagicMock

from atb_llm.models.base.tool_call_parser import ToolParserManager, ToolsCallProcessor


IMPORTED_MODULES_NAME = [
    'chatglm2_6b', 'chatglm_v2_6b', 'chatglm_v2', 'chatglm2',  # ToolsCallProcessorChatglmV2
    'chatglm3_6b', 'chatglm_v3_6b', 'chatglm_v3', 'chatglm3',  # ToolsCallProcessorChatglmV3
    'chatglm4_9b', 'chatglm_v4_9b', 'glm_4', 'glm_4_9b', 'chatglm',  # ToolsCallProcessorChatglmV4
    'qwen1_5', 'qwen_1_5', 'qwen2', 'qwen_2', 'qwen1_5_or_2', 'qwen_1_5_or_2',  # ToolsCallProcessorQwen1_5_or_2
    'qwen2_5', 'qwen_2_5', 'qwen_auto',  # ToolsCallProcessorQwen2_5
    'qwen3', 'qwen3_moe', 'hermes',  # ToolsCallProcessorQwen3
    'qwen3_coder', 'qwen3coder',  # ToolsCallProcessorQwen3Coder
    'deepseek_v2', 'deepseek_v3', 'deepseekv2', 'deepseekv3',  # ToolsCallProcessorDeepseekV3
    'deepseek_v31', 'deepseekv31',  # ToolsCallProcessorDeepseekV31
    'llama', 'llama3', 'llama3_1'  # ToolsCallProcessorLlama
]


class TestToolCallProcessorsRegistry(unittest.TestCase):
    def test_tool_call_processors_registry(self):
        from atb_llm.models.base import tool_call_processors_registry
        imported_modules = ToolParserManager.get_tool_call_processors()
        self.assertEqual(list(imported_modules.keys()), IMPORTED_MODULES_NAME)

        for module_name in list(imported_modules):
            ToolParserManager.remove_tool_call_processor(module_name)
        self.assertEqual(imported_modules, {})

    def test_tool_parser_manager(self):
        # using as decorator
        @ToolParserManager.register_module(["fake_decorator"])
        class FakeDecoratorToolsCallProcessor(ToolsCallProcessor):
            def __init__(self):
                super().__init__(model_version="fake_decorator")

        # using as function
        class FakeFunctionToolsCallProcessor(ToolsCallProcessor):
            def __init__(self):
                super().__init__(model_version='fake_function')

        ToolParserManager.register_module(module_names=["fake_function"], module=FakeFunctionToolsCallProcessor)
        ToolParserManager.register_module(module=FakeFunctionToolsCallProcessor)

        self.assertIs(
            ToolParserManager.get_tool_call_processor("fake_decorator"), FakeDecoratorToolsCallProcessor
        )
        self.assertIs(
            ToolParserManager.get_tool_call_processor("fake_function"), FakeFunctionToolsCallProcessor
        )
        self.assertIs(
            ToolParserManager.get_tool_call_processor("FakeFunctionToolsCallProcessor"), FakeFunctionToolsCallProcessor
        )

        with patch('atb_llm.models.base.tool_call_parser.logger') as mock_logger:
            ToolParserManager.register_module(
                module_names=True, module=FakeFunctionToolsCallProcessor
            )
            mock_logger.warning.assert_called_with(
                f"'module_names' must be None, str or a list of str, but got '<class 'bool'>', "
                f"will skip this module register"
            )

        with patch('atb_llm.models.base.tool_call_parser.logger') as mock_logger:
            ToolParserManager.register_module(
                module_names="fake_function", force=False, module=FakeFunctionToolsCallProcessor
            )
            mock_logger.warning.assert_called_with(
                f"'fake_function' is already registered with module 'FakeFunctionToolsCallProcessor', "
                f"will skip this module register"
            )

        with self.assertRaises(TypeError):
            ToolParserManager.register_module(module_names="wrong_module", module=MagicMock)

        with self.assertRaises(TypeError):
            ToolParserManager.register_module(force="wrong_type", module=FakeFunctionToolsCallProcessor)

        with self.assertRaises(KeyError):
            ToolParserManager.get_tool_call_processor("non_existent_processor")


if __name__ == "__main__":
    unittest.main()