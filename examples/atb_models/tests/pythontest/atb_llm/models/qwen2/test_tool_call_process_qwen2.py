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
from unittest.mock import MagicMock

from ddt import ddt

from atb_llm.models.qwen2.tool_call_process_qwen2 import ToolsCallProcessorQwen1_5_or_2, ToolsCallProcessorQwen2_5


@ddt
class TestToolsCallProcessorQwen2(unittest.TestCase):
    def setUp(self):
        tokenizer = MagicMock()
        self.tools_obj_2 = ToolsCallProcessorQwen1_5_or_2(tokenizer)
        self.tools_obj_2_5 = ToolsCallProcessorQwen2_5(tokenizer)
        self.origin_content_2_5 = """the user is asking for the delivery date of their order with ID 12345. </think>
        <tool_call>
        {"name": "get_delivery_date", "arguments": {"order_id": "12345"}}
        </tool_call>
        """
        self.origin_content_2 = "✿FUNCTION✿get_rectangle_property" \
                                "✿ARGS✿{\"perimeter\": 14, \"area\": 15, \"property\": \"length\"}"

    def test_decode_qwen2_5(self):
        result = self.tools_obj_2_5.decode(self.origin_content_2_5)
        self.assertIsNotNone(result.get("content"))
        self.assertIsNotNone(result.get("tool_calls"))

    def test_decode_qwen2(self):
        result = self.tools_obj_2.decode(self.origin_content_2)
        self.assertIsNotNone(result.get("tool_calls"))