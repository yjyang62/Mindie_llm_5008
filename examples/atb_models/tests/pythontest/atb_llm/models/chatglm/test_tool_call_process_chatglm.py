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

from atb_llm.models.chatglm.tool_call_process_chatglm import \
    ToolsCallProcessorChatglmV2, ToolsCallProcessorChatglmV3, ToolsCallProcessorChatglmV4


@ddt
class TestToolsCallProcessorQwen2(unittest.TestCase):
    def setUp(self):
        tokenizer = MagicMock()
        self.tool_parser_chatglmv2 = ToolsCallProcessorChatglmV2(tokenizer)
        self.tool_parser_chatglmv3 = ToolsCallProcessorChatglmV3(tokenizer)
        self.tool_parser_chatglmv4 = ToolsCallProcessorChatglmV4(tokenizer)
        self.origin_content_2 = """get_weather\n{"location": "Beijing, China", "unit": "celsius"}"""
        self.origin_content_3 = """get_weather\n ```python\ntool_call(location='Beijing', unit='celsius')\n```"""
        self.origin_content_4 = """get_weather\n{"location": "Beijing, China", "unit": "celsius"}"""

    def test_decode_chatglmv2(self):
        result = self.tool_parser_chatglmv2.decode(self.origin_content_2)
        self.assertIsNotNone(result.get("content"))

    def test_decode_chatglmv3(self):
        result = self.tool_parser_chatglmv3.decode(self.origin_content_3)
        self.assertIsNotNone(result.get("tool_calls"))

    def test_decode_chatglmv4(self):
        result = self.tool_parser_chatglmv4.decode(self.origin_content_4)
        self.assertIsNotNone(result.get("tool_calls"))


if __name__ == "__main__":
    unittest.main()