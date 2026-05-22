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
import json
from atb_llm.models.llama.tool_call_process_llama import ToolsCallProcessorLlama


class TestToolsCallProcessorLlama(unittest.TestCase):
    def test_function_call_decode(self):
        tool_call_processor = ToolsCallProcessorLlama("llama3.1")
        function_call_content = {"name": "get_delivery_date",
                                 "parameters": {"order_id": "999888"}}
        function_call_content = tool_call_processor.decode(json.dumps(function_call_content))
        tool_call_list = function_call_content.get('tool_calls', None)
        self.assertIsNotNone(tool_call_list)
        current_tool_call = tool_call_list[0]
        self.assertIn('type', current_tool_call)
        self.assertIn('id', current_tool_call)
        self.assertIn('function', current_tool_call)

if __name__ == "__main__":
    unittest.main()