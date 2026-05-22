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

from atb_llm.models.qwen3_moe.tool_call_process_qwen3_coder import ToolsCallProcessorQwen3Coder
from ddt import ddt, data, unpack


@ddt
class TestToolsCallProcessorDeepseekv2(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_tokenizer = MagicMock()
        self.tool_parser = ToolsCallProcessorQwen3Coder(self.mock_tokenizer)
        self.tool_parser.tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        },
                        "role": {
                            "type": "int",
                        },
                        "test_bool":{
                            "type": "bool"
                        },
                        "test_list":{
                            "type": "list"
                        },
                        "test_float":{
                            "type": "float"
                        }
                    },
                    "required": [
                        "location",
                        "unit"
                    ]
                }
            }
        }]

    def test_property(self):
        self.assertIsInstance(self.tool_parser.tool_call_start_token, str)
        self.assertIsInstance(self.tool_parser.tool_call_end_token, str)
        self.assertIsInstance(self.tool_parser.tool_call_start_token_id, int)
        self.assertIsInstance(self.tool_parser.tool_call_end_token_id, int)

    def test_get_tool_call_json(self):
        mock_content = [
            "<tool_call>\n"
            "<function=get_current_weather>\n"
            "<parameter=location>\n"
            "Boston\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>\n"
            "<parameter=role>\n"
            "1\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ]
        result = self.tool_parser.get_tool_call_json(mock_content)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], "get_current_weather")
        self.assertEqual(result[0]['arguments']['location'], "Boston")
        self.assertEqual(result[0]['arguments']['role'], 1)

    def test_get_tool_call_json_batch2(self):
        mock_content = [
            "<tool_call>\n"
            "<function=get_current_weather>\n"
            "<parameter=location>\n"
            "Boston\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>\n"
            "<parameter=role>\n"
            "-2\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ]
        mock_content.extend(mock_content)
        result = self.tool_parser.get_tool_call_json(mock_content)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], "get_current_weather")
        self.assertEqual(result[0]['arguments']['location'], "Boston")
        self.assertEqual(result[0]['arguments']['role'], -2)
        
    def test_get_tool_call_json_type_error(self):
        mock_content = [
            "<tool_call>\n"
            "<function=get_current_weather>\n"
            "<parameter=role>\n"
            "1e2\n"
            "</parameter>\n"
            "<parameter=test_bool>\n"
            "ttrue\n"
            "</parameter>\n"
            "<parameter=test_list>\n"
            "[1, 2, -3, 'abc', [-2, 3]]\n"
            "</parameter>\n"
            "<parameter=test_float>\n"
            "1e3\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ]
        result = self.tool_parser.get_tool_call_json(mock_content)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], "get_current_weather")
        self.assertEqual(result[0]['arguments']['role'], "1e2")
        self.assertFalse(result[0]['arguments']['test_bool'])
        self.assertEqual(len(result[0]['arguments']['test_list']), 5)
        self.assertEqual(result[0]['arguments']['test_float'], 1000)

    def test_extract_single_tool_call(self):
        mock_content = (
            "<function=get_current_weather>\n"
            "<parameter=location>\n"
            "Boston\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>\n"
            "</function>"
        )
        result = self.tool_parser.extract_single_tool_call(mock_content)
        self.assertEqual(result['name'], "get_current_weather")
        self.assertEqual(result['arguments']['location'], "Boston")

    def test_extract_arguments(self):
        mock_content = (
            "<parameter=location>\n"
            "Boston\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>"
        )
        result = self.tool_parser.extract_arguments("get_current_weather", mock_content)
        self.assertEqual(result['location'], "Boston")
        self.assertEqual(result['unit'], "celsius")

    def test_decode_stream_tool_call_case1(self):
        tool_call_portion_dict = {}
        tool_call_portion_dict['tool_call_portion'] = (
            "<tool_call>\n"
            "<function=get_current_weather>"
        )
        self.tool_parser.current_tool_name_sent = False
        result = self.tool_parser._decode_stream_tool_call(tool_call_portion_dict)
        self.assertTrue(self.tool_parser.current_tool_name_sent)
        self.assertEqual(result['tool_calls'][0]['function']['name'], "get_current_weather")

    def test_decode_stream_tool_call_case2(self):
        tool_call_portion_dict = {}
        tool_call_portion_dict['tool_call_portion'] = (
            "<tool_call>\n"
            "<function=get_current_weather>\n"
            "<parameter=location>\n"
            "Boston\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>\n"
            "</function>"
        )
        self.tool_parser.current_tool_name_sent = True
        result = self.tool_parser._decode_stream_tool_call(tool_call_portion_dict)
        self.assertEqual(result['tool_calls'][0]['function']['arguments'], "}")

    def test_parse_tool_call_portion_to_json_func_name(self):
        tool_call_portion = (
            "<function=get_current_weather>\n"
            "<parameter=location>"
        )
        self.tool_parser.current_tool_name_sent = True
        result = self.tool_parser._parse_tool_call_portion_to_json(tool_call_portion)
        self.assertEqual(result['name'], "get_current_weather")
        self.assertEqual(result.get('arguments'), None)

    def test_parse_tool_call_portion_to_json_argument_case1(self):
        tool_call_portion = (
            "<function=get_current_weather>\n"
            "<parameter=location>\n"
            "Boston\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>"
        )
        self.tool_parser.current_tool_name_sent = True
        result = self.tool_parser._parse_tool_call_portion_to_json(tool_call_portion)
        self.assertEqual(result['name'], "get_current_weather")
        self.assertEqual(result['arguments'], ", \"unit\": \"celsius\"")

    def test_parse_tool_call_portion_to_json_argument_case2(self):
        tool_call_portion = (
            "<function=get_current_weather>\n"
            "<parameter=location>\n"
            "Boston\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>\n"
            "<parameter=role>\n"
            "-2\n"
            "</parameter>\n"
        )
        self.tool_parser.current_tool_name_sent = True
        result = self.tool_parser._parse_tool_call_portion_to_json(tool_call_portion)
        self.assertEqual(result['name'], "get_current_weather")
        self.assertEqual(result['arguments'], ", \"role\": -2")

    def test_parse_tool_call_portion_to_json_argument_case3(self):
        tool_call_portion = (
            "<function=get_current_weather>\n"
            "<parameter=location>\n"
            "Boston\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>\n"
            "<parameter=test_bool>\n"
            "true\n"
            "</parameter>\n"
        )
        self.tool_parser.current_tool_name_sent = True
        result = self.tool_parser._parse_tool_call_portion_to_json(tool_call_portion)
        self.assertEqual(result['name'], "get_current_weather")
        self.assertEqual(result['arguments'], ", \"test_bool\": true")

    def test_convert_param_value(self):
        self.assertEqual(self.tool_parser._convert_param_value(
            param_value="NULL",
            param_name="test",
            param_config={},
            func_name="test_func"
        ), None)
        self.assertEqual(self.tool_parser._convert_param_value(
            param_value="123",
            param_name="test",
            param_config={},
            func_name="test_func"
        ), "123")
        self.assertEqual(self.tool_parser._convert_param_value(
            param_value="123",
            param_name="test",
            param_config={"test2": {"type": "float"}},
            func_name="test_func"
        ), "123")
        self.assertEqual(self.tool_parser._convert_param_value(
            param_value="1abc3",
            param_name="test",
            param_config={"test": {"type": "float"}},
            func_name="test_func"
        ), "1abc3")
        self.assertEqual(self.tool_parser._convert_param_value(
            param_value="1e4",
            param_name="test",
            param_config={"test": {"type": "float"}},
            func_name="test_func"
        ), 10000)        
        self.assertEqual(self.tool_parser._convert_param_value(
            param_value="[error_json, a, b ,c]",
            param_name="test",
            param_config={"test": {"type": "unknow"}},
            func_name="test_func"
        ), "[error_json, a, b ,c]")
        self.assertEqual(self.tool_parser._convert_param_value(
            param_value="['right_json', 1, 2, 3]",
            param_name="test",
            param_config={"test": {"type": "unknow"}},
            func_name="test_func"
        ), ['right_json', 1, 2, 3])

if __name__ == '__main__':
    unittest.main()