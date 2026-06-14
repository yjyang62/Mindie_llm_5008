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

from atb_llm.models.deepseekv2.tool_call_process_deepseekv2 import ToolsCallProcessorDeepseekv3
from ddt import ddt, data, unpack


class FakeTokenizer:
    @staticmethod
    def decode(token_ids, skip_special_tokens):
        if token_ids == [128808]:
            return "<｜tool▁call▁begin｜>"
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455]:
            return '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164]:
            return '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date\n```'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497]:
            return '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date\n```json'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788]:
            return '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date' + \
            '\n```json\n{"order_id":'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330]:
            return """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            '\n```json\n{"order_id": "'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24]:
            return """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            '\n```json\n{"order_id": "999888'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 23]:
            return """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            '\n```json\n{"order_id": "999888"}\n'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 23, 95642]:
            return """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            '\n```json\n{"order_id": "999888"}\n```'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 25, 23, 95642]:
            return """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            '\n```json\n{"order_id": "999 888"}\n```'
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 25]:
            return """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            '\n```json\n{"order_id": "999 '
        elif token_ids == [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 23, 95642, 128809]:
            return """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            """\n```json\n{"order_id": "999888"}\n```<｜tool▁call▁end｜>"""
        return "A test string"


@ddt
class TestToolsCallProcessorDeepseekv2(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_tokenizer = FakeTokenizer()
        self.parser = ToolsCallProcessorDeepseekv3(self.mock_tokenizer)

    def test_tool_call_regex(self):
        mock_content = """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            """\n```json\n{"order_id": "999888"}\n```<｜tool▁call▁end｜>"""
        match = self.parser.tool_call_regex.findall(mock_content)
        tool_type, name, arguments = match[0]
        self.assertEqual(tool_type, "function")
        self.assertEqual(name, "get_delivery_date")
        self.assertEqual(arguments, '{"order_id": "999888"}')

    def test_stream_tool_call_portion_regex(self):
        mock_content = """function<｜tool▁sep｜>get_delivery_date""" + \
            """\n```json\n{"order_id": "999888"}\n"""
        current_tool_call_matches = (self.parser.stream_tool_call_portion_regex.match(mock_content))
        tool_type, tool_name, tool_args = (
                                current_tool_call_matches.groups())
        self.assertEqual(tool_type, "function")
        self.assertEqual(tool_name, "get_delivery_date")
        self.assertEqual(tool_args, '{"order_id": "999888"}')

    def test_stream_tool_call_name_regex(self):
        mock_content = """function<｜tool▁sep｜>get_delivery_date""" + \
            """\n```json\n{"order_id": "99"""
        current_tool_call_matches = (self.parser.stream_tool_call_portion_regex.match(mock_content))
        tool_type, tool_name, _ = (
                                current_tool_call_matches.groups())
        self.assertEqual(tool_type, "function")
        self.assertEqual(tool_name, "get_delivery_date")

    def test_get_tool_call_json(self):
        mock_matches = [('function', 'get_delivery_date', '{"order_id": "999888"}')]
        tool_call = self.parser.get_tool_call_json(mock_matches)[0]
        self.assertEqual(tool_call["name"], 'get_delivery_date')
        self.assertEqual(tool_call["arguments"], '{"order_id": "999888"}')

    def test_decode(self):
        mock_content = """<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_delivery_date""" + \
            """\n```json\n{"order_id": "999888"}\n```<｜tool▁call▁end｜>"""
        result = self.parser.decode(mock_content)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("name"), "get_delivery_date")
        self.assertEqual(tool_call.get("function", {}).get("arguments"), "{\"order_id\": \"999888\"}")

    @data(([128808], "<｜tool▁call▁begin｜>"))
    @unpack
    def test_decode_stream_case_callbegin(self, all_token_ids, delta_text):
        # <｜tool▁call▁begin｜>
        prev_decode_index = 0
        curr_decode_index = 0
        skip_special_tokens = True
        self.parser.current_tool_id = -1
        self.current_tool_name_sent = False
        self.current_tool_arguments_sent = False
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertEqual(result, {})

    @data(([128808, 198, 4913, 606, 788, 330, 455], "```"))
    @unpack
    def test_decode_stream_case_namesend1(self, all_token_ids, delta_text):
        # name send
        prev_decode_index = 0
        curr_decode_index = 8
        skip_special_tokens = True
        # self.parser current_tool_name_sent send
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = False
        self.parser.current_tool_arguments_sent = False
        self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertTrue(not self.parser.current_tool_name_sent)
        self.assertTrue(not self.parser.current_tool_arguments_sent)

    @data(([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164], "```", "get_delivery_date"),
        ([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164], 'json', "get_delivery_date"))
    @unpack
    def test_decode_stream_case_namesend2(self, all_token_ids, delta_text, golden_name):
        # name send
        prev_decode_index = 0
        curr_decode_index = 8
        skip_special_tokens = True
        # self.parser current_tool_name_sent send
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = False
        self.parser.current_tool_arguments_sent = False
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                        delta_text)
        self.assertTrue(self.parser.current_tool_name_sent)
        self.assertTrue(not self.parser.current_tool_arguments_sent)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("name"), golden_name)

    @data(([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330], ' "', "{\"order_id\": \""),
          ([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330, 24, 24], '999888', "{\"order_id\": \"999888"))
    @unpack
    def test_decode_stream_case_argsend1(self, all_token_ids, delta_text, golden_name):
        # arguments send
        prev_decode_index = 16
        curr_decode_index = 17
        skip_special_tokens = True
        # self.parser current_tool_arguments_sent send
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = False

        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertTrue(self.parser.current_tool_name_sent)
        self.assertTrue(self.parser.current_tool_arguments_sent)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), golden_name)


    @data(([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330, 24, 24, 24, 23, 23, 23, 95642], '\"}\n```', False, "{\"order_id\": \"999888\"}"),
          ([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330, 24, 24, 24, 23, 23, 23], '\n```json\n{"order_id": "999888"}\n',
                         False, "{\"order_id\": \"999888\"}"),
          ([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330, 24, 24, 24, 23, 23, 23, 95642], "```",
                         False, "{\"order_id\": \"999888\"}"),
          ([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 25, 23, 95642], "```",
                         False, "{\"order_id\": \"999 888\"}"),
          ([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 25], "999 ",
                         False, "{\"order_id\": \"999 "),
        )
    @unpack
    def test_decode_stream_argesend2(self, all_token_ids, delta_text, current_tool_arguments_sent, golden_name):
        # arguments send
        prev_decode_index = 16
        curr_decode_index = 17
        skip_special_tokens = True
        # self.parser current_tool_arguments_sent send
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = current_tool_arguments_sent

        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertTrue(self.parser.current_tool_name_sent)
        self.assertTrue(self.parser.current_tool_arguments_sent)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), golden_name)

    @data(
        ([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330], ": \"",
        [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                        788, 330, 24, 24], "999888",
        "999888"),
        ([128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                        788, 330, 24, 24], "999888",
        [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                        788, 330, 24, 24, 24, 23, 23, 23], "\"}\n",
        "\"}")
    )
    @unpack
    def test_decode_stream_argesend3(self, prev_all_token_ids, prev_delta,
        current_all_token_ids, cur_delta, golden_name):
        # arguments send
        prev_decode_index = 16
        curr_decode_index = 17
        skip_special_tokens = True
        # self.parser current_tool_arguments_sent send
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = False

        result = self.parser.decode_stream(prev_all_token_ids, prev_decode_index,
            curr_decode_index, skip_special_tokens, prev_delta)
        self.assertTrue(self.parser.current_tool_name_sent)
        self.assertTrue(self.parser.current_tool_arguments_sent)
        result = self.parser.decode_stream(current_all_token_ids, prev_decode_index + 1, curr_decode_index + 2, 
            skip_special_tokens, cur_delta)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), golden_name)

    def test_decode_stream_over(self):
        # <｜tool▁call▁end｜> over
        all_token_ids = [128808, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330, 24, 24, 24, 23, 23, 23, 95642, 128809]
        prev_decode_index = 24
        curr_decode_index = 25
        skip_special_tokens = True
        delta_text = '<｜tool▁call▁end｜>'
        # <｜tool▁call▁end｜> over
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = True
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()