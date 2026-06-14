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

from atb_llm.models.qwen3.tool_call_process_qwen3 import ToolsCallProcessorQwen3


class FakeTokenizer:
    @staticmethod
    def decode(token_ids, skip_special_tokens):
        if token_ids == [151657]:
            return "<tool_call>"
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497]:
            return '<tool_call> {"name": "get_delivery_date",'
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788]:
            return '<tool_call> {"name": "get_delivery_date", "arguments":{"order_id":'
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330]:
            return '<tool_call> {"name": "get_delivery_date", "arguments":{"order_id": "'
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 24]:
            return '<tool_call> {"name": "get_delivery_date", "arguments":{"order_id": 9'
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 5212]:
            return '<tool_call> {"name": "get_delivery_date", "arguments":{"order_id": {"'
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 23, 95642]:
            return '<tool_call> {"name": "get_delivery_date", "arguments":{"order_id": "999888"}}'
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 330, 24, 24, 24, 23, 23, 23, 95642, 151658]:
            return '<tool_call> {"name": "get_delivery_date", "arguments":{"order_id": "999888"}}</tool_call>'
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 5112, 307, 788, 330, 24, 24, 24, 23, 23, 23, 30975]:
            return '<tool_call> {"name": "get_delivery_date", "arguments":{"order_id": {"id": 999888}}'
        elif token_ids == [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                           788, 5112, 307, 788, 330, 24, 24, 24, 23, 23, 23, 30975, 532]:
            return '<tool_call> {"name": "get_delivery_date", "arguments":{"order_id": {"id": 999888}}}\n'
        return "A test string"


class TestToolsCallProcessorQwen3(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_tokenizer = FakeTokenizer()
        self.parser = ToolsCallProcessorQwen3(self.mock_tokenizer)

    def test_decode(self):
        mock_content = """<tool_call>{"name": "get_delivery_date", "arguments": {"order_id": 999888}}</tool_call>"""
        result = self.parser.decode(mock_content)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("name"), "get_delivery_date")
        self.assertEqual(tool_call.get("function", {}).get("arguments"), "{\"order_id\": 999888}")

    def test_decode_stream_case1(self):
        # <tool_call>
        all_token_ids = [151657]
        prev_decode_index = 0
        curr_decode_index = 0
        skip_special_tokens = True
        delta_text = "<tool_call>"
        self.parser.current_tool_id = -1
        self.parser.current_tool_name_sent = False
        self.parser.current_tool_arguments_sent = False
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertEqual(result, {})

    def test_decode_stream_case2(self):
        # name send
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497]
        prev_decode_index = 8
        curr_decode_index = 9
        skip_special_tokens = True
        delta_text = '",'
        # self.parser current_tool_name_sent send
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = False
        self.parser.current_tool_arguments_sent = False
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("name"), "get_delivery_date")
        self.assertTrue(self.parser.current_tool_name_sent)

    def test_decode_stream_case3(self):
        # before arguments send
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788]
        prev_decode_index = 15
        curr_decode_index = 16
        skip_special_tokens = True
        delta_text = '":'
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = False
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertEqual(result, {})

    def test_decode_stream_case4(self):
        # arguments send
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330]
        prev_decode_index = 16
        curr_decode_index = 17
        skip_special_tokens = True
        delta_text = ' "'
        # self.parser current_tool_arguments_sent send
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = False
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertTrue(self.parser.current_tool_arguments_sent)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), "{\"order_id\": \"")

    def test_decode_stream_case5(self):
        # </tool_call> over
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330, 24, 24, 24, 23, 23, 23, 95642, 151658]
        prev_decode_index = 24
        curr_decode_index = 25
        skip_special_tokens = True
        delta_text = '</tool_call>'
        # </tool_call> over
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = True
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertEqual(result, {})
    
    def test_decode_stream_case_int_in_argument(self):
        # int in argument
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 24]
        prev_decode_index = 16
        curr_decode_index = 17
        skip_special_tokens = True
        delta_text = '9'

        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = False
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertTrue(self.parser.current_tool_arguments_sent)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), "{\"order_id\": 9")

    def test_decode_stream_case_object_in_argument_start(self):
        # object in argument, object start
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 5212]
        prev_decode_index = 16
        curr_decode_index = 17
        skip_special_tokens = True
        delta_text = '{"'

        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = False
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        self.assertTrue(self.parser.current_tool_arguments_sent)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), "{\"order_id\": {\"")
    
    def test_decode_stream_case_end_of_argument_remove_brace(self):
        # double brace, remove one
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 330, 24, 24, 24, 23, 23, 23, 95642]
        prev_decode_index = 23
        curr_decode_index = 24
        skip_special_tokens = True
        delta_text = '}}'

        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = True
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), "}")
    
    def test_decode_stream_case_end_of_argument_object_not_remove_brace(self):
        # remove single brace for it is the ending brace of the whole tool call json
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 5112, 307, 788, 330, 24, 24, 24, 23, 23, 23, 30975]
        prev_decode_index = 26
        curr_decode_index = 27
        skip_special_tokens = True
        delta_text = '"}}'

        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = True
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), "\"}}")

    def test_decode_stream_case_end_of_argument_object_remove_brace(self):
        # remove single brace for it is the ending brace of the whole tool call json
        all_token_ids = [151657, 198, 4913, 606, 788, 330, 455, 50562, 4164, 497, 330, 16370, 788, 5212, 1358, 842,
                         788, 5112, 307, 788, 330, 24, 24, 24, 23, 23, 23, 30975, 532]
        prev_decode_index = 27
        curr_decode_index = 28
        skip_special_tokens = True
        delta_text = '}\n'

        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.current_tool_arguments_sent = True
        result = self.parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens,
                                           delta_text)
        tool_call = result.get("tool_calls", [])[0]
        self.assertEqual(tool_call.get("function", {}).get("arguments"), "")

if __name__ == '__main__':
    unittest.main()