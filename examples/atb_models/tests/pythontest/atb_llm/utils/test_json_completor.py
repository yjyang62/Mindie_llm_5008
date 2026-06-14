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
from ddt import ddt, data, unpack
from atb_llm.utils.json_completor import complete_json_for_toolcall, FillMode


@ddt
class TestJsonCompletor(unittest.TestCase):
    @data(
            ('\n{', {}), ('\n{"name', {}), ('\n{"name":', {}), ('\n{"name":', {}), ('\n{"name": "', {}),
            ('\n{"name": "get', {}), ('\n{"name": "get_delivery', {}), ('\n{"name": "get_delivery_date', {}),
            ('\n{"name": "get_delivery_date",', {'name': 'get_delivery_date'})
    )
    @unpack
    def test_brace_only_completion(self, text, golden_out):
        self.assertDictEqual(complete_json_for_toolcall(text, FillMode.BraceOnly), golden_out)

    @data(
            ('\n{"name": "get_delivery_date", "', {'name': 'get_delivery_date'}),
            ('\n{"name": "get_delivery_date", "arguments', {'name': 'get_delivery_date'}),
            ('\n{"name": "get_delivery_date", "arguments": {"', {'name': 'get_delivery_date', 'arguments': {}}),
            ('\n{"name": "get_delivery_date", "arguments": {"order', {'name': 'get_delivery_date', 'arguments': {}}),
            ('\n{"name": "get_delivery_date", "arguments": {"order_id": "',
             {'name': 'get_delivery_date', 'arguments': {'order_id': ''}}),
            ('\n{"name": "get_delivery_date", "arguments": {"order_id": "9998',
             {'name': 'get_delivery_date', 'arguments': {'order_id': '9998'}})
    )
    @unpack
    def test_full_completion(self, text, golden_out):
        self.assertDictEqual(complete_json_for_toolcall(text, FillMode.Full), golden_out)
    
    @data(
        ('''
        {
            "a": {
                "b": [1, 2, {
                    "c": "value",
                    "d": null
                }],
                "e": false
            },
            "f": "string"
        }
        ''', {
            "a": {
                "b": [1, 2, {
                    "c": "value",
                    "d": None
                }],
                "e": False
            },
            "f": "string"
        }),
    )
    @unpack
    def test_complex_nested_structures(self, input_str, expected):
        result = complete_json_for_toolcall(input_str, FillMode.Full)
        self.assertEqual(result, expected)
        
        incomplete = input_str.rstrip('}\n')
        result = complete_json_for_toolcall(incomplete, FillMode.BraceOnly)
        self.assertEqual(result, expected)

    @data(
        ('{"num": 123}', {"num": 123}),
        ('{"num": -123}', {"num": -123}),
        ('{"num": 12.34}', {"num": 12.34}),
        ('{"num": 1e5}', {"num": 1e5}),
        ('{"num": -1.2e-3}', {"num": -1.2e-3}),
        ('{"num": .}', {}),
    )
    @unpack
    def test_number_parsing_edge_cases(self, input_str, expected):
        result = complete_json_for_toolcall(input_str, FillMode.Full)
        self.assertEqual(result, expected)

    @data(
        ('{"b1": true}', {"b1": True}),
        ('{"b2": false}', {"b2": False}),
        ('{"n": null}', {"n": None}),
        ('{"t": tru}', {}),
        ('{"f": fals}', {}),
        ('{"n": nul}', {}),
    )
    @unpack
    def test_literal_parsing(self, input_str, expected):
        result = complete_json_for_toolcall(input_str, FillMode.Full)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()