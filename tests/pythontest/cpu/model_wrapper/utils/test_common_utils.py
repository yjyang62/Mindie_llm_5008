#!/usr/bin/env python
# coding=utf-8
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

import numpy as np

from mindie_llm.model_wrapper.utils.common_util import (
    get_ipv4_obj, get_ipv6_obj, get_ip_obj,
    ipv4_to_list, ipv6_to_list, ip_string_to_list,
    ip_array_to_ipv4, ip_array_to_ipv6, ip_array_to_string
)
from mindie_llm.model_wrapper.utils.common_util import (
    TransferType, generate_lora_strings, generate_user_request_id_string,
    generate_mem_pool_decisions, generate_dp_inst_id, split_list_equally
)


class TestIPFunctions(unittest.TestCase):

    def test_get_ipv4_obj_valid(self):
        # 测试有效的 IPv4 地址
        ipv4 = get_ipv4_obj('192.128.1.1', 'ip')
        self.assertEqual(str(ipv4), '192.128.1.1')
    
    def test_get_ipv4_obj_invalid(self):
        # 测试无效的 IPv4 地址
        with self.assertRaises(ValueError) as context:
            _ = get_ipv4_obj('256.128.1.1', 'ip')
        
        self.assertIn("ip=256.128.1.1 is invalid IPv4 address.", str(context.exception))
    
    def test_get_ipv6_obj_valid(self):
        # 测试有效的 IPv6 地址
        ipv6 = get_ipv6_obj('2001:0db8:85a3:0000:0000:8a2e:0370:7334', 'ip')
        self.assertEqual(str(ipv6), '2001:db8:85a3::8a2e:370:7334')
    
    def test_get_ipv6_obj_invalid(self):
        # 测试无效的 IPv6 地址
        with self.assertRaises(ValueError) as context:
            _ = get_ipv6_obj('2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234', 'ip')
        
        self.assertIn("ip=2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234 is invalid IPv6 address.",
                      str(context.exception))
    
    def test_get_ip_obj_valid_ipv4(self):
        # 测试有效的 IPv4 地址
        ip_obj = get_ip_obj('192.128.1.1', 'ip')
        self.assertEqual(str(ip_obj), '192.128.1.1')
    
    def test_get_ip_obj_valid_ipv6(self):
        # 测试有效的 IPv6 地址
        ip_obj = get_ip_obj('2001:0db8:85a3:0000:0000:8a2e:0370:7334', 'ip')
        self.assertEqual(str(ip_obj), '2001:db8:85a3::8a2e:370:7334')
    
    def test_get_ip_obj_invalid(self):
        # 测试无效的 IP 地址
        with self.assertRaises(ValueError) as context:
            _ = get_ip_obj('1234', 'ip')
        
        self.assertIn("ip=1234 is invalid IP address.", str(context.exception))
    
    def test_ipv4_to_list_valid(self):
        # 测试有效的 IPv4 地址
        ip_list = ipv4_to_list('192.128.1.1')
        self.assertEqual(ip_list, [192, 128, 1, 1, -1, -1, -1, -1])
    
    def test_ipv4_to_list_format_error(self):
        # 测试格式错误的 IPv4 地址
        with self.assertRaises(ValueError) as context:
            _ = ipv4_to_list('192.128.1')
        
        self.assertIn("192.128.1 is invalid IPv4 format.", str(context.exception))
    
    def test_ipv4_to_list_item_not_number(self):
        # 测试元素不是数字的 IPv4 地址
        with self.assertRaises(ValueError) as context:
            _ = ipv4_to_list('192.128.1.a')
        
        self.assertIn("IPv4 segment 'a' is not a valid number.", str(context.exception))
    
    def test_ipv4_to_list_item_out_of_range(self):
        # 测试元素超出范围 [0, 255] 的 IPv4 地址
        with self.assertRaises(ValueError) as context:
            _ = ipv4_to_list('192.128.1.256')
        
        self.assertIn("IPv4 segment '256' out of range [0, 255].", str(context.exception))
    
    def test_ipv6_to_list_valid(self):
        # 测试有效的 IPv6 地址
        ip_list = ipv6_to_list('2001:0db8:85a3:0000:0000:8a2e:0370:7334')
        self.assertEqual(ip_list, [0x2001, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334])
    
    def test_ipv6_to_list_invalid(self):
        # 测试无效的 IPv6 地址
        with self.assertRaises(ValueError) as context:
            _ = ipv6_to_list('2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234')
        
        self.assertIn("2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234 is invalid IPv6 address.", str(context.exception))
    
    def test_ip_string_to_list_valid_ipv4(self):
        # 测试有效的 IPv4 地址
        ip_list = ip_string_to_list('192.128.1.1')
        self.assertEqual(ip_list, [192, 128, 1, 1, -1, -1, -1, -1])
    
    def test_ip_string_to_list_valid_ipv6(self):
        # 测试有效的 IPv6 地址
        ip_list = ip_string_to_list('2001:0db8:85a3:0000:0000:8a2e:0370:7334')
        self.assertEqual(ip_list, [0x2001, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334])
    
    def test_ip_string_to_list_invalid(self):
        # 测试无效的 IP 地址
        with self.assertRaises(ValueError) as context:
            _ = ip_string_to_list('1234')
        
        self.assertIn("1234 is invalid IP address.", str(context.exception))
    
    def test_ip_array_to_ipv4_valid(self):
        # 测试有效的 IPv4 数组
        ipv4_str = ip_array_to_ipv4([192, 128, 1, 1, -1, -1, -1, -1])
        self.assertEqual(ipv4_str, '192.128.1.1')
    
    def test_ip_array_to_ipv4_invalid(self):
        # 测试无效的 IPv4 数组
        with self.assertRaises(ValueError) as context:
            _ = ip_array_to_ipv4([256, 128, 1, 1, -1, -1, -1, -1])
        
        self.assertIn("IPv4 segment '256' is out of range [0, 255].", str(context.exception))
    
    def test_ip_array_to_ipv6_valid(self):
        # 测试有效的 IPv6 数组
        ipv6_str = ip_array_to_ipv6([0x2001, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334])
        self.assertEqual(ipv6_str, '2001:db8:85a3::8a2e:370:7334')
    
    def test_ip_array_to_ipv6_invalid(self):
        # 测试无效的 IPv6 数组
        with self.assertRaises(ValueError) as context:
            _ = ip_array_to_ipv6([0x20010, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334])
        
        self.assertIn("IPv6 segment '0x20010' is out of range [0, 0xFFFF].", str(context.exception))
    
    def test_ip_array_to_string_valid_ipv4(self):
        # 测试有效的 IPv4 数组
        ip_str = ip_array_to_string([192, 128, 1, 1, -1, -1, -1, -1])
        self.assertEqual(ip_str, '192.128.1.1')
    
    def test_ip_array_to_string_valid_ipv6(self):
        # 测试有效的 IPv6 数组
        ip_str = ip_array_to_string([0x2001, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334])
        self.assertEqual(ip_str, '2001:db8:85a3::8a2e:370:7334')
    
    def test_ip_array_to_string_invalid_array_len(self):
        # 测试无效的数组长度
        with self.assertRaises(ValueError) as context:
            _ = ip_array_to_string([192, 128, 1, 1, -1, -1, -1])
        
        self.assertIn("ip_array must be an array of 8 integers.", str(context.exception))


class TestGenerateLoraStrings(unittest.TestCase):

    def test_generate_lora_strings_lora_id_is_none(self):
        class MockRequest:
            def get_tensor_by_name(self, name):
                return "None"
        request = MockRequest()
        result = generate_lora_strings(request)
        self.assertIsNone(result)
    
    def test_generate_lora_strings_lora_id_is_valid(self):
        class MockRequest:
            def get_tensor_by_name(self, name):
                return "Lora1"
        request = MockRequest()
        result = generate_lora_strings(request)
        self.assertEqual(result, "Lora1")


class TestGenerateUserRequestIdString(unittest.TestCase):

    def test_generate_user_request_id_string_valid(self):
        class MockRequest:
            def get_tensor_by_name(self, name):
                return np.array([[65, 66, 67], [68, 69, 70]], dtype=np.uint8)
        request = MockRequest()
        result = generate_user_request_id_string(request)
        self.assertEqual(result, "ABC")
    
    def test_generate_user_request_id_string_invalid(self):
        class MockRequest:
            def get_tensor_by_name(self, name):
                return np.array([1, 2, 3], dtype=np.uint8)
        request = MockRequest()
        result = generate_user_request_id_string(request)
        self.assertIsNone(result)


class TestGenerateMemPoolDecisions(unittest.TestCase):
    
    def test_generate_mem_pool_decisions_valid_transfer_tensor(self):
        class MockRequest:
            def get_tensor_by_name(self, name):
                return np.array([b'[1, 1, 1]'], dtype=np.string_)
        requests = [MockRequest()]
        transfer_operation = TransferType.H2D
        excepted_array = np.array([[1, 1, 1]])
        result = generate_mem_pool_decisions(requests, transfer_operation)
        self.assertTrue(np.array_equal(result, excepted_array))
    
    def test_generate_mem_pool_decisions_invalid_transfer_tensor(self):
        class MockRequest:
            def get_tensor_by_name(self, name):
                return None
        requests = [MockRequest()]
        transfer_operation = TransferType.H2D
        result = generate_mem_pool_decisions(requests, transfer_operation)
        self.assertIsNone(result)
    
    def test_generate_mem_pool_decisions_invalid_transfer_operation(self):
        class MockRequest:
            def get_tensor_by_name(self, name):
                return np.array([65, 66, 67, 68, 69, 70], dtype=np.uint8)
        requests = [MockRequest()]
        transfer_operation = "invalid"
        result = generate_mem_pool_decisions(requests, transfer_operation)
        self.assertIsNone(result)
    
    def test_generate_mem_pool_decisions_requests_is_none(self):
        requests = None
        transfer_operation = TransferType.H2D
        with self.assertRaises(ValueError) as context:
            _ = generate_mem_pool_decisions(requests, transfer_operation)
        
        self.assertIn("requests is not set.", str(context.exception))
    
    def test_generate_mem_pool_decisions_decode_exception(self):
        class MockRequest:
            def get_tensor_by_name(self, name):
                return np.array([b'[1]'], dtype=np.string_)
        requests = [MockRequest()]
        transfer_operation = TransferType.H2D
        result = generate_mem_pool_decisions(requests, transfer_operation)
        self.assertIsNone(result)


class TestGenerateDpInstId(unittest.TestCase):

    def test_generate_dp_inst_id(self):
        inst_id = "123"
        dp_size = 3
        expected_list = ["1230000000", "1230000001", "1230000002"]
        result = generate_dp_inst_id(inst_id, dp_size)
        self.assertEqual(result, expected_list)


class TestSplitListEqually(unittest.TestCase):

    def test_split_list_equally_valid_split(self):
        lst = [1, 2, 3, 4, 5, 6]
        n = 3
        expected_list = [[1, 2], [3, 4], [5, 6]]
        result = split_list_equally(lst, n)
        self.assertEqual(result, expected_list)
    
    def test_split_list_equally_invalid_n(self):
        lst = [1, 2, 3, 4, 5, 6]
        n = 0
        with self.assertRaises(ValueError) as context:
            _ = split_list_equally(lst, n)
        
        self.assertIn("Number of chunks 0 must be greater than 0", str(context.exception))
    
    def test_split_list_equally_uneven_split(self):
        lst = [1, 2, 3, 4, 5, 6]
        n = 4
        with self.assertRaises(ValueError) as context:
            _ = split_list_equally(lst, n)
        
        self.assertIn("Length 6 of the list cannot be divided evenly by 4", str(context.exception))


if __name__ == "__main__":
    unittest.main()
