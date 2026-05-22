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

import hashlib
import unittest
import torch
from mindie_llm.text_generator.mempool.utils import flatten_tensors, str_to_md5_hex, parse_global_segment_size


class TestMemPoolConfig(unittest.TestCase):

    def test_flatten_tensors_single_tensor(self):
        t1 = torch.tensor([1, 2])
        nested = t1
        result = flatten_tensors(nested)
        self.assertEqual(result, [t1])

    def test_flatten_tensors_simple(self):
        t1 = torch.tensor([1, 2])
        t2 = torch.tensor([3, 4])
        nested = [t1, [t2]]
        result = flatten_tensors(nested)
        self.assertEqual(result, [t1, t2])

    def test_flatten_tensors_deep(self):
        t1 = torch.tensor([1])
        t2 = torch.tensor([2])
        t3 = torch.tensor([3])
        t4 = torch.tensor([3])
        nested = [[[t1, t2], [t3, t4]]]
        result = flatten_tensors(nested)
        self.assertEqual(result, [t1, t2, t3, t4])

    def test_flatten_tensors_type_error(self):
        with self.assertRaises(TypeError):
            flatten_tensors(["not", "a", "tensor"])

    def test_str_to_md5_hex(self):
        input_str = "hello"
        expected = hashlib.md5(input_str.encode("utf-8")).hexdigest()
        result = str_to_md5_hex(input_str)
        self.assertEqual(result, expected)

    def test_str_to_md5_hex_empty(self):
        input_str = ""
        expected = hashlib.md5(b"").hexdigest()
        result = str_to_md5_hex(input_str)
        self.assertEqual(result, expected)

    def test_parse_global_segment_size_basic(self):
        self.assertEqual(parse_global_segment_size('1kb'), 1024)
        self.assertEqual(parse_global_segment_size('1mb'), 1024**2)
        self.assertEqual(parse_global_segment_size('1GB'), 1024**3)
        self.assertEqual(parse_global_segment_size(100), 100)
        self.assertEqual(parse_global_segment_size('100'), 100)
        self.assertEqual(parse_global_segment_size('1.5KB'), int(1.5 * 1024))

    def test_parse_global_segment_size_error(self):
        with self.assertRaises(ValueError):
            parse_global_segment_size('abc')
        
        with self.assertRaises(ValueError):
            parse_global_segment_size("")
