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
import torch
from mindie_llm.text_generator.mempool.base import MemPool


class TestMempoolBase(unittest.TestCase):
    def test_abstract_or_notimplemented_contract(self):
        m = MemPool()
        data = torch.rand(2, 3)
        with self.assertRaises(NotImplementedError):
            m.exists("x")

        with self.assertRaises(NotImplementedError):
            m.put("x", [data])

        with self.assertRaises(NotImplementedError):
            m.get("x", [data])