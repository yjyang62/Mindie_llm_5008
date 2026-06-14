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

import time
import unittest
import gc

from mindie_llm.connector.common.gc_monitor import GCMonitor


class TestGCMonitor(unittest.TestCase):

    def setUp(self):
        """测试前初始化"""
        GCMonitor.__instance = None

    def test_singleton_pattern(self):
        instance1 = GCMonitor.get_instance()
        instance2 = GCMonitor.get_instance()

        self.assertIs(instance1, instance2)
        self.assertIsInstance(instance1, GCMonitor)

    def test_callback_initialization(self):
        monitor = GCMonitor.get_instance()
        self.assertIn(monitor._callback, gc.callbacks)

    def test_gc_start_event(self):
        monitor = GCMonitor.get_instance()
        # 模拟GC开始事件
        monitor._callback("start", {})
        time.sleep(0.01)
        monitor._callback("stop", {"generation0": 10,
                                   "generation1": 5,
                                   "generation2": 2})
        # 验证时间记录
        self.assertGreater(monitor.total_collections, 0)
