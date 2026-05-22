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
import gc
import time
from mindie_llm.connector.common.adaptive_garbage_collector import AdaptiveGarbageCollector


class TestAdaptiveGarbageCollector(unittest.TestCase):
    def setUp(self):
        self.original_threshold = gc.get_threshold()
        AdaptiveGarbageCollector._instance = None
        self.gc_collector = AdaptiveGarbageCollector.get_instance(
            window_size=4,
            monitor_interval=0.1,
            gc_threshold_idle=[5000, 20, 20],
            gc_threshold_busy=[100000, 50, 50]
        )

    def tearDown(self):
        self.gc_collector.stop()
        gc.set_threshold(*self.original_threshold)
        AdaptiveGarbageCollector._instance = None

    def test_singleton_instance(self):
        instance1 = AdaptiveGarbageCollector.get_instance()
        instance2 = AdaptiveGarbageCollector.get_instance()
        self.assertIs(instance1, instance2)

    def test_start_stop(self):
        self.assertFalse(self.gc_collector.running)
        self.gc_collector.start()
        self.assertTrue(self.gc_collector.running)
        self.assertIsNotNone(self.gc_collector.thread)
        self.gc_collector.stop()
        self.assertFalse(self.gc_collector.running)

    def test_counter(self):
        self.assertEqual(self.gc_collector._prev_req_count, 0)
        self.assertEqual(self.gc_collector._cur_req_count, 0)
        for i in range(1, 5):
            self.gc_collector.request_counter_increase()
            self.assertEqual(self.gc_collector._prev_req_count, 0)
            self.assertEqual(self.gc_collector._cur_req_count, i)

    def test_system_is_busy(self):
        print("### test_system_is_busy ####")
        
        # init
        self.assertFalse(self.gc_collector._system_is_busy())
        
        # receiving requests
        for _ in range(5):
            self.gc_collector._sliding_window.append(1)
            print(f"sliding_window = {self.gc_collector._sliding_window}")
        self.assertTrue(self.gc_collector._system_is_busy())
        
        # not receiving requests
        for _ in range(5):
            self.gc_collector._sliding_window.append(0)
            print(f"sliding_window = {self.gc_collector._sliding_window}")
        self.assertFalse(self.gc_collector._system_is_busy())

    def test_adaptive_gc(self):
        print("### test_adaptive_gc ####")
        
        # initialization, the system is idle
        self.gc_collector.start()
        time.sleep(0.5)
        self.assertEqual(gc.get_threshold(), (5000, 20, 20))

        # receive request, the system gets busy
        for _ in range(5):
            self.gc_collector.request_counter_increase() # add request
            print(f"add request. sliding_window = {self.gc_collector._sliding_window}")
            time.sleep(0.2)
        self.assertEqual(gc.get_threshold(), (100000, 50, 50))

        # send request, the system resumes idle
        for _ in range(5):
            # do nothing
            print(f"do nothing. sliding_window = {self.gc_collector._sliding_window}")
            time.sleep(0.2)
        time.sleep(0.5)
        self.assertEqual(gc.get_threshold(), (5000, 20, 20))


if __name__ == '__main__':
    unittest.main()