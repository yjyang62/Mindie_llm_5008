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

import os
import unittest
from unittest.mock import patch
from mindie_llm.model_wrapper.utils.metrics import FileMetrics



class TestFileMetrics(unittest.TestCase):
    def setUp(self):
        os.environ.clear()
        os.environ.update({
            "MODEL_WRAPPER_METRICS_OUTPUT_ENABLE": "0"
        })

    def test_metric_disable(self):
        self.metrics = FileMetrics()
    
    @patch.dict(os.environ, {"MODEL_WRAPPER_METRICS_OUTPUT_ENABLE": "1"})
    def test_metric_enable(self):
        self.metrics = FileMetrics()
        self.assertTrue(self.metrics.metric_enable)

        event_str = "test_event"
        flag = "flag"
        tid = 123
        details_generate = {
            "event": "generate",
            "batch_req_ids": [1, 2, 3],
            "batch_seq_len": [1, 2, 3]
        }
        details_pullkv = {
            "event": "pullkv",
            "batch_req_ids": [1, 2, 3],
            "batch_seq_len": [1, 2, 3],
            "get_pull_size": lambda x: x
        }

        self.metrics.add_event(event_str, flag, tid, details_generate)
        self.metrics.add_event(event_str, flag, tid, details_pullkv)
        self.metrics.output()
        self.metrics.metric_enable = False
