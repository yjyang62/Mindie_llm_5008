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
from unittest.mock import MagicMock, patch

from atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_planner_executer import EplbPlannerExecuter


class TestEplbPlanner(unittest.TestCase):
    def setUp(self):
        self.executor = EplbPlannerExecuter()

    def tearDown(self):
        self.executor = None

    @patch("queue.Queue.get", return_value=MagicMock())
    def test_set_load_prepare_done_and_wait(self, mock_get):
        self.executor.set_load_prepare_done_and_wait()

    def test_is_load_prepare_done(self):
        self.executor.is_load_prepare_done()

    def test_set_load_deploy_done_and_notify(self):
        self.executor.set_load_deploy_done_and_notify()


if __name__ == '__main__':
    unittest.main()
