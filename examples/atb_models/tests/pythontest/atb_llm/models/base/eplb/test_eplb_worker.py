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
from unittest.mock import patch, MagicMock
from atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker import EplbWorker


class TestDeepseekV2Config(unittest.TestCase):
    @patch("atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker.EplbRebalanceLoader")
    @patch("atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker.EplbForwarder")
    @patch("atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker.EplbPlanner")
    @patch("atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker.do_eplb")
    def test_init(self, func1, func2, func3, func4):
        modelrunner = MagicMock()
        modelrunner.model.num_redundant_experts, modelrunner.model.mapping.world_size = 16, 16
        EplbWorker(modelrunner, 0, "ds", 0)

if __name__ == '__main__':
    unittest.main()
