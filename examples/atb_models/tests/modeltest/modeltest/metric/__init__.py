#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import importlib

metric_map = {
    "acc": "AccMetric",
    "longbench": "LongbenchMetric",
    "pass_k": "PassKMetric", # HumanEval Task
    "truthfulqa": "TruthfulqaMetric"
}


def get_metric_cls(metric_type):
    module = importlib.import_module(f".{metric_type}", package=__name__)
    return getattr(module, metric_map.get(metric_type))