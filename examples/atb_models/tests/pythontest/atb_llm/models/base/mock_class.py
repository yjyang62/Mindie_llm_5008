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

from unittest.mock import MagicMock


class MockBaseLM:
    def __init__(self, **kwargs):
        self.mapping = MagicMock()
        self.mapping.rank = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def model_torch_class_name(self):
        return "FakeModelTorch"

    @classmethod
    def build_engine(cls, is_prefill):
        return MagicMock()


class MockModelTorch:
    def __init__(self, name):
        pass

    def set_param(self, param):
        pass

    def set_kv_cache(self, k_caches, v_caches):
        pass

    def set_weight(self, weight):
        pass


class MockOperationTorch:
    def __init__(self, name):
        pass

    def set_param(self, param):
        pass


class MockTorchClasses():
    ModelTorch = MagicMock()
    ModelTorch.ModelTorch = MagicMock(spec=MockModelTorch)
    OperationTorch = MagicMock()
    OperationTorch.OperationTorch = MagicMock(spec=MockOperationTorch)
    load_library = MagicMock()
