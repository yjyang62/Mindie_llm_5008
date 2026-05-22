# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import unittest
from unittest.mock import patch, MagicMock, call

import torch

from atb_llm.models.base.graph_manager.layerwise_combined_graph_wrapper import LayerwiseCombinedATBGraphWrapper
from tests.pythontest.atb_llm.models.base.graph_manager.test_graph_manager import MockATBGraphWrapper


class MockOperation:
    def __init__(self):
        pass

    def set_param(self, params):
        pass

    def set_weight(self, weights):
        pass

    def set_kv_cache(self, k_caches, v_caches):
        pass
    
    def execute(self, inputs, runtime_params):
        return 10


class TestLayerwiseCombinedATBGraphWrapper(unittest.TestCase):
    def setUp(self):
        self.graph_wrapper = LayerwiseCombinedATBGraphWrapper([MockATBGraphWrapper("A"), MockATBGraphWrapper("B")])
        self.model_type = "test_class"

        torch.classes = MagicMock()
        torch.classes.ModelTorch = MagicMock()
        torch.classes.ModelTorch.ModelTorch = MagicMock()
        torch.classes.ModelTorch.ModelTorch.return_value = MockOperation()
    

    def test_set_param(self):
        test_params = {
            'head': {},
            'tail': {}
        }
        test_weights = {
            'head': {},
            'tail': {}
        }
        test_k_caches = {
            'head': {},
            'tail': {}
        }

        test_v_caches = {
            'head': {},
            'tail': {}
        }
        self.graph_wrapper.set_param(self.model_type, test_params)
        self.graph_wrapper.set_weight(test_weights)
        self.graph_wrapper.set_kv_cache(test_k_caches, test_v_caches)
        self.graph_wrapper.atb_graph.execute = MagicMock()
        out = self.graph_wrapper.execute(None, {})
        self.graph_wrapper.atb_graph.execute.assert_called_once()